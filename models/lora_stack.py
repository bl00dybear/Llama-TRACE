import torch
import torch.nn as nn
import logging
import math

logger = logging.getLogger(__name__)

class LoRAStackLinear(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        self.base = base
        self.rank = cfg.adapters.rank
        self.alpha = cfg.adapters.alpha
        self.adapt_bias = cfg.adapters.adapt_bias
        
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        self.delta_bias_list = nn.ParameterList()
        self.scalings = []
        
        self.add_adapter(trainable=True)

    def add_adapter(self, trainable=True):
        param_kwargs = {
            "device": self.base.weight.device,
            "dtype": self.base.weight.dtype,
        }
        
        A = nn.Parameter(torch.empty(self.rank, self.base.in_features, **param_kwargs))
        B = nn.Parameter(torch.empty(self.base.out_features, self.rank, **param_kwargs))
        
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        nn.init.zeros_(B)
        
        delta_bias = nn.Parameter(torch.zeros(self.base.out_features, **param_kwargs))
        
        self.A_list.append(A)
        self.B_list.append(B)
        self.delta_bias_list.append(delta_bias)
        self.scalings.append(self.alpha / self.rank)
        
        idx = len(self.A_list) - 1
        self.set_adapter_trainable(idx, trainable)
        
        return idx

    def set_adapter_trainable(self, idx, trainable):
        self.A_list[idx].requires_grad = trainable
        self.B_list[idx].requires_grad = trainable
        
        if self.adapt_bias:
            self.delta_bias_list[idx].requires_grad = trainable
        else:
            self.delta_bias_list[idx].requires_grad = False

    def freeze_all_adapters(self):
        for i in range(len(self.A_list)):
            self.set_adapter_trainable(i, False)

    def forward(self, x):
        output = self.base(x)
        
        for i in range(len(self.A_list)):
            lora_out = (x @ self.A_list[i].T) @ self.B_list[i].T
            output = output + (lora_out * self.scalings[i])
            
            if self.adapt_bias:
                output = output + self.delta_bias_list[i].to(dtype=output.dtype)
                
        return output

def _get_target_modules(model, cfg):
    modules = {}
    for name, module in model.named_modules():
        for proj_name in cfg.adapters.target_modules:
            if hasattr(module, proj_name):
                modules[f"{name}.{proj_name}"] = getattr(module, proj_name)
    return modules

def inject_lora(model, device, cfg):
    has_lora = any(isinstance(m, LoRAStackLinear) for m in model.modules())
    if has_lora:
        return model
        
    for name, module in model.named_modules():
        for proj_name in cfg.adapters.target_modules:
            if hasattr(module, proj_name):
                base_linear = getattr(module, proj_name)
                if isinstance(base_linear, nn.Linear):
                    setattr(
                        module,
                        proj_name,
                        LoRAStackLinear(base=base_linear, cfg=cfg)
                    )
            
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
        
    for m in model.modules():
        if isinstance(m, LoRAStackLinear):
            m.freeze_all_adapters()
            m.set_adapter_trainable(len(m.A_list) - 1, True)
            
    return model

def add_task_adapter(model, cfg):
    for m in model.modules():
        if isinstance(m, LoRAStackLinear):
            m.freeze_all_adapters()
            new_idx = m.add_adapter(trainable=True)
            m.set_adapter_trainable(new_idx, True)
    return model

def prepare_task_adapter(model, is_first_task, device, cfg):
    if is_first_task:
        model = inject_lora(model, device, cfg)
    else:
        model = add_task_adapter(model, cfg)
    return model
