import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        self.A = nn.Parameter(torch.empty(self.rank, self.base.in_features))
        self.B = nn.Parameter(torch.empty(self.base.out_features, self.rank))
        self.delta_bias = nn.Parameter(torch.zeros(self.base.out_features))
        
        self.register_buffer('merged_delta', torch.zeros_like(self.base.weight))
        self.register_buffer('merged_bias', torch.zeros(self.base.out_features))
        
        self.register_buffer('merged_A_past', torch.zeros(0, self.base.in_features))
        
        self.scaling = self.alpha / self.rank
        self.reset_active_adapter()
        self.to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    def reset_active_adapter(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        with torch.no_grad():
            self.delta_bias.zero_()
        self.set_trainable(True)

    def merge_active_to_frozen(self):
        with torch.no_grad():
            active_delta_W = (self.B @ self.A) * self.scaling
            self.merged_delta.add_(active_delta_W)
            if self.adapt_bias:
                self.merged_bias.add_(self.delta_bias)
            
            new_A_past = torch.cat([self.merged_A_past, self.A.data], dim=0)
            self.register_buffer('merged_A_past', new_A_past)
            
        self.set_trainable(False)

    def set_trainable(self, trainable):
        self.A.requires_grad = trainable
        self.B.requires_grad = trainable
        if self.adapt_bias:
            self.delta_bias.requires_grad = trainable
        else:
            self.delta_bias.requires_grad = False

    def forward(self, x):
        output = F.linear(x, self.base.weight, self.base.bias)
        output = output + F.linear(x, self.merged_delta, self.merged_bias)
        
        low_rank_out = (x @ self.A.T) @ self.B.T
        output = output + (low_rank_out * self.scaling)
        
        if self.adapt_bias:
            output = output + self.delta_bias

        return output

    def get_olora_penalty(self):
        if self.merged_A_past.shape[0] == 0:
            return torch.tensor(0.0, device=self.A.device, dtype=self.A.dtype)
            
        prod = torch.matmul(self.A, self.merged_A_past.T)
        penalty = torch.norm(prod, p='fro')**2
        return penalty

def inject_lora(model, device, cfg):
    for name, module in model.named_modules():
        for proj_name in cfg.adapters.target_modules:
            if hasattr(module, proj_name):
                base_linear = getattr(module, proj_name)
                if isinstance(base_linear, nn.Linear):
                    new_layer = LoRAStackLinear(base=base_linear, cfg=cfg)
                    setattr(module, proj_name, new_layer)
    model.to(device)
    return model

def get_total_olora_penalty(model):
    total_penalty = 0.0
    for m in model.modules():
        if isinstance(m, LoRAStackLinear):
            total_penalty += m.get_olora_penalty()
    return total_penalty

def prepare_task_adapter(model, is_first_task, device, cfg):
    model_to_use = model.module if hasattr(model, "module") else model
    if is_first_task:
        model_to_use = inject_lora(model_to_use, device, cfg)
    else:
        for m in model_to_use.modules():
            if isinstance(m, LoRAStackLinear):
                m.merge_active_to_frozen()
                m.reset_active_adapter()

    for p in model_to_use.parameters():
        p.requires_grad = False

    for m in model_to_use.modules():
        if isinstance(m, LoRAStackLinear):
            m.set_trainable(True)

    return model_to_use
