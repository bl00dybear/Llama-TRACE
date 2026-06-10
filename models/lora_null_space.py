import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

class NullProjectedLoRAStackLinear(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        self.base = base
        self.rank = cfg.adapters.rank
        self.alpha = cfg.adapters.alpha
        self.adapt_bias = cfg.adapters.adapt_bias
        self.null_space_rank = cfg.adapters.null_space_rank
        self.svd_eps = cfg.adapters.svd_eps
        
        self.A = nn.Parameter(torch.empty(self.rank, self.base.in_features))
        self.B = nn.Parameter(torch.empty(self.base.out_features, self.rank))
        self.delta_bias = nn.Parameter(torch.zeros(self.base.out_features))
        
        self.register_buffer('merged_delta', torch.zeros_like(self.base.weight))
        self.register_buffer('merged_bias', torch.zeros(self.base.out_features))
        
        self.register_buffer('U_basis', torch.zeros(self.base.out_features, 0))
        self.register_buffer('V_basis', torch.zeros(self.base.in_features, 0))
        
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
        self.set_trainable(False)

    def update_null_space_projector(self):
        with torch.no_grad():
            ref = (self.base.weight.data + self.merged_delta).float()
            
            try:
                U, S, Vh = torch.linalg.svd(ref, full_matrices=False)
                
                s_max = S.max()
                if s_max > 0:
                    if self.null_space_rank is not None:
                        r = min(int(self.null_space_rank), int(S.numel()))
                    else:
                        r = int((S > (self.svd_eps * s_max)).sum().item())
                    
                    if r > 0:
                        self.register_buffer('U_basis', U[:, :r].to(self.base.weight.dtype))
                        self.register_buffer('V_basis', Vh[:r, :].T.to(self.base.weight.dtype))
                        logger.info(f"Updated Null-Space Projector for {r} principal components")
                    else:
                        self.register_buffer('U_basis', torch.zeros(self.base.out_features, 0, device=ref.device, dtype=self.base.weight.dtype))
                        self.register_buffer('V_basis', torch.zeros(self.base.in_features, 0, device=ref.device, dtype=self.base.weight.dtype))
            except Exception as e:
                logger.warning(f"SVD failed in Null-Space update: {e}. Keeping previous projector.")

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
        
        if self.V_basis.shape[1] > 0:
            x_proj = x - (x @ self.V_basis) @ self.V_basis.T
        else:
            x_proj = x
            
        lora_out = (x_proj @ self.A.T) @ self.B.T
        
        if self.U_basis.shape[1] > 0:
            lora_out = lora_out - (lora_out @ self.U_basis) @ self.U_basis.T
            
        output = output + (lora_out * self.scaling)
        
        if self.adapt_bias:
            output = output + self.delta_bias

        return output

def inject_lora(model, device, cfg):
    for name, module in model.named_modules():
        for proj_name in cfg.adapters.target_modules:
            if hasattr(module, proj_name):
                base_linear = getattr(module, proj_name)
                if isinstance(base_linear, nn.Linear):
                    new_layer = NullProjectedLoRAStackLinear(base=base_linear, cfg=cfg)
                    setattr(module, proj_name, new_layer)
    model.to(device)
    return model

def prepare_task_adapter(model, is_first_task, device, cfg):
    model_to_use = model.module if hasattr(model, "module") else model

    if is_first_task:
        model_to_use = inject_lora(model_to_use, device, cfg)
    else:
        for m in model_to_use.modules():
            if isinstance(m, NullProjectedLoRAStackLinear):
                m.merge_active_to_frozen()
                m.update_null_space_projector()
                m.reset_active_adapter()
                
    return model_to_use
