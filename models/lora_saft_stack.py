import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from tqdm import tqdm
from data_pipeline.trace_loader import TraceTaskDataset, CollateFunction
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

class LoRAStackLinear(nn.Module):
    def __init__(self, base, cfg, initial_mask=None):
        super().__init__()
        self.base = base
        self.rank = cfg.adapters.rank
        self.alpha = cfg.adapters.alpha
        self.adapt_bias = cfg.adapters.adapt_bias
        
        self.A = nn.Parameter(torch.empty(self.rank, self.base.in_features))
        self.B = nn.Parameter(torch.empty(self.base.out_features, self.rank))
        self.delta_bias = nn.Parameter(torch.zeros(self.base.out_features))
        
        self.register_buffer('active_mask', torch.ones_like(self.base.weight))
        
        self.register_buffer('merged_delta', torch.zeros_like(self.base.weight))
        self.register_buffer('merged_bias', torch.zeros(self.base.out_features))
        
        self.scaling = self.alpha / self.rank
        
        self.reset_active_adapter(initial_mask)
        self.to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    def reset_active_adapter(self, binary_mask=None):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        
        with torch.no_grad():
            self.delta_bias.zero_()
            if binary_mask is None:
                self.active_mask.fill_(1.0)
            else:
                self.active_mask.copy_(binary_mask)
        
        self.set_trainable(True)

    def merge_active_to_frozen(self):
        with torch.no_grad():
            active_delta_W = (self.B @ self.A) * self.active_mask * self.scaling
            self.merged_delta.add_(active_delta_W)
            
            if self.adapt_bias:
                self.merged_bias.add_(self.delta_bias)
        
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

        active_delta_W = (self.B @ self.A) * self.active_mask
        output = output + F.linear(x, active_delta_W, self.delta_bias) * self.scaling
        
        return output

def inject_lora(model, binary_masks, device, cfg):
    for name, module in model.named_modules():
        for proj_name in cfg.adapters.target_modules:
            if hasattr(module, proj_name):
                full_name = f"{name}.{proj_name}"
                if full_name in binary_masks:
                    base_linear = getattr(module, proj_name)
                    if isinstance(base_linear, nn.Linear):
                        new_layer = LoRAStackLinear(
                            base=base_linear,
                            cfg=cfg,
                            initial_mask=binary_masks[full_name]
                        )
                        setattr(module, proj_name, new_layer)
    model.to(device)
    return model

def _get_target_weight(module):
    if isinstance(module, LoRAStackLinear):
        return module.base.weight
    return module.weight

def accumulate_gradients_task(model, tokenizer, task_name, device, cfg):
    train_ds = TraceTaskDataset(cfg.data.root, task_name, split="train", max_examples=cfg.data.smoke_max_train_examples_per_task)
    collate_fn = CollateFunction(tokenizer, cfg.data.max_length)
        
    train_dl = DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers, 
        shuffle=True, 
        collate_fn=collate_fn,
        multiprocessing_context=mp.get_context("spawn") if cfg.training.num_workers > 0 else None
    )

    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.requires_grad = False
        param.grad = None

    target_weights_dict = {}
    for name, module in model.named_modules():
        for proj_name in cfg.adapters.target_modules:
            if hasattr(module, proj_name):
                full_name = f"{name}.{proj_name}"
                layer = getattr(module, proj_name)
                tw = _get_target_weight(layer)
                tw.requires_grad = True
                target_weights_dict[full_name] = tw
    
    if not target_weights_dict:
        logger.warning("No target modules found for gradient accumulation!")

    model.train()
    pbar = tqdm(train_dl, desc=f"SAFT-grads {task_name}", leave=True)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if not torch.isnan(loss):
            loss.backward()
        del input_ids, attention_mask, labels, outputs, loss

    return model, target_weights_dict

def find_threshold(target_weights_dict, cfg):
    abs_grads = []
    for name in sorted(target_weights_dict.keys()):
        tw = target_weights_dict[name]
        if tw.grad is not None:
            abs_grads.append(tw.grad.abs().view(-1))

    if not abs_grads:
        return 0.0

    all_abs_grads = torch.cat(abs_grads)
    sparsity = getattr(cfg.adapters, "sparsity", 0.9)
    k = int(sparsity * all_abs_grads.numel())
    k = max(0, min(k, all_abs_grads.numel() - 1))
    threshold = torch.kthvalue(all_abs_grads, k + 1).values
    del all_abs_grads
    return threshold

def compute_binary_masks(target_weights_dict, threshold, cfg):
    binary_masks = {}
    for name in sorted(target_weights_dict.keys()):
        tw = target_weights_dict[name]
        if tw.grad is not None:
            mask = (tw.grad.abs() > threshold).float().detach()
        else:
            mask = torch.ones_like(tw, dtype=torch.float32)
        binary_masks[name] = mask
    return binary_masks

def prepare_task_adapter(model, is_first_task, device, task_list, cfg):
    task_name = task_list[0]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not is_first_task:
        for m in model.modules():
            if isinstance(m, LoRAStackLinear):
                m.merge_active_to_frozen()

    model, target_weights_dict = accumulate_gradients_task(model, tokenizer, task_name, device, cfg)

    threshold = find_threshold(target_weights_dict, cfg)
    binary_masks = compute_binary_masks(target_weights_dict, threshold, cfg)

    if is_first_task:
        model = inject_lora(model, binary_masks, device, cfg)
    else:
        for name, m in model.named_modules():
            if isinstance(m, LoRAStackLinear):
                m.reset_active_adapter(binary_masks.get(f"{name}", None))

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Set only active adapter parameters to trainable
    for m in model.modules():
        if isinstance(m, LoRAStackLinear):
            m.set_trainable(True)

    for p in model.parameters():
        p.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    
    return model
