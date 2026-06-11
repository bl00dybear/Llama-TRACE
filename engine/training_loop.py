import os
import gc
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from heavyball import utils as heavyball_utils
heavyball_utils.compile_mode = None

from data_pipeline.trace_loader import TraceTaskDataset, CollateFunction
from evaluation.evaluator import test_task
from evaluation.metrics_calculator import calculate_fwt, calculate_bwt, calculate_op, plot_results_matrix, log_heatmap_to_wandb

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)

def get_llm(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    generation_config = model.generation_config
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.typical_p = None
    return model, tokenizer

def train_task_loop(model, tokenizer, task_name, device, cfg):
    train_ds = TraceTaskDataset(
        cfg.data.root, 
        task_name, 
        split="train", 
        max_examples=cfg.data.smoke_max_train_examples_per_task
    )
    collate_fn = CollateFunction(tokenizer, cfg.data.max_length)
        
    train_dl = DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers, 
        shuffle=True, 
        pin_memory=True, 
        persistent_workers=(cfg.training.num_workers > 0), 
        prefetch_factor=4 if cfg.training.num_workers > 0 else None, 
        collate_fn=collate_fn,
        multiprocessing_context=mp.get_context("spawn") if cfg.training.num_workers > 0 else None
    )
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if "Muon" in cfg.optimizer._target_:
        trainable_params = [p for p in trainable_params if p.ndim >= 2]

    opt_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    opt_cfg.pop("optimizer_name", None)
    optimizer = hydra.utils.instantiate(opt_cfg, params=trainable_params)
    
    model.train()
    history = {"epoch": [], "train_loss": []}
    grad_accum_steps = max(1, cfg.training.grad_accum_steps)
    
    adapter_method = cfg.adapters.method
    ella_lambda = getattr(cfg.adapters, "ella_lambda", 0.0) if adapter_method == "ella" else 0.0
    olora_lambda = getattr(cfg.adapters, "olora_lambda", 0.0) if adapter_method == "olora" else 0.0
    
    for epoch in range(cfg.training.epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_total = 0.0
        num_batches = len(train_dl)
        
        pbar = tqdm(
            train_dl, 
            desc=f"{task_name} Epoch {epoch+1}/{cfg.training.epochs}", 
            leave=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            n_valid = (labels != -100).sum().item()
            
            if n_valid == 0 or torch.isnan(loss):
                loss = outputs.logits.sum() * 0.0
                valid_to_add = 0
            else:
                valid_to_add = n_valid
            
            if adapter_method == "ella" and ella_lambda > 0:
                from models.lora_ella import get_total_ella_penalty
                loss = loss + ella_lambda * get_total_ella_penalty(model)
            
            if adapter_method == "olora" and olora_lambda > 0:
                from models.olora import get_total_olora_penalty
                loss = loss + olora_lambda * get_total_olora_penalty(model)
            
            loss = loss / grad_accum_steps
            epoch_loss_sum += loss.item() * valid_to_add * grad_accum_steps
            epoch_total += valid_to_add
            
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            del input_ids, attention_mask, labels, outputs, loss
            
        total = max(epoch_total, 1.0)
        epoch_loss = epoch_loss_sum / total
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)

        if wandb is not None and wandb.run is not None:
            wandb.log({
                f"train/{task_name}/loss": epoch_loss,
                f"train/{task_name}/epoch": epoch + 1,
            })

        gc.collect()
        torch.cuda.empty_cache()
        
    return model, history

def run_continual_learning_pipeline(cfg):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if wandb is not None and cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=list(cfg.wandb.get("tags", [])),
            reinit=True,
        )
    
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    model, tokenizer = get_llm(cfg)
    model = model.to(device)
    
    num_tasks = len(cfg.data.tasks)
    results_matrix = [[0.0] * num_tasks for _ in range(num_tasks)]
    baselines = [0.0] * num_tasks
    fwt_per_task = [None] * num_tasks

    # logger.info("=" * 60)
    # logger.info("ZERO-SHOT BASELINE EVALUATION")
    # logger.info("=" * 60)
    # for eval_id, eval_task in enumerate(cfg.data.tasks):
    #     score = test_task(model, tokenizer, eval_task, device, cfg)
    #     baselines[eval_id] = score
    #     if wandb is not None and wandb.run is not None:
    #         wandb.log({f"baseline/{eval_task}": score})
    # logger.info("Baselines: %s", {t: f"{b:.4f}" for t, b in zip(cfg.data.tasks, baselines)})

    for task_id, task_name in enumerate(cfg.data.tasks):
        logger.info("=" * 60)
        logger.info("TASK %d/%d: %s", task_id + 1, num_tasks, task_name)
        logger.info("=" * 60)
        adapter_method = cfg.adapters.method
        logger.info("Preparing adapter: method=%s", adapter_method)
        
        if adapter_method == "saft":
            from models.lora_saft_stack import prepare_task_adapter
            model = prepare_task_adapter(
                model=model,
                is_first_task=(task_id == 0),
                device=device,
                task_list=[task_name],
                cfg=cfg
            )
        elif adapter_method == "ella":
            from models.lora_ella import prepare_task_adapter
            model = prepare_task_adapter(
                model=model,
                is_first_task=(task_id == 0),
                device=device,
                cfg=cfg
            )
        elif adapter_method == "olora":
            from models.olora import prepare_task_adapter
            model = prepare_task_adapter(
                model=model,
                is_first_task=(task_id == 0),
                device=device,
                cfg=cfg
            )
        elif adapter_method == "null_space":
            from models.lora_null_space import prepare_task_adapter
            model = prepare_task_adapter(
                model=model,
                is_first_task=(task_id == 0),
                device=device,
                cfg=cfg
            )
        else:
            from models.lora_stack import prepare_task_adapter
            model = prepare_task_adapter(
                model=model, 
                is_first_task=(task_id == 0), 
                device=device,
                cfg=cfg
            )
        logger.info("Adapter ready — starting training on '%s'", task_name)

        model, train_history = train_task_loop(
            model=model,
            tokenizer=tokenizer,
            task_name=task_name,
            device=device,
            cfg=cfg
        )
        
        logger.info("Training done for '%s' — final loss: %.6f", task_name, train_history["train_loss"][-1])

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("--- Evaluating all tasks after training T%d (%s) ---", task_id + 1, task_name)
        for eval_id, eval_task in enumerate(cfg.data.tasks):
            score = test_task(model, tokenizer, eval_task, device, cfg)
            results_matrix[task_id][eval_id] = score
            if wandb is not None and wandb.run is not None:
                wandb.log({f"eval/{eval_task}/after_T{task_id+1}": score})
                
        if task_id > 0:
            fwt_per_task[task_id] = calculate_fwt(baselines, results_matrix, num_tasks=task_id + 1)
            if wandb is not None and wandb.run is not None:
                wandb.log({f"metrics/fwt_after_T{task_id+1}": fwt_per_task[task_id]})
        if task_id == num_tasks - 1:
            bwt = calculate_bwt(results_matrix, num_tasks)
            op = calculate_op(results_matrix, num_tasks)
            logger.info("=" * 60)
            logger.info("FINAL CL METRICS  —  BWT=%.4f  |  OP=%.4f", bwt, op)
            logger.info("=" * 60)
            if wandb is not None and wandb.run is not None:
                wandb.log({"metrics/bwt": bwt, "metrics/op": op})
        if cfg.save_plots:
            plot_results_matrix(
                results_matrix=results_matrix,
                output_dir=cfg.plots_dir,
                cfg=cfg,
                baselines=baselines,
                fwt_per_task=fwt_per_task,
            )
        log_heatmap_to_wandb(
            results_matrix=results_matrix[:task_id+1],
            cfg=cfg,
            baselines=baselines,
            fwt_per_task=fwt_per_task,
            step_label=f"After T{task_id+1}",
        )

        gc.collect()
        torch.cuda.empty_cache()

    if wandb is not None and wandb.run is not None:
        wandb.finish()
