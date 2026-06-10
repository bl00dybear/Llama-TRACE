import gc
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from data_pipeline.trace_loader import TraceTaskDataset
import torch.multiprocessing as mp
try:
    from rouge_score import rouge_scorer as rouge_lib
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

logger = logging.getLogger(__name__)

def test_task(model, tokenizer, task_name, device, cfg):
    task_type = cfg.data.task_types[task_name]
    logger.info(f"Evaluating task '{task_name}' (type={task_type})")
    ds = TraceTaskDataset(
        cfg.data.root,
        task_name,
        split="test",
        max_examples=cfg.data.smoke_max_eval_examples_per_task,
    )
    logger.info(f"  Loaded {len(ds)} test examples for '{task_name}'")

    eval_bs = getattr(cfg.training, "eval_batch_size", cfg.training.batch_size)
    eval_nw = getattr(cfg.training, "eval_num_workers", cfg.training.num_workers)

    dl = DataLoader(
        ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=eval_nw,
        pin_memory=True,
        multiprocessing_context=mp.get_context("spawn") if eval_nw > 0 else None,
    )

    was_training = model.training
    model.eval()
    model.config.use_cache = True

    if task_type == "classification":
        score = _eval_classification(model, tokenizer, dl, device, cfg, task_name)
    else:
        score = _eval_generation(model, tokenizer, dl, device, cfg, task_name)

    model.config.use_cache = False
    if was_training:
        model.train()

    logger.info(f"  '{task_name}' score: {score:.4f}")
    return score

def _eval_classification(model, tokenizer, dl, device, cfg, task_name=""):
    correct = 0
    total = 0
    max_new = getattr(cfg.data, "max_new_tokens_cls", 16)
    with torch.inference_mode():
        for prompts, answers in tqdm(dl, desc=f"Eval {task_name} (cls)", leave=False):
            inputs = tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.data.max_length,
            ).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            for i, gen_ids in enumerate(out):
                prompt_len = inputs["input_ids"].shape[1]
                new_tokens = gen_ids[prompt_len:]
                pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
                gold = answers[i].strip().lower()
                if pred.startswith(gold) or gold in pred:
                    correct += 1
                total += 1
    gc.collect()
    torch.cuda.empty_cache()
    acc = correct / max(total, 1)
    logger.info(f"  Classification {task_name}: {correct}/{total} correct  (acc={acc:.4f})")
    return acc

def _eval_generation(model, tokenizer, dl, device, cfg, task_name=""):
    if not HAS_ROUGE:
        logger.warning("rouge_score not installed — returning 0.0 for generation eval")
        return 0.0
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    with torch.inference_mode():
        for prompts, answers in tqdm(dl, desc=f"Eval {task_name} (gen)", leave=False):
            inputs = tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.data.max_length,
            ).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.data.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            for i, gen_ids in enumerate(out):
                prompt_len = inputs["input_ids"].shape[1]
                new_tokens = gen_ids[prompt_len:]
                pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                gold = answers[i].strip()
                score = scorer.score(gold, pred)["rougeL"].fmeasure
                scores.append(score)
    gc.collect()
    torch.cuda.empty_cache()
    avg = sum(scores) / max(len(scores), 1)
    logger.info(f"  Generation {task_name}: {len(scores)} examples  (rougeL={avg:.4f})")
    return avg