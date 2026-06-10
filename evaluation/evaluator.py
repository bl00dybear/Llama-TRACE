import gc
import torch
from torch.utils.data import DataLoader
import logging
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
    ds = TraceTaskDataset(
        cfg.data.root,
        task_name,
        split="test",
        max_examples=cfg.data.smoke_max_eval_examples_per_task,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        multiprocessing_context=mp.get_context("spawn") if cfg.training.num_workers > 0 else None
    )
    model.eval()
    if task_type == "classification":
        return _eval_classification(model, tokenizer, dl, device, cfg)
    else:
        return _eval_generation(model, tokenizer, dl, device, cfg)

def _eval_classification(model, tokenizer, dl, device, cfg):
    correct = 0
    total = 0
    with torch.no_grad():
        for prompts, answers in dl:
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
                pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
                gold = answers[i].strip().lower()
                if pred.startswith(gold) or gold in pred:
                    correct += 1
                total += 1
    gc.collect()
    torch.cuda.empty_cache()
    return correct / max(total, 1)

def _eval_generation(model, tokenizer, dl, device, cfg):
    if not HAS_ROUGE:
        return 0.0
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    with torch.no_grad():
        for prompts, answers in dl:
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
    return sum(scores) / max(len(scores), 1)