import os
import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from engine.training_loop import run_continual_learning_pipeline

def main_worker(rank, world_size, cfg):
    run_continual_learning_pipeline(rank, world_size, cfg)

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    available_devices = torch.cuda.device_count()
    world_size = cfg.num_devices if cfg.num_devices is not None else available_devices
    
    if world_size > available_devices:
        print(f"Warning: Requested {world_size} devices but only {available_devices} available. Using {available_devices}.")
        world_size = available_devices

    if world_size > 1:
        mp.spawn(
            main_worker,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True
        )
    else:
        main_worker(0, 1, cfg)

if __name__ == "__main__":
    main()
