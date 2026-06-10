import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
from omegaconf import DictConfig
from engine.training_loop import run_continual_learning_pipeline

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    run_continual_learning_pipeline(cfg)

if __name__ == "__main__":
    main()
