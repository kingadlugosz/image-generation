import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_preparation.load_data import load_data
from src.model_preparation.load_model import load_model


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dataloader = load_data(cfg)
    model = load_model(cfg)


if __name__ == "__main__":
    main()
