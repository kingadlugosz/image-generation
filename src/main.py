import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_preparation.load_data import load_data


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    load_data(cfg)


if __name__ == "__main__":
    main()
