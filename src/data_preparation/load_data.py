import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import hydra


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def load_data(cfg):
    dataset = datasets.ImageFolder(cfg.paths.dataset + cfg.dataset.name)
    dataset[476][0].show()


if __name__ == "__main__":
    load_data()
