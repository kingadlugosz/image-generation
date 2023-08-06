import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def load_data(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = datasets.ImageFolder(cfg.paths.dataset + cfg.dataset.name, transform=transform)

    dataloader = DataLoader(dataset, **cfg.dataset.data_loader)

    return dataloader


if __name__ == "__main__":
    load_data()
