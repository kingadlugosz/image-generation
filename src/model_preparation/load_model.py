import hydra
from src.model_preparation.dcgan import DCGan



@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def load_model(cfg):
    model = DCGan(**cfg.model)
    return model


if __name__ == "__main__":
    load_model()
