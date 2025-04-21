import yaml
import torch as t
from easydict import EasyDict as ed

def getConfig(config_path="./configs/config.yaml"):
    config = {}
    with open(config_path, "r") as cfgFile:
        try:
            config = yaml.safe_load(cfgFile)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def getTestConfig():
    return getConfig("./configs/test_config.yaml")

def load_fins_config(file_path):
    with open(file_path, encoding="utf-8") as f:
        contents = yaml.load(f, Loader=yaml.FullLoader)
    return ed(contents)
