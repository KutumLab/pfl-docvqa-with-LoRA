from collections import OrderedDict
from flwr.server.strategy import FedAvg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Function to get the weights of a model
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


# Function to set the weights of a model
def set_weights(model, weights) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)