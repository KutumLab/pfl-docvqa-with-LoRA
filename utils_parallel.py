import torch
from collections import OrderedDict
from peft import get_peft_model_state_dict,set_peft_model_state_dict


""" Deprecated functions: They will send and load all the model's weights.
def get_parameters_from_model(model):
    return [val.cpu().numpy() for _, val in model.model.state_dict().items()]

def set_parameters_model(model, parameters):
    params_dict = zip(model.model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.model.load_state_dict(state_dict, strict=True)
"""


def get_lora_parameters(model):
    # state_dict= get_peft_model_state_dict(model.model)
    # np_state_dict= {key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in state_dict.items()}
    # return np_state_dict
    keyed_parameters = {n: p.requires_grad for n, p in model.model.named_parameters()}
    frozen_parameters = [not keyed_parameters[n] if n in keyed_parameters else False for n, p in model.model.state_dict().items()]
    return [val.cpu().numpy() for val, is_frozen in zip(model.model.state_dict().values(), frozen_parameters) if not is_frozen]

def set_lora_parameters(model, parameters):
    # set_peft_model_state_dict(model.model, parameters)
    keyed_parameters = {n: p.requires_grad for n, p in model.model.named_parameters()}
    frozen_parameters = [not keyed_parameters[n] if n in keyed_parameters else False for n, p in model.model.state_dict().items()]

    i = 0
    params_dict = model.model.state_dict()
    for key, is_frozen in zip(model.model.state_dict().keys(), frozen_parameters):

        # Update state dict with new params.
        if not is_frozen:
            params_dict[key] = torch.Tensor(parameters[i])
            i += 1

    model.model.load_state_dict(params_dict, strict=True)

    

# New functions. They will send and load only weights of NON-frozen layers.
def get_parameters_from_model(model):
    keyed_parameters = {n: p.requires_grad for n, p in model.model.named_parameters()}
    frozen_parameters = [not keyed_parameters[n] if n in keyed_parameters else False for n, p in model.model.state_dict().items()]
    return [val.cpu().numpy() for val, is_frozen in zip(model.model.state_dict().values(), frozen_parameters) if not is_frozen]


def set_parameters_model(model, parameters):
    keyed_parameters = {n: p.requires_grad for n, p in model.model.named_parameters()}
    frozen_parameters = [not keyed_parameters[n] if n in keyed_parameters else False for n, p in model.model.state_dict().items()]

    i = 0
    params_dict = model.model.state_dict()
    for key, is_frozen in zip(model.model.state_dict().keys(), frozen_parameters):

        # Update state dict with new params.
        if not is_frozen:
            params_dict[key] = torch.Tensor(parameters[i])
            i += 1

    model.model.load_state_dict(params_dict, strict=True)


def weighted_average(metrics_dict):
    metrics = list(metrics_dict[0][1].keys())
    aggregated_metrics_dict = {}
    dataset_length = sum([num_samples for num_samples, _ in metrics_dict])
    for metric in metrics:
        aggregated_metrics_dict[metric] = sum(
            [m[metric] * num_examples / dataset_length for num_examples, m in metrics_dict])

    return aggregated_metrics_dict
