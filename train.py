import copy
import json
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
from collections import OrderedDict
from communication.log_communication import log_communication

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.PFL_DocVQA import collate_fn

from tqdm import tqdm
from build_utils import (build_dataset, build_model, build_optimizer, build_provider_dataset, build_lora_model)
from differential_privacy.dp_utils import (add_dp_noise, clip_parameters, flatten_params, get_shape, reconstruct_shape)
from eval import evaluate, fl_centralized_evaluation, fl_train
from logger import Logger
from metrics import Evaluator
from checkpoint import save_model
from utils import load_config, parse_args, seed_everything
from utils_parallel import get_parameters_from_model, set_parameters_model, weighted_average, get_lora_parameters, set_lora_parameters
from collections import OrderedDict
import torch.multiprocessing as mp 
import time
import wandb
from torch.utils.data import SubsetRandomSampler, random_split


class FlowerClient(fl.client.NumPyClient):
    # def __init__(self, model, trainloader, valloader, optimizer, lr_scheduler, evaluator, logger, config, client_id):
    def __init__(self,  config, client_id):
        # self.model = model
        #self.trainloader = trainloader
        #self.valloader = valloader
        # self.optimizer = optimizer
        # # self.lr_scheduler = lr_scheduler
        self.evaluator = Evaluator(case_sensitive=False)
        self.logger = Logger(config, client_id)
        # self.logger.log_model_parameters(self.model)
        self.config = config
        self.client_id = client_id
        

    def fit(self, parameters, config):
        
        log_communication(federated_round=config.current_round, sender=-1, receiver=self.client_id, data=parameters, log_location=self.logger.comms_log_file)
        

        if config.use_dp:
            # Pick a subset of providers
            provider_to_doc = json.load(open(config.provider_docs, 'r'))
            provider_to_doc = provider_to_doc["client_" + self.client_id]
            providers = random.sample(list(provider_to_doc.keys()), k=config.dp_params.providers_per_fl_round)  # 50
            train_datasets = [build_provider_dataset(config, 'train', provider_to_doc, provider, self.client_id) for provider in tqdm(providers, desc="Preparing Train Data Loader: ")]
            
        else:
            train_datasets = [build_dataset(config, 'train', client_id=self.client_id)]

        train_data_loaders = [DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn) for train_dataset in train_datasets]


        
        #self.set_parameters(self.model, parameters, config)
        # Prepare multiprocess
        
        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
    
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process 
        p = mp.Process(target=fl_train, args=( train_data_loaders, parameters,self.logger,self.evaluator, self.client_id, config, return_dict))#self.trainloader, self.model, self.optimizer, self.evaluator, self.logger,
       
        # Start the process
        
        p.start()
        # Wait for it to end
        p.join()
        # Close it
        try:
            p.close()
        except ValueError as e:
            print(f"Coudln't close the training process: {e}")
        # Get the return values

        new_parameters = return_dict["parameters"]

        log_communication(federated_round=config.current_round, sender=self.client_id, receiver=-1, data=new_parameters, log_location=self.logger.comms_log_file)

        #data_size = return_dict["data_size"]
        # Del everything related to multiprocessing
        del (manager, return_dict, p)
        return new_parameters, 1, {}
    
        # self.set_parameters(self.model, parameters, config)
        # # updated_weigths = fl_train(self.trainloader, self.model, self.optimizer, self.lr_scheduler, self.evaluator, self.logger, self.client_id, config)
        # updated_weigths = fl_train(self.trainloader, self.model, self.optimizer, self.evaluator, self.logger, self.client_id, config)
        #return updated_weigths, 1, {}  # TODO 1 ==> Number of selected clients.

    def set_parameters(self, model, parameters, config):
        set_lora_parameters(model, parameters)  # Use standard set_parameters model function.
        # params_dict = zip(model.model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        log_communication(federated_round=config.current_round, sender=-1, receiver=self.client_id, data=parameters, log_location=self.logger.comms_log_file)

        # model.model.load_state_dict(state_dict, strict=True)

    # The `evaluate` function will be called by Flower after every round
    def evaluate(self, parameters, config):

        # Create validation data loader
        val_dataset = build_dataset(config, 'val')
        val_dataset,_= random_split(val_dataset, [0.10, 0.9])
        val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        #logger = Logger(config=config)
        # Prepare multiprocess

        manager = mp.Manager()
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process 
        p = mp.Process(target=evaluate, args=(val_data_loader,parameters, self.evaluator, config, return_dict))#self.trainloader, self.model, self.optimizer, self.evaluator, self.logger,
        # Start the process
        p.start()
        # Wait for it to end
        p.join()
        # Close it
        try:
            p.close()
        except ValueError as e:
            print(f"Coudln't close the training process: {e}")

        # set_parameters_model(self.model, parameters)
        # accuracy, anls, _, _ = evaluate(self.valloader, parameters, self.evaluator, config)  # data_loader, model, evaluator, **kwargs
        is_updated = self.evaluator.update_global_metrics(return_dict['accuracy'], return_dict['total_anls'], 0)
        self.logger.log_val_metrics(return_dict['accuracy'], return_dict['total_anls'], update_best=is_updated)
        #save_model(model, config.current_round, update_best=is_updated, kwargs=config)

        #logger.logger.finish()

        return float(0), len(val_data_loader), {"accuracy": float(return_dict['accuracy']), "anls": return_dict['total_anls']}


def client_fn(client_id):
    """Create a Flower client representing a single organization."""
    # Create a list of train data loaders with one dataloader per provider

    # if config.use_dp:
    #     # Pick a subset of providers
        
    #     provider_to_doc = json.load(open(config.provider_docs, 'r'))
    #     provider_to_doc = provider_to_doc["client_" + client_id]
    #     providers = random.sample(list(provider_to_doc.keys()), k=config.dp_params.providers_per_fl_round)  # 50
    #     train_datasets = [build_provider_dataset(config, 'train', provider_to_doc, provider, client_id) for provider in providers]
        
        
    # else:
    #     train_datasets = [build_dataset(config, 'train', client_id=client_id)]

    # train_data_loaders = [DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn) for train_dataset in train_datasets]
    # total_training_steps = sum([len(data_loader) for data_loader in train_data_loaders])

    # # Create validation data loader
    # val_dataset = build_dataset(config, 'val')
    # val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # # lr_scheduler disabled due to malfunction in FL setup.
    # # optimizer, lr_scheduler = build_optimizer(model, length_train_loader=total_training_steps, config=config)
    # optimizer = build_optimizer(model, config=config)
    #evaluator = Evaluator(case_sensitive=False)
    #logger = Logger(config=config)

    return FlowerClient( config, client_id )#train_data_loaders, val_data_loader,   logger, optimizer, evaluator,


def get_config_fn():
    """Return a function which returns custom configuration."""

    def custom_config(server_round: int):
        """Return evaluate configuration dict for each round."""
        config.current_round = server_round
        return config

    return custom_config


if __name__ == '__main__':
    #wandb.require("service")

    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)
    print(config)
    time.sleep(15)
    # Set `MASTER_ADDR` and `MASTER_PORT` environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9957'

    model = build_lora_model(config)
    params = get_lora_parameters(model)
    del model
    logger=None#Logger(config)
    
    #model.share_memory_()
    #params.share_memory()

    

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit=config.dp_params.client_sampling_probability,  # Sample 100% of available clients for training
        fraction_fit=config.fl_params.sample_clients/config.fl_params.total_clients,
        # fraction_evaluate=config.fl_params.sample_clients/config.fl_params.total_clients,  # Sample N of available clients for evaluation
        fraction_evaluate=0.1,  # Sample only 1 client for evaluation
        min_fit_clients=config.fl_params.sample_clients,  # Never sample less than N clients for training
        # min_evaluate_clients=config.fl_params.sample_clients,  # Never sample less than N clients for evaluation
        min_evaluate_clients=1,  # Sample only 1 client for evaluation.
        min_available_clients=config.fl_params.sample_clients,  # Wait until N clients are available
        fit_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        on_fit_config_fn=get_config_fn(),  # Log path hardcoded according to /save dir
        #evaluate_fn=fl_centralized_evaluation,  # Pass the centralized evaluation function
        on_evaluate_config_fn=get_config_fn(),
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if config.device == "cuda":
        client_resources = {"num_gpus": 1}  # TODO Check number of GPUs

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.fl_params.total_clients,
        config=fl.server.ServerConfig(num_rounds=config.fl_params.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        # ray_init_args={"local_mode": True}  # run in one process to avoid zombie ray processes
    )
