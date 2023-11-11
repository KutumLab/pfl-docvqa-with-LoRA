import os, time
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.PFL_DocVQA import collate_fn

from logger import Logger
from metrics import Evaluator
from utils import parse_args, time_stamp_to_hhmmss, load_config, save_json
from build_utils import build_model, build_dataset, build_lora_model, build_optimizer
from checkpoint import save_model

import flwr as fl
from utils_parallel import get_parameters_from_model, set_parameters_model, weighted_average, get_lora_parameters, set_lora_parameters
from differential_privacy.dp_utils import (add_dp_noise, clip_parameters, flatten_params, get_shape, reconstruct_shape)
from utils import load_config, parse_args
from collections import OrderedDict
import copy
import random

def fl_train(data_loaders, parameters, logger,evaluator, client_id, fl_config,return_dict):
    """
    Trains and returns the updated weights.
    """
    config=fl_config
   
    
    model=build_lora_model(config)
    set_lora_parameters(model, parameters)
    logger.log_model_parameters(model)

    optimizer = build_optimizer(model, config=config)


    model.model.train()
    param_keys = list(model.model.state_dict().keys())
    parameters = copy.deepcopy(list(model.model.state_dict().values()))

    keyed_parameters = {n: p.requires_grad for n, p in model.model.named_parameters()}
    frozen_parameters = [not keyed_parameters[n] if n in keyed_parameters else False for n, p in model.model.state_dict().items()]

    #logger.current_epoch += 1

    agg_update = None
    if not config.use_dp and len(data_loaders) > 1:
        raise ValueError("Non private training should only use one data loader.")

    total_training_steps = sum([len(data_loader) for data_loader in data_loaders]) * config.fl_params.iterations_per_fl_round
    total_training_samples = sum([len(data_loader.dataset) for data_loader in data_loaders]) * config.fl_params.iterations_per_fl_round
    pbar = tqdm(total=total_training_steps)

    total_loss = 0
    fl_round_acc = 0
    fl_round_anls = 0
    n=0
    for provider_dataloader in data_loaders:
        # Set model weights to state of beginning of federated round
        state_dict = OrderedDict({k: v for k, v in zip(param_keys, parameters)})
        model.model.load_state_dict(state_dict, strict=True)
        model.model.train()

        # Reset the optimizer
        if config.use_dp:
            optimizer = build_optimizer(model, config)

        # Perform N provider iterations (each provider has their own dataloader in the non-private case)
        for iter in range(config.fl_params.iterations_per_fl_round):
            for batch_idx, batch in enumerate(provider_dataloader):

                gt_answers = batch['answers']
                outputs, pred_answers, answer_conf = model.forward(batch, return_pred_answer=True)
                loss = outputs.loss

                # total_loss += loss.item() / len(batch['question_id'])
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()

                metric = evaluator.get_metrics(gt_answers, pred_answers)

                total_loss += outputs.loss.item()
                fl_round_acc += np.sum(metric['accuracy'])
                fl_round_anls += np.sum(metric['anls'])
                n+=len(metric["anls"])
                log_dict = {
                    'Train/Batch loss': outputs.loss.item(),
                    'Train/Batch Accuracy': np.mean(metric['accuracy']),
                    'Train/Batch ANLS': np.mean(metric['anls']),
                    'lr': optimizer.param_groups[0]['lr']
                }

                logger.logger.log(log_dict)
                pbar.set_postfix({"loss":total_loss/n,'accuracy':fl_round_acc/n, "ANLS":fl_round_anls/n})
                pbar.update()
                

        # After all the iterations:
        # Get the update
        new_update = [w - w_0 for w, w_0 in zip(list(model.model.state_dict().values()), parameters)]  # Get model update

        if config.use_dp:
            # flatten update
            shapes = get_shape(new_update)
            new_update = flatten_params(new_update)

            # clip update:
            new_update = clip_parameters(new_update, clip_norm=config.dp_params.sensitivity)

            # Aggregate (Avg)
            if agg_update is None:
                agg_update = new_update
            else:
                agg_update += new_update
        
    # Handle DP after all updates are done
    if config.use_dp:
        # Add the noise
        agg_update = add_dp_noise(agg_update, noise_multiplier=config.dp_params.noise_multiplier, sensitivity=config.dp_params.sensitivity)

        # Divide the noisy aggregated update by the number of providers (Average update).
        agg_update = torch.div(agg_update, len(data_loaders))

        # Add the noisy update to the original model
        agg_update = reconstruct_shape(agg_update, shapes)

        # Restore original weights (without noise) from frozen layers.
        agg_update = [upd if not is_frozen else 0 for upd, params, is_frozen in zip(agg_update, parameters, frozen_parameters)]

        # all([torch.all(params == new_params).item() == is_frozen for params, new_params, is_frozen in zip(parameters, agg_update, frozen_parameters)])  Restoration Test

    else:
        agg_update = new_update

    # upd_weights = [torch.add(agg_upd, w_0).cpu() for agg_upd, w_0 in zip(agg_update, copy.deepcopy(parameters))]  # Send all weights
    upd_weights = [torch.add(agg_upd, w_0).cpu() for agg_upd, w_0, is_frozen in zip(agg_update, copy.deepcopy(parameters), frozen_parameters) if not is_frozen]  # Send weights of NON-Frozen layers.

    pbar.close()

    fl_round_log_dict = {
        'Train/FL Round loss': total_loss / total_training_samples,
        'Train/FL Round Accuracy': fl_round_acc / total_training_samples,
        'Train/FL Round ANLS': fl_round_anls / total_training_samples,
        'fl_round': logger.current_epoch
    }

    logger.logger.log(fl_round_log_dict)

    # if fl_config["log_path"] is not None:
    # if config.flower:
    #     # log_communication(federated_round=fl_config.current_round, sender=client_id, receiver=-1, data=upd_weights, log_location=logger.comms_log_file)  # Store model's weights bytes.
    #     log_communication(federated_round=fl_config.current_round, sender=client_id, receiver=-1, data=upd_weights, log_location=logger.comms_log_file)  # Store only communicated weights (sent parameters).

    # Send the weights to the server
    return_dict["parameters"]=copy.deepcopy(upd_weights)
    #return upd_weights



def evaluate(data_loader, parameters, evaluator, config, return_dict):

    model = build_lora_model(config)
    set_lora_parameters(model, parameters)
    model.model=model.model.merge_and_unload()

    return_scores_by_sample = getattr(config, 'return_scores_by_sample', False)
    return_answers = getattr(config, 'return_answers', False)

    if return_scores_by_sample:
        scores_by_samples = {}
        total_accuracies = []
        total_anls = []

    else:
        total_accuracies = 0
        total_anls = 0

    all_pred_answers = []
    model.model.eval()

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        bs = len(batch['question_id'])
        skipped=0
        # try:
        with torch.no_grad():
            outputs, pred_answers, answer_conf = model.forward(batch, return_pred_answer=True)
        # except:
        #     print("skipped one batch because of some error")
        #     skipped+=1
        #     continue

        metric = evaluator.get_metrics(batch['answers'], pred_answers, batch.get('answer_type', None))

        if return_scores_by_sample:
            for batch_idx in range(bs):
                scores_by_samples[batch['question_id'][batch_idx]] = {
                    'accuracy': metric['accuracy'][batch_idx],
                    'anls': metric['anls'][batch_idx],
                    'pred_answer': pred_answers[batch_idx],
                    'pred_answer_conf': answer_conf[batch_idx]
                }

        if return_scores_by_sample:
            total_accuracies.extend(metric['accuracy'])
            total_anls.extend(metric['anls'])

        else:
            total_accuracies += sum(metric['accuracy'])
            total_anls += sum(metric['anls'])

        if return_answers:
            all_pred_answers.extend(pred_answers)

    if not return_scores_by_sample:
        total_accuracies = total_accuracies/(len(data_loader.dataset) - skipped)
        total_anls = total_anls/(len(data_loader.dataset)-skipped)
        scores_by_samples = []
    
    return_dict["accuracy"]=total_accuracies
    return_dict["total_anls"]= total_anls
    return_dict["pred_answers"]=all_pred_answers
    return_dict["scores_by_samples"]=scores_by_samples

    is_updated = evaluator.update_global_metrics(return_dict['accuracy'], return_dict['total_anls'], 0)
    save_model(model, config.current_round, update_best=is_updated, kwargs=config)

    
    return return_dict #total_accuracies, total_anls, all_pred_answers, scores_by_samples


def main_eval(config):
    start_time = time.time()

    config.return_scores_by_sample = True
    config.return_answers = True

    dataset = build_dataset(config, 'val')
    sampler = None
    pin_memory = False

    val_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory, sampler=sampler)

    model = build_model(config)

    logger = Logger(config=config)
    logger.log_model_parameters(model)

    evaluator = Evaluator(case_sensitive=False)
    accuracy_list, anls_list, pred_answers, scores_by_samples = evaluate(val_data_loader, model, evaluator, config)
    accuracy, anls = np.mean(accuracy_list), np.mean(anls_list)

    inf_time = time_stamp_to_hhmmss(time.time() - start_time, string=True)
    logger.log_val_metrics(accuracy, anls, update_best=False)

    save_data = {
        "Model": config.model_name,
        "Model_weights": config.model_weights,
        "Dataset": config.dataset_name,
        "Page retrieval": getattr(config, 'page_retrieval', '-').capitalize(),
        "Inference time": inf_time,
        "Mean accuracy": accuracy,
        "Mean ANLS": anls,
        "Scores by samples": scores_by_samples,
    }

    results_file = os.path.join(config.save_dir, 'results', config.experiment_name)
    save_json(results_file, save_data)

    print("Results correctly saved in: {:s}".format(results_file))


""" I think that in current version 1.4.0 centralized evaluation is still not working correctly.
    See https://github.com/adap/flower/blob/1982f5f4f1f0698c56122b627b64b857e619f3bf/src/py/flwr/server/strategy/fedavg.py#L164, they send empty dictionary as config.
"""
def fl_centralized_evaluation(server_round, parameters, config):
    # config= argparse.Namespace(
    #     model_name="vt5",
    #     model_weights= '/media/chs.hdsi/DATA/PFL-DocVQA/models/vt5_base.ckpt/vm_model',
    #     imdb_dir='/media/chs.hdsi/DATA/PFL-DocVQA/data/clients',
    #     images_dir='/media/chs.hdsi/DATA/PFL-DocVQA/data/images', 
    #     provider_docs='/media/chs.hdsi/DATA/PFL-DocVQA/data/clients/data_points.json', 
    #     current_round=server_round
    # )
    args = parse_args()
    config=load_config(args)
    config.server_round=server_round
    #model = build_model(config)
    val_loader = build_dataset(config, 'val')
    #set_parameters_model(model, parameters)  # Update model with the latest parameters
    # loss, accuracy = test(net, val_loader)

    evaluator = Evaluator(case_sensitive=False)
    #logger = Logger(config=config)

    accuracy, anls, _, _ = list(evaluate(val_loader, parameters, evaluator, config,{}).values())  # data_loader, model, evaluator, **kwargs
    is_updated = evaluator.update_global_metrics(accuracy, anls, 0)
    logger.log_val_metrics(accuracy, anls, update_best=is_updated)
    save_model(model, config.current_round, update_best=is_updated, kwargs=config)

    print("Server-side evaluation accuracy {:2.4f} / ANLS {1.6f}".format(accuracy, anls))
    return float(0), len(val_loader), {"accuracy": float(accuracy), "anls": anls}


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters_from_model(self.model)

    def evaluate(self, parameters, config):
        set_parameters_model(self.model, parameters)
        evaluator = Evaluator(case_sensitive=False)
        # loss, accuracy = test(self.model, self.valloader)
        total_accuracies, total_anls, all_pred_answers, scores_by_samples = evaluate(self.valloader, self.model, evaluator, config)  # data_loader, model, evaluator, **kwargs
        return float(0), len(self.valloader), {"accuracy": float(total_accuracies), "anls": total_anls}   # First parameter is loss.


def client_fn(client_id):
    """Create a Flower client representing a single organization."""
    model = build_model(config)
    dataset = build_dataset(config, 'val')
    val_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    return FlowerClient(model, val_data_loader, val_data_loader)


if __name__ == '__main__':

    # Set `MASTER_ADDR` and `MASTER_PORT` environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9957'

    args = parse_args()
    config = load_config(args)

    if not config.flower:
        main_eval(config)

    else:
        # Create FedAvg strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0,  # Sample 100% of available clients for training
            fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=0,  # Never sample less than 10 clients for training
            min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
            min_available_clients=1,  # Wait until all 10 clients are available
            evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        )

        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
        client_resources = None
        # DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
        if config.device == "cuda":
            client_resources = {"num_gpus": 1}

        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=1,
            config=fl.server.ServerConfig(num_rounds=config.fl_params.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )

    # Centralized evaluation
    # If fraction_evaluate is set to 0.0, federated evaluation will be disabled.
    # https://flower.dev/docs/evaluation.html


