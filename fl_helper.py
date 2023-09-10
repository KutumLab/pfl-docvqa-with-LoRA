import torch
from collections import OrderedDict
from peft import get_peft_model_state_dict,set_peft_model_state_dict
import json 
import random
from build_utils import build_dataset, build_provider_dataset, build_model, build_lora_model, build_optimizer
from torch.utils.data import DataLoader
from communication.log_communication import log_communication
from datasets.PFL_DocVQA import collate_fn
import copy
from tqdm  import tqdm
from eval import Evaluator
from differential_privacy.dp_utils import (add_dp_noise, clip_parameters, flatten_params, get_shape, reconstruct_shape)




def fl_train(parameters,logger, client_id, fl_config,return_dict):#data_loaders, model, optimizer, evaluator, logger,
    """
    Trains and returns the updated weights.
    """
    config=fl_config
    if fl_config.use_dp:
        # Pick a subset of providers
        provider_to_doc = json.load(open(fl_config.provider_docs, 'r'))
        provider_to_doc = provider_to_doc["client_" + client_id]
        providers = random.sample(list(provider_to_doc.keys()), k=fl_config.dp_params.providers_per_fl_round)  # 50
        train_datasets = [build_provider_dataset(fl_config, 'train', provider_to_doc, provider, client_id) for provider in providers]
        

    else: 
        train_datasets = [build_dataset(fl_config, 'train', client_id=client_id)]
        #train_datasets,_=random_split(train_datasets, [0.1, 0.9])

    data_loaders = [DataLoader(train_dataset, batch_size=fl_config.batch_size, shuffle=False, collate_fn=collate_fn) for train_dataset in train_datasets]

    print("in fit function")
    model=build_lora_model(fl_config)

    #set_parameters_model(model, parameters)
    
    total_training_steps = sum([len(data_loader) for data_loader in data_loaders])
    print("prepared data")
    #logger = Logger(config=config)

    # lr_scheduler disabled due to malfunction in FL setup.
    # optimizer, lr_scheduler = build_optimizer(model, length_train_loader=total_training_steps, config=config)
    optimizer = build_optimizer(model, config=fl_config)
    print("built optimizer")
    #logger.log_model_parameters(model)
    
    print("setting model parameters")
    # model.model.train()
    
    param_keys = list(model.model.state_dict().keys())
    parameters = copy.deepcopy([val.cpu().numpy() for val in model.model.state_dict().values()])

    keyed_parameters = {n: p.requires_grad for n, p in model.model.named_parameters()}
    frozen_parameters = [not keyed_parameters[n] if n in keyed_parameters else False for n, p in model.model.state_dict().items()]

    logger.current_epoch += 1

    agg_update = None
    if not config.use_dp and len(data_loaders) > 1:
        raise ValueError("Non private training should only use one data loader.")
    
    
    total_training_steps = sum([len(data_loader) for data_loader in data_loaders]) * fl_config.fl_params.iterations_per_fl_round
    total_training_samples = sum([len(data_loader.dataset) for data_loader in data_loaders]) * fl_config.fl_params.iterations_per_fl_round
    pbar = tqdm(total=total_training_steps)
    

    total_loss = 0
    fl_round_acc = 0
    fl_round_anls = 0
    n=0
    evaluator = Evaluator(case_sensitive=False)
    print("just above for loop")
    for provider_dataloader in data_loaders:
        # Set model weights to state of beginning of federated round
        state_dict = OrderedDict({k: v for k, v in zip(param_keys, parameters)})
        model.model.load_state_dict(state_dict, strict=True)
        model.model.train()

        # Reset the optimizer
        if config.use_dp:
            optimizer = build_optimizer(model, fl_config)

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
                n+=len(metric['anls'])

                log_dict = {
                    'Train/Batch loss': outputs.loss.item(),
                    'Train/Batch Accuracy': np.mean(metric['accuracy']),
                    'Train/Batch ANLS': np.mean(metric['anls']),
                    'lr': optimizer.param_groups[0]['lr']
                }

                logger.logger.log(log_dict)
                #pbar.set_description(f"Loss: {outputs.loss.item()}")
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
    if config.flower:
        # log_communication(federated_round=fl_config.current_round, sender=client_id, receiver=-1, data=upd_weights, log_location=logger.comms_log_file)  # Store model's weights bytes.
        log_communication(federated_round=fl_config.current_round, sender=client_id, receiver=-1, data=upd_weights, log_location=logger.comms_log_file)  # Store only communicated weights (sent parameters).

    # Send the weights to the server
     # Prepare return values
    return_dict["parameters"] = upd_weights

    #logger.logger.finish()
    #return return_dict
