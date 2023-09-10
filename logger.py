import os, socket, datetime

import torch
import wandb as wb

from utils import Singleton


class Logger(metaclass=Singleton):

    def __init__(self, config, client_id):
        self.client_id=client_id
        self.log_folder = config.save_dir
        self.experiment_name = config.experiment_name
        self.comms_log_file = os.path.join(self.log_folder, "communication_logs", "{:}.csv".format(self.experiment_name))

        machine_dict = {'cvc117': 'Local', 'cudahpc16': 'DAG', 'cudahpc25': 'DAG-A40'}
        machine = machine_dict.get(socket.gethostname(), socket.gethostname())

        dataset = config.dataset_name
        visual_encoder = getattr(config, 'visual_module', {}).get('model', '-').upper()

        tags = [config.model_name, dataset, machine]
        log_config = {
            'Model': config.model_name, 'Weights': config.model_weights, 'Dataset': dataset,
            'Visual Encoder': visual_encoder, 'Batch size': config.batch_size,
            'Max. Seq. Length': getattr(config, 'max_sequence_length', '-'), 'lr': config.lr, 'seed': config.seed,
        }

        if config.flower:
            tags.append('FL Flower')

            log_config.update({
                'FL Flower': True,
                'Sample Clients': config.fl_params.sample_clients,
                'Total Clients': config.fl_params.total_clients,
                'FL Rounds': config.fl_params.num_rounds,
                'Iterations per FL Round': config.fl_params.iterations_per_fl_round
            })

        if config.use_dp:
            tags.append('DP')
            sampling_prob = (config.fl_params.sample_clients / config.fl_params.total_clients) * (config.dp_params.providers_per_fl_round / 400)
            log_config.update({
                'DP': True,
                'DP Sensitivity': config.dp_params.sensitivity,
                'Noise Multiplier': config.dp_params.noise_multiplier,
                'Client sampling prob.': sampling_prob,
                'Providers per FL Round': config.dp_params.providers_per_fl_round
            })

        self.logger = wb.init(project="PFL-DocVQA-Competition",group=self.experiment_name, name=self.experiment_name+str(config.current_round)+str(self.client_id), dir=self.log_folder, tags=tags, config=log_config)
        self.logger.define_metric("Train/FL Round *", step_metric="fl_round")
        self.logger.define_metric("Val/FL Round *", step_metric="fl_round")
        self._print_config(log_config)

        self.current_epoch = config.current_round
        self.len_dataset = 0

    def _print_config(self, config):
        print("{:s}: {:s} \n{{".format(config['Model'], config['Weights']))
        for k, v in config.items():
            if k != 'Model' and k != 'Weights':
                print("\t{:}: {:}".format(k, v))
        print("}\n")

    def log_model_parameters(self, model):
        total_params = 0
        trainable_params = 0
        for attr in dir(model):
            if isinstance(getattr(model, attr), torch.nn.Module):
                total_params += sum(p.numel() for p in getattr(model, attr).parameters())
                trainable_params += sum(p.numel() for p in getattr(model, attr).parameters() if p.requires_grad)

        self.logger.config.update({
            'Model Params': int(total_params / 1e6),  # In millions
            'Model Trainable Params': int(trainable_params / 1e6)  # In millions
        })

        print("Model parameters: {:d} - Trainable: {:d} ({:2.2f}%)".format(
            total_params, trainable_params, trainable_params / total_params * 100))

    def log_val_metrics(self, accuracy, anls, update_best=False):
        str_msg = "FL Round {:d}: Accuracy {:2.2f}     ANLS {:2.4f}".format(self.current_epoch, accuracy*100, anls)

        if self.logger:
            self.logger.log({
                'Val/FL Round Accuracy': accuracy,
                'Val/FL Round ANLS': anls,
                'fl_round': self.current_epoch
            })

            if update_best:
                str_msg += "\tBest Accuracy!"
                self.logger.config.update({
                    "Best Accuracy": accuracy,
                    "Best FL Round": self.current_epoch
                }, allow_val_change=True)

        print(str_msg)

