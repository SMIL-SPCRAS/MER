import os
import time
import torch
import json
import toml
import logging
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from data.dataset import AVTDataset
from models.architectures import AVTmodel
from net_trainer.net_trainer import train_one_epoch_sl, val_one_epoch_sl, fix_seeds, get_weights

# Set logging level to suppress warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AVTModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['general']['device']
        self.setup_training()
        self.load_data()
        self.build_model()
        self.setup_logging()

    def setup_training(self):
        training_config = self.config['training']
        self.batch_size = training_config['batch_size']
        self.epochs = training_config['epochs']
        self.seed = training_config['seed']
        self.learning_rate = training_config['learning_rate']
        self.weight_decay = training_config['weight_decay']
        self.gated_dim = training_config['gated_dim']
        self.n_classes = training_config['n_classes']
        self.dropout = training_config['dropout']
        self.corpus = training_config['corpus']
        self.modality = training_config['modality']
        self.patience = training_config['patience']
        self.optimizer_choice = training_config['optimizer']

        model_config = self.config['model']
        self.input_dim_a = model_config['input_dim_a']
        self.input_dim_v = model_config['input_dim_v']
        self.input_dim_t = model_config['input_dim_t']

        # Scheduler configuration
        scheduler_config = self.config.get('scheduler', None)
        if scheduler_config and scheduler_config.get('type', '').lower() != 'none':
            self.scheduler_type = scheduler_config.get('type', 'CosineAnnealingWarmRestarts')
            self.scheduler_params = {k: v for k, v in scheduler_config.items() if k != 'type'}
        else:
            self.scheduler_type = None  # No scheduler will be used

        fix_seeds(self.seed)
        self.stop_training = False
        self.stop_flag_training = 0
        self.max_uar = 0

    def load_data(self):
        paths = self.config['paths']
        train_data_path = paths['train_data_path']
        val_data_path = paths['val_data_path']

        train_feature_paths = [os.path.join(train_data_path, i) for i in os.listdir(train_data_path)]
        val_feature_paths = [os.path.join(val_data_path, i) for i in os.listdir(val_data_path)]

        train_data = AVTDataset(train_feature_paths)
        val_data = AVTDataset(val_feature_paths)

        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

    def build_model(self):
        self.model = AVTmodel(
            self.input_dim_a, self.input_dim_v, self.input_dim_t,
            gated_dim=self.gated_dim, n_classes=self.n_classes, drop=self.dropout
        )
        self.model.to(self.device)

        class_weights = get_weights(self.train_dataloader)
        logger.info(f"Class weights: {class_weights}")
        self.criterion = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))

        if self.optimizer_choice.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_choice.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_choice.lower() == 'lion':
            try:
                from lion_pytorch import Lion
                self.optimizer = Lion(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            except ImportError:
                raise ImportError("Optimizer 'Lion' is not available. Please ensure it is installed.")
        else:
            raise ValueError(f"Unknown optimizer choice: {self.optimizer_choice}")

        # Initialize scheduler if specified
        if self.scheduler_type:
            if self.scheduler_type == 'CosineAnnealingWarmRestarts':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, **self.scheduler_params
                )
            elif self.scheduler_type == 'StepLR':
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, **self.scheduler_params
                )
            else:
                raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
        else:
            self.scheduler_type = None  # No scheduler will be used
            self.scheduler = None       # No scheduler will be used

    def setup_logging(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        paths = self.config['paths']
        self.save_model_path = os.path.join(paths['save_model_path'], f'{self.corpus}_{current_time}')
        os.makedirs(self.save_model_path, exist_ok=True)
        logger.info(f"Model will be saved in: {self.save_model_path}")

        logging_config = self.config['logging']
        log_dir = os.path.join(self.save_model_path, logging_config.get('log_dir', 'logs'))
        self.train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
        self.val_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

        config_save_path = os.path.join(self.save_model_path, 'training_config.toml')
        training_config = {
            'general': {'device': self.config['general']['device']},
            'training': self.config['training'],
            'model': {
                'input_dims': [self.input_dim_a, self.input_dim_v, self.input_dim_t],
                'gated_dim': self.gated_dim,
                'n_classes': self.n_classes,
                'dropout': self.dropout
            },
            'optimizer': {
                'type': self.optimizer_choice,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay
            },
            'seed': self.seed
        }

        # Include scheduler config only if a scheduler is used
        if self.scheduler_type:
            training_config['scheduler'] = {
                'type': self.scheduler_type,
                **self.scheduler_params
            }

        with open(config_save_path, 'w') as f:
            toml.dump(training_config, f)
        logger.info(f"Training config saved to: {config_save_path}")

        self.metrics_file = os.path.join(self.save_model_path, 'metrics.json')
        self.metrics_data = []

    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=4)
        logger.info(f"Metrics saved to: {self.metrics_file}")

    def train(self):
        for epoch in range(self.epochs):
            logger.info(f"Epoch: {epoch + 1}/{self.epochs}")
            epoch_metrics = {'epoch': epoch + 1}

            # Training
            self.model.train()
            avg_loss, uar, acc, mf1, wf1 = train_one_epoch_sl(
                self.train_dataloader, self.train_writer, epoch, self.optimizer,
                self.model, self.device, self.criterion, self.modality, self.scheduler_type, self.scheduler
            )
            epoch_metrics.update({
                'train_avg_loss': avg_loss,
                'train_uar': uar,
                'train_acc': acc,
                'train_mf1': mf1,
                'train_wf1': wf1
            })

            # Validation
            self.model.eval()
            avg_vloss, vuar, vacc, vmf1, vwf1 = val_one_epoch_sl(
                self.val_dataloader, self.val_writer, epoch, self.model,
                self.device, self.criterion, self.modality
            )
            if self.scheduler_type == 'StepLR':
                self.scheduler.step()
            epoch_metrics.update({
                'val_avg_loss': avg_vloss,
                'val_uar': vuar,
                'val_acc': vacc,
                'val_mf1': vmf1,
                'val_wf1': vwf1
            })

            # Logging
            self.metrics_data.append(epoch_metrics)
            self.save_metrics()

            logger.info(f"Epoch {epoch + 1} results:")
            logger.info(f"Train Loss: {avg_loss:.4f}, UAR: {uar:.4f}, Acc: {acc:.4f}, MF1: {mf1:.4f}, WF1: {wf1:.4f}")
            logger.info(f"Val Loss: {avg_vloss:.4f}, UAR: {vuar:.4f}, Acc: {vacc:.4f}, MF1: {vmf1:.4f}, WF1: {vwf1:.4f}")

            # Model Saving
            if self.max_uar < vuar:
                self.stop_flag_training = 0
                logger.info(f'Validation UAR Increased ({self.max_uar:.6f} --> {vuar:.6f}). Saving The Model.')
                self.max_uar = vuar
                model_save_path = os.path.join(self.save_model_path, f'best_model_epoch_{epoch + 1}.pth')
                torch.save(self.model.state_dict(), model_save_path)
                logger.info(f"Model saved to: {model_save_path}")
                if os.path.exists(model_save_path):
                    logger.info(f"Model file exists: {model_save_path}")
                else:
                    logger.warning(f"Failed to save model to: {model_save_path}")
            else:
                self.stop_flag_training += 1

            if self.stop_flag_training > self.patience:
                logger.info('Early stopping triggered. Stop training.')
                break

if __name__ == "__main__":
    config = toml.load('src/config.toml')

    trainer = AVTModelTrainer(config)
    trainer.train()
