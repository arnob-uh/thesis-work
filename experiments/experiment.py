import datetime
import json
import os
import random
import hashlib
import time
import codecs
import numpy as np
import torch
import wandb
from dataclasses import asdict, dataclass, field
from prettytable import PrettyTable
from typing import Dict, List, Type, Union
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torchmetrics import MeanSquaredError, MetricCollection, MeanAbsoluteError, MeanAbsolutePercentageError
from tqdm import tqdm

from data.scaler import *
from datasets import *
from datasets.dataset import TimeSeriesDataset
from datasets.dataloader import (
    ChunkSequenceTimefeatureDataLoader, ETTHLoader, ETTMLoader
)
from nn.metric import R2, Corr, RMSE
from normalizations import *
from utils.early_stopping import EarlyStopping
from experiments.Model import Model


@dataclass
class ResultRelatedSettings:
    dataset_type: str
    optm_type: str = "Adam"
    model_type: str = ""
    scaler_type: str = "StandarScaler"
    loss_func_type: str = "mse"
    batch_size: int = 32
    lr: float = 0.0003
    l2_weight_decay: float = 0.0005
    epochs: int = 100

    horizon: int = 3
    windows: int = 384
    pred_len: int = 1

    patience: int = 5
    max_grad_norm: float = 5.0
    invtrans_loss: bool = False

    norm_type: str = ''
    split_type: str = "custom"
    norm_config: dict = field(default_factory=lambda: {})


@dataclass
class Settings(ResultRelatedSettings):
    data_path: str = "./data"
    device: str = "cuda:0"
    num_worker: int = 8
    save_dir: str = "./results"
    experiment_label: str = str(int(time.time()))

class NormExperiment(Settings):
    def _use_wandb(self):
        return hasattr(self, "wandb")

    def _run_print(self, *args, **kwargs):
        time_str = '[' + str(datetime.datetime.utcnow() + datetime.timedelta(hours=8))[:19] + '] -'
        print(*args, **kwargs)
        if hasattr(self, "run_setuped") and getattr(self, "run_setuped") is True:
            with open(os.path.join(self.run_save_dir, 'output.log'), 'a+') as f:
                print(time_str, *args, flush=True, file=f)

    def _init_loss_func(self):
        loss_func_map = {"mse": MSELoss, "l1": L1Loss}
        self.loss_func = loss_func_map[self.loss_func_type]()

    def _init_metrics(self):
        if self.pred_len == 1:
            self.metrics = MetricCollection(
                metrics={
                    "r2": R2(self.dataset.num_features, multioutput="uniform_average"),
                    "r2_weighted": R2(self.dataset.num_features, multioutput="variance_weighted"),
                    "mse": MeanSquaredError(),
                    "corr": Corr(),
                    "mae": MeanAbsoluteError(),
                }
            )
        else:
            self.metrics = MetricCollection(
                metrics={
                    "mse": MeanSquaredError(),
                    "mae": MeanAbsoluteError(),
                    "mape": MeanAbsolutePercentageError(),
                    "rmse": RMSE()
                }
            )
        self.metrics.to(self.device)

    @property
    def result_related_config(self):
        ident = asdict(self)
        for key in ["data_path", "device", "num_worker", "save_dir", "experiment_label"]:
            if key in ident:
                del ident[key]
        return ident

    def _run_identifier(self, seed) -> str:
        ident = self.result_related_config
        ident["seed"] = seed
        ident["invtrans_loss"] = False
        if self.norm_config is None:
            del ident['norm_config']
        ident_md5 = hashlib.md5(json.dumps(ident, sort_keys=True).encode("utf-8")).hexdigest()
        return str(ident_md5)

    def _init_data_loader(self):
        self.dataset: TimeSeriesDataset = self._parse_type(self.dataset_type)(root=self.data_path)
        self.scaler = self._parse_type(self.scaler_type)()
        if self.split_type == "popular" and self.dataset_type[:3] == "ETT":
            if self.dataset_type[:4] == "ETTh":
                self.dataloader = ETTHLoader(
                    self.dataset, self.scaler, window=self.windows, horizon=self.horizon,
                    steps=self.pred_len, shuffle_train=True, freq="h",
                    batch_size=self.batch_size, num_worker=self.num_worker
                )
            elif self.dataset_type[:4] == "ETTm":
                self.dataloader = ETTMLoader(
                    self.dataset, self.scaler, window=self.windows, horizon=self.horizon,
                    steps=self.pred_len, shuffle_train=True, freq="h",
                    batch_size=self.batch_size, num_worker=self.num_worker
                )
        else:
            self.dataloader = ChunkSequenceTimefeatureDataLoader(
                self.dataset,
                self.scaler,
                window=self.windows,
                horizon=self.horizon,
                steps=self.pred_len,
                scale_in_train=False,
                shuffle_train=True,
                freq="h",
                batch_size=self.batch_size,
                train_ratio=0.7,
                val_ratio=0.2,
                num_worker=self.num_worker,
            )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = self.dataloader.train_size
        self.val_steps = self.dataloader.val_size
        self.test_steps = self.dataloader.test_size

    def _init_optimizer(self):
        self.model_optim = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optim, T_max=self.epochs
        )

    def _init_n_model(self):
        Ty = self._parse_type(self.norm_type)
        if self.norm_type == 'RevIN':
            self.n_model: torch.nn.Module = Ty(self.dataset.num_features, True, **self.norm_config)
        elif self.norm_type == 'SAN':
            self.n_model: torch.nn.Module = Ty(self.windows, self.pred_len, 12, self.dataset.num_features,
                                               **self.norm_config)
        elif self.norm_type in ['DAIN']:
            self.n_model: torch.nn.Module = Ty(self.dataset.num_features)
        elif self.norm_type in ['ZScore']:
            self.n_model: torch.nn.Module = Ty()
        else:
            self.n_model: torch.nn.Module = Ty(self.windows, self.pred_len, self.dataset.num_features,
                                               **self.norm_config)
        self.n_model = self.n_model.to(self.device)

    def _init_model(self):
        self.model = Model(self.model_type, self.f_model, self.n_model).to(self.device)

    def is_sep_loss(self):
        return "seploss" in self.norm_config and self.norm_config['seploss']

    def _setup(self):
        self._init_data_loader()
        self._init_metrics()
        self._init_loss_func()
        self.current_epochs = 0
        self.current_run = 0
        self.setuped = True

    def _setup_run(self, seed):
        if not hasattr(self, "setuped"):
            self._setup()
        self.reproducible(seed)
        self._init_n_model()
        self._init_f_model()
        self._init_model()

        if self.is_sep_loss():
            self._init_sep_optimizer()
        else:
            self._init_optimizer()

        self.current_epoch = 0
        self.run_save_dir = os.path.join(
            self.save_dir,
            "runs",
            self.model_type,
            self.dataset_type,
            f"w{self.windows}h{self.horizon}s{self.pred_len}",
            self._run_identifier(seed),
        )
        self.best_checkpoint_filepath = os.path.join(self.run_save_dir, "best_model.pth")
        self.run_checkpoint_filepath = os.path.join(self.run_save_dir, "run_checkpoint.pth")
        self.early_stopper = EarlyStopping(
            self.patience, verbose=True, path=self.best_checkpoint_filepath
        )
        self.run_setuped = True

    def _parse_type(self, str_or_type: Union[Type, str]) -> Type:
        if isinstance(str_or_type, str):
            return eval(str_or_type)
        elif isinstance(str_or_type, type):
            return str_or_type
        else:
            raise RuntimeError(f"{str_or_type} should be string or type")

    def get_run_state(self):
        if self.is_sep_loss():
            return {
                "model": self.model.state_dict(),
                "current_epoch": self.current_epoch,
                "n_optimizer": self.n_model_optim.state_dict(),
                "f_optimizer": self.f_model_optim.state_dict(),
                "rng_state": torch.get_rng_state(),
                "early_stopping": self.early_stopper.get_state(),
            }
        else:
            return {
                "model": self.model.state_dict(),
                "current_epoch": self.current_epoch,
                "optimizer": self.model_optim.state_dict(),
                "rng_state": torch.get_rng_state(),
                "early_stopping": self.early_stopper.get_state(),
            }

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        self.run_state = self.get_run_state()
        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")

    def reproducible(self, seed):
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determinstic = True

    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()
        if dataloader is self.train_loader:
            length = self.dataloader.train_size
        elif dataloader is self.val_loader:
            length = self.dataloader.val_size
        else:
            length = self.dataloader.test_size

        y_truths, y_preds = [], []
        with torch.no_grad():
            with tqdm(total=length, position=0, leave=True) as progress_bar:
                for batch_x, batch_y, batch_origin_y, batch_x_date_enc, batch_y_date_enc in dataloader:
                    batch_size = batch_x.size(0)
                    batch_x = batch_x.to(self.device, dtype=torch.float32)
                    batch_y = batch_y.to(self.device, dtype=torch.float32)
                    batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                    batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                    preds, truths = self._process_batch(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc)
                    batch_origin_y = batch_origin_y.to(self.device)
                    if self.invtrans_loss:
                        preds = self.scaler.inverse_transform(preds)
                        truths = batch_origin_y
                    if self.pred_len == 1:
                        self.metrics.update(preds.view(batch_size, -1), truths.view(batch_size, -1))
                    else:
                        self.metrics.update(preds.contiguous(), truths.contiguous())
                    progress_bar.update(batch_x.shape[0])
                    y_preds.append(preds)
                    y_truths.append(truths)
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def _test(self) -> Dict[str, float]:
        test_result = self._evaluate(self.test_loader)
        if self._use_wandb():
            for name, metric_value in test_result.items():
                wandb.run.summary["test_" + name] = metric_value
        self._run_print(f"test_results: {test_result}")
        return test_result

    def _val(self):
        val_result = self._evaluate(self.val_loader)
        if self._use_wandb():
            for name, metric_value in val_result.items():
                wandb.run.summary["val_" + name] = metric_value
        self._run_print(f"vali_results: {val_result}")
        return val_result

    def _train(self):
        with torch.enable_grad(), tqdm(total=self.train_steps, position=0, leave=True) as progress_bar:
            self.model.train()
            times = []
            train_loss = []
            for i, (batch_x, batch_y, origin_y, batch_x_date_enc, batch_y_date_enc) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device).float()
                self.model_optim.zero_grad()
                bs = batch_x.size(0)
                batch_x = batch_x.to(self.device, dtype=torch.float32).float()
                batch_y = batch_y.to(self.device, dtype=torch.float32).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                start = time.time()
                pred, true = self._process_batch(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc)
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred).float()
                    true = origin_y
                if isinstance(self.model.nm, SAN):
                    mean = self.model.pred_stats[:, :, :self.dataset.num_features]
                    std = self.model.pred_stats[:, :, self.dataset.num_features:]
                    sliced_true = true.reshape(bs, -1, 12, self.dataset.num_features)
                    loss = (self.loss_func(pred, true) +
                            self.loss_func(mean, sliced_true.mean(2)) +
                            self.loss_func(std, sliced_true.std(2)))
                else:
                    loss = self.loss_func(pred, true) + self.model.nm.loss(true)
                    if self.scaler_type is NoScaler:
                        loss = 10000 * self.loss_func(pred, true) + 10000 * self.model.nm.loss(true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.model_optim.step()
                end = time.time()
                times.append(end - start)
            return train_loss

    def _check_run_exist(self, seed: str):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        with codecs.open(os.path.join(self.run_save_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)
        return os.path.exists(self.run_checkpoint_filepath)

    def _load_best_model(self):
        self.model.load_state_dict(torch.load(self.best_checkpoint_filepath, map_location=self.device))

    def count_parameters(self, print_fun):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print_fun(table)
        print_fun(f"Total Trainable Params: {total_params}")
        return total_params

    def run(self, seed=42) -> Dict[str, float]:
        if hasattr(self, "finished") and self.finished is True:
            return {}
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)
        self._run_print(f"run : {self.current_run} in seed: {seed}")
        self.model_parameters_num = self.count_parameters(self._run_print)
        self._run_print(f"model parameters: {self.model_parameters_num}")
        if self._use_wandb():
            wandb.run.summary["parameters"] = self.model_parameters_num

        epoch_time = time.time()
        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop:
                self._run_print(f"loss no decreased for {self.patience} epochs,  early stopping ....")
                break
            if self._use_wandb():
                wandb.run.summary["at_epoch"] = self.current_epoch
            self.reproducible(seed + self.current_epoch)

            if self.is_sep_loss():
                train_losses = self._sep_train()
            else:
                train_losses = self._train()

            self._run_print(f"Epoch: {self.current_epoch + 1} cost time: {time.time() - epoch_time}")
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")

            result = self._val()
            self.current_epoch += 1
            self.early_stopper(result[self.loss_func_type], model=self.model)
            self._save_run_check_point(seed)
            self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test()
        self.run_setuped = False
        return best_test_result

    def runs(self, seeds: List[int] = [42, 43, 44]):
        if hasattr(self, "finished") and self.finished is True:
            return
        if self._use_wandb():
            wandb.config.update({"seeds": seeds})
        results = []
        for i, seed in enumerate(seeds):
            self.current_run = i
            if self._use_wandb():
                wandb.run.summary["at_run"] = i
            torch.cuda.empty_cache()
            result = self.run(seed=seed)
            torch.cuda.empty_cache()
            results.append(result)
            if self._use_wandb():
                for name, metric_value in result.items():
                    wandb.run.summary["test_" + name] = metric_value

        import pandas as pd
        df = pd.DataFrame(results)
        self.metric_mean_std = df.agg(["mean", "std"]).T
        print(self.metric_mean_std.apply(lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1))
        if self._use_wandb():
            for index, row in self.metric_mean_std.iterrows():
                wandb.run.summary[f"{index}_mean"] = row["mean"]
                wandb.run.summary[f"{index}_std"] = row["std"]
                wandb.run.summary[index] = f"{row['mean']:.4f}±{row['std']:.4f}"
        wandb.finish()
