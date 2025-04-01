import os
import uuid
from pathlib import Path
from typing import Any

from numpy import ndarray
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from maye import training, utils

Scalar = Tensor | ndarray | int | float


class WandBLogger:
    def __init__(
        self,
        project: str = "maye",
        name: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        log_dir: str | None = None,
        **kwargs,
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "``wandb`` package not found. Please install wandb using `pip install wandb` to use WandBLogger."
                "Alternatively, use the ``StdoutLogger``, which can be specified by setting metric_logger_type='stdout'."
            ) from e
        self._wandb = wandb

        # Use dir if specified, otherwise use log_dir.
        self.log_dir = kwargs.pop("dir", log_dir)

        _, self.rank = training.get_world_size_and_rank()
        if name is not None:
            random_suffix = str(uuid.uuid4().hex[:8])
            name = f"{name}_{random_suffix}"

        if self._wandb.run is None and self.rank == 0:
            # we check if wandb.init got called externally,
            _ = self._wandb.init(
                project=project,
                name=name,
                entity=entity,
                group=group,
                dir=self.log_dir,
                **kwargs,
            )

        if self._wandb.run:
            self._wandb.run._label(repo="camouflage")

        # define default x-axis (for latest wandb versions)
        if getattr(self._wandb, "define_metric", None):
            self._wandb.define_metric("global_step")
            self._wandb.define_metric("*", step_metric="global_step", step_sync=True)

        self.config_allow_val_change = kwargs.get("allow_val_change", False)

    def log_config(self, config: DictConfig) -> None:
        if self._wandb.run:
            resolved = OmegaConf.to_container(config, resolve=True)
            self._wandb.config.update(
                resolved, allow_val_change=self.config_allow_val_change
            )
            try:
                os.makedirs(config.output_dir, exist_ok=True)
                config_dir = os.path.join(config.output_dir, "config.yaml")
                output_config_fname = Path(config_dir)
                OmegaConf.save(config, output_config_fname)

                utils.logger.info(f"Logging {output_config_fname} to W&B under Files")
                self._wandb.save(
                    output_config_fname, base_path=output_config_fname.parent
                )

            except Exception as e:
                utils.logger.warning(
                    f"Error saving {output_config_fname} to W&B.\nError: \n{e}."
                    "Don't worry the config will be logged the W&B workspace"
                )

    def log_table(self, name: str, lines: list[Any], step: int) -> None:
        table = self._wandb.Table(columns=[name], data=[[item] for item in lines])
        self._wandb.log({name: table, "global_step": step})

    def log(self, name: str, data: Scalar, step: int, key: str = "global_step") -> None:
        if self._wandb.run:
            self._wandb.log({name: data, key: step})

    def log_dict(self, payload: dict[str, Scalar], step: int) -> None:
        if self._wandb.run:
            self._wandb.log({**payload, "global_step": step})

    def __del__(self) -> None:
        if self._wandb.run:
            self._wandb.finish()

    def close(self) -> None:
        if self._wandb.run:
            self._wandb.finish()
