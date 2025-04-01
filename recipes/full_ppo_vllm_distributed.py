import json
import sys
import time
from functools import partial
from itertools import chain
from pathlib import Path
from unittest.mock import patch

import hydra
import pandas as pd
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.distributed._tensor import DTensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from transformers import BatchFeature
from transformers.modeling_utils import init_empty_weights
from vllm import LLM, SamplingParams

from maye import rlhf, training, utils
from maye.datasets import Dataset
from maye.utils import collate_vision_inputs, generation, pad_sequence


class PPOFullFinetuneRecipeDistributed:
    def __init__(self, cfg: DictConfig) -> None:
        device_type = cfg.device
        self.device = utils.get_device(device=device_type)
        self.dtype = training.get_dtype(cfg.dtype, device=self.device)

        if self.dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # logging attributes
        self.output_dir = Path(cfg.output_dir)
        self.log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self.save_every_n_epochs = cfg.get("save_every_n_epochs", 0)
        self.save_eval_files = cfg.get("save_eval_files", False)

        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)

        self.distributed_backend = training.get_distributed_backend(
            device_type, offload_ops_to_cpu=self.fsdp_cpu_offload
        )
        from datetime import timedelta

        dist.init_process_group(self.distributed_backend, timeout=timedelta(minutes=60))

        self.world_size, self.rank = training.get_world_size_and_rank()
        self.is_rank_zero = self.rank == 0

        # Training cfg
        self.clip_grad_norm = cfg.get("clip_grad_norm", None)

        # activation checkpointing/offloading
        self.enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self.enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )

        if self.enable_activation_offloading:
            if self.device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self.enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif self.enable_activation_checkpointing:
            utils.log_rank_zero(
                "Hint: enable_activation_checkpointing is True,"
                "but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        self.seed = training.set_seed(seed=cfg.seed)
        self.total_epochs = cfg.num_epochs
        self.global_step = 0
        self.steps_run = 0
        self.total_steps = 0
        self.epochs_run = 0

    def setup(self, cfg: DictConfig) -> None:
        OmegaConf.resolve(cfg)
        utils.log_rank_zero(OmegaConf.to_yaml(cfg))

        if self.fsdp_cpu_offload:
            # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
            # speed up when benchmarking fused AdamW on CPU
            training.set_torch_num_threads()

        if self.is_rank_zero:
            self.metric_logger = instantiate(cfg.metric_logger)

            # log config with parameter override
            self.metric_logger.log_config(cfg)

        utils.log_rank_zero(OmegaConf.to_yaml(cfg))

        self.compile = cfg.get("compile", False)

        self.train_vit = cfg.train_vit
        self.train_connector = cfg.train_connector
        self.train_llm = cfg.train_llm

        self.policy_model, self.ref_policy_model = self.setup_model(
            cfg.config,
            cfg.model,
            cfg.checkpoint,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            enable_activation_offloading=cfg.enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
        )
        self.processor = instantiate(cfg.processor)

        self.loss_fn = instantiate(cfg.loss)
        if self.compile:
            self.loss_fn = training.compile_loss(self.loss_fn)

        utils.log_rank_zero("Loss is initialized.")

        (
            self.ds,
            self.sampler,
            self.dataloader,
        ) = self.setup_dataset_sampler_and_dataloader(
            cfg.dataset, cfg.collate_fn, batch_size=cfg.batch_size
        )

        (
            self.validation_dataloader,
            self.test_dataloader,
        ) = self.setup_validation_and_test_dataloader(
            cfg.validation_dataset,
            cfg.validation_collate_fn,
            cfg.test_dataset,
            cfg.test_collate_fn,
        )

        self.optimizer = self.setup_optimizer(
            cfg_optimizer=cfg.optimizer,
        )

        self.setup_training_hyperparameters(cfg)
        self.setup_training_parameters(cfg)

        # one "step" is a single gradient update update over a minibatch of trajectories
        self.global_step = self.epochs_run * self.steps_per_epoch

        self.enable_lr_scheduler = cfg.get("enable_lr_scheduler", False)

        self.lr_scheduler = self.setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_steps
            * (self.batch_size // self.ppo_batch_size)
            * self.ppo_epochs,
            last_epoch=self.global_step - 1,
        )
        self.setup_vllm(cfg.vllm)
        self._sync_weight_to_vllm(self.policy_model)

    def setup_vllm(self, cfg: DictConfig):
        if self.is_rank_zero:
            vllm_device = cfg.device
            if cfg.device == "auto":
                device_id = utils.get_visible_device(-1)
                vllm_device = f"cuda:{device_id}"
            elif "cuda:" in vllm_device:
                device_id = int(vllm_device.split(":")[1])
            else:
                raise ValueError(
                    f"Unrecognized Device {vllm_device}, please set cuda:x or auto."
                )

            if (
                vllm_device.split(":")[0] == "cuda"
                and device_id >= torch.cuda.device_count()
            ):
                raise ValueError(
                    f"The requested device for vllm ({cfg.device}) is not available. You are likely using vLLM "
                    "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                    "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                    f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                )

            if vllm_device in {f"cuda:{idx}" for idx in range(self.world_size)}:
                raise ValueError(
                    f"The requested device {vllm_device} is also being used for training. For higher throughput "
                    "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                    "If this is intentional, you may ignore this warning but should adjust "
                    "`vllm_gpu_memory_utilization` accordingly."
                )
            world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                return_value=None,
            )
            with world_size_patch, profiling_patch:
                self.llm = LLM(
                    model=cfg.model,
                    device=vllm_device,
                    gpu_memory_utilization=cfg.gpu_memory_utilization,
                    dtype=cfg.dtype,
                    distributed_executor_backend="external_launcher",
                )

        dist.barrier()

    def setup_training_hyperparameters(self, cfg: DictConfig) -> None:
        # KL hyperparameters
        self.kl_reward_coeff = cfg.kl_reward_coeff

        # GAE hyperparameters
        self.gamma = cfg.gamma
        self.whiten_rewards = cfg.whiten_rewards

        # trajectory generation args
        self.generation_kwargs = instantiate(cfg.generation_kwargs)

        # reward masking args
        self.min_response_length = cfg.min_response_length
        self.penalise_no_eos = cfg.penalise_no_eos
        self.reward_penalty = cfg.reward_penalty

        # lots of hand holding for stop tokens
        if cfg.get("stop_token_ids", False):
            stop_token_ids = cfg.stop_token_ids
            if self.processor.tokenizer.eos_token_id not in stop_token_ids:
                utils.logger.warning(
                    f"tokenizer eos_id ({self.processor.tokenizer.eos_token_id}) is not in stop_token_ids ({stop_token_ids})."
                    "This may lead to unexpected behaviour."
                )
        else:
            stop_token_ids = []
            if hasattr(self.processor.tokenizer, "eos_token_id"):
                stop_token_ids.append(self.processor.tokenizer.eos_token_id)

        self.stop_token_ids = torch.tensor(stop_token_ids, device=self.device)

    def setup_training_parameters(self, cfg: DictConfig) -> None:
        self.batch_size = cfg.batch_size
        self.forward_batch_size = cfg.forward_batch_size
        self.ppo_epochs = cfg.ppo_epochs
        self.ppo_batch_size = cfg.ppo_batch_size
        self.gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self.ppo_backward_batch_size = (
            cfg.ppo_batch_size // self.gradient_accumulation_steps
        )

        if self.batch_size % self.forward_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"forward_batch_size ({self.forward_batch_size})."
            )
        if self.batch_size % self.ppo_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"ppo_batch_size ({self.ppo_batch_size})."
            )
        if self.ppo_batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError(
                f"ppo_batch_size ({self.ppo_batch_size}) must be exactly divisible "
                f"by gradient_accumulation_steps ({self.gradient_accumulation_steps})."
            )

        self.steps_per_epoch = len(self.dataloader)
        self.total_steps = self.total_epochs * self.steps_per_epoch

        if self.total_steps == 0:
            if self.total_epochs == 0:
                raise ValueError(
                    f"num_epochs {cfg.num_epochs} must be greater than zero."
                )
            if self.steps_per_epoch == 0:
                raise ValueError(
                    f"batchsize {cfg.batch_size} must be greater than zero."
                )

        utils.log_rank_zero(
            f"Total steps to run: {self.total_steps}, Total epochs to run: {self.total_epochs}"
        )

    def setup_model(
        self,
        config_cfg: DictConfig,
        model_cfg: DictConfig,
        ckpt_cfg: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        custom_sharded_layers: list[str] | None,
    ):
        utils.log_rank_zero(
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ..."
        )
        init_start = time.perf_counter()

        model_state_dict = instantiate(ckpt_cfg).state_dict()

        with training.set_default_dtype(self.dtype), init_empty_weights():
            policy_config = instantiate(config_cfg)
            policy_config.use_cache = False
            policy_model = instantiate(model_cfg, policy_config)

            ref_policy_config = instantiate(config_cfg)
            ref_policy_config.use_cache = False
            ref_policy_model = instantiate(model_cfg, ref_policy_config)

        # disabling grad and dropout in reward and reference policy models
        policy_model.train()
        ref_policy_model.eval()
        for p in ref_policy_model.parameters():
            p.requires_grad = False

        device_mesh = dist.init_device_mesh(
            self.device.type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("dp",),
        )
        self.dp_size = device_mesh["dp"].size()
        self.dp_rank = device_mesh["dp"].get_local_rank()

        if self.compile:
            training.compile_model(policy_model)
            training.compile_model(ref_policy_model)

        if "Qwen" in policy_model.__class__.__name__:
            llm_backend, vit_backend, connector = (
                policy_model.model,
                policy_model.visual,
                policy_model.visual.merger,
            )
            ac_auto_wrap_policy = {
                type(llm_backend.layers[0].self_attn),
                type(llm_backend.layers[0].mlp),
                type(vit_backend.blocks[0].attn),
                type(vit_backend.blocks[0].mlp),
            }
            if not self.train_llm:
                for param in llm_backend.parameters():
                    param.requires_grad = False

            if not self.train_vit:
                for param in vit_backend.parameters():
                    param.requires_grad = False

            if self.train_connector:
                for param in connector.parameters():
                    param.requires_grad = True
        else:
            raise ValueError(
                f"Only support Qwen2/2.5-VL series, but found {policy_model.__class__.__name__}"
            )

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                policy_model,
                auto_wrap_policy=ac_auto_wrap_policy,
            )

        # Apply Fully Sharded Data Parallelism to the model
        fsdp_shard_conditions = [
            partial(
                training.get_shard_conditions,
                names_to_match=custom_sharded_layers,
            )
        ]
        training.shard_model(
            model=policy_model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
            dp_mesh=device_mesh["dp"],
        )
        training.shard_model(
            model=ref_policy_model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=True,
            dp_mesh=device_mesh["dp"],
        )

        training.load_from_full_model_state_dict(
            policy_model,
            model_state_dict,
            device=self.device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )
        training.load_from_full_model_state_dict(
            ref_policy_model,
            model_state_dict,
            device=self.device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            policy_model, enable_activation_offloading
        )

        training.validate_no_params_on_meta_device(policy_model)
        training.validate_no_params_on_meta_device(ref_policy_model)

        training.disable_dropout(policy_model)
        training.disable_dropout(ref_policy_model)

        utils.log_rank_zero(
            f"Instantiating policy model, ref policy model and processor took {time.perf_counter() - init_start:.2f} secs"
        )
        # synchronize before training begins
        dist.barrier()

        return policy_model, ref_policy_model

    def setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
    ) -> Optimizer:
        optimizer = instantiate(cfg_optimizer, self.policy_model.parameters())
        utils.log_rank_zero("Optimizer is initialized.")
        return optimizer

    def setup_validation_and_test_dataloader(
        self,
        validation_dataset_cfg: DictConfig,
        validation_collate_cfg: DictConfig,
        test_dataset_cfg: DictConfig,
        test_collate_cfg: DictConfig,
    ) -> tuple[DataLoader, DataLoader]:
        validation_collate_fn = instantiate(validation_collate_cfg)
        validation_ds = instantiate(validation_dataset_cfg)
        validation_sampler = RandomSampler(validation_ds)
        validation_dataloader = DataLoader(
            dataset=validation_ds,
            batch_size=len(validation_ds),
            sampler=validation_sampler,
            drop_last=False,
            collate_fn=partial(validation_collate_fn, processor=self.processor),
        )

        test_collate_fn, test_ds = instantiate(test_collate_cfg), instantiate(
            test_dataset_cfg
        )
        test_sampler = RandomSampler(test_ds)
        test_dataloader = DataLoader(
            dataset=test_ds,
            batch_size=len(test_ds),
            sampler=test_sampler,
            drop_last=False,
            collate_fn=partial(test_collate_fn, processor=self.processor),
        )

        return validation_dataloader, test_dataloader

    def setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig | None,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer | None:
        if cfg_lr_scheduler is None or not self.enable_lr_scheduler:
            utils.log_rank_zero(
                "No learning rate scheduler configured. Using constant learning rate."
            )
            return None

        optimizer = self.optimizer

        # Instantiate the learning rate scheduler
        lr_scheduler = instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_warmup_steps=0.1 * num_training_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        utils.log_rank_zero("Learning rate scheduler is initialized.")

        return lr_scheduler

    def setup_dataset_sampler_and_dataloader(
        self,
        dataset_cfg: DictConfig,
        collate_cfg: DictConfig,
        batch_size: int,
    ) -> tuple[Dataset, DistributedSampler, DataLoader]:
        ds = instantiate(dataset_cfg)
        collate_fn = instantiate(collate_cfg, processor=self.processor)

        sampler = DistributedSampler(
            ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_fn,
        )

        utils.log_rank_zero("Sampler and dataloader are initialized.")
        return ds, sampler, dataloader

    def generate_trajectory_vllm(
        self,
        encodings: list[BatchFeature],
        samples: list[dict],
        responses: torch.Tensor,
        response_texts: list[str],
    ) -> rlhf.Trajectory:
        context_length = encodings[0].input_ids.shape[1]

        trajectories: list[rlhf.Trajectory] = []
        for batch_start in range(0, self.batch_size, self.forward_batch_size):
            # Chunked processing:
            # extract a minibatch encodings, responses, and samples for forward computation.
            torch.cuda.empty_cache()
            batch_encodings = encodings[
                batch_start : batch_start + self.forward_batch_size
            ]
            batch = BatchFeature(
                {
                    key: torch.cat([d[key] for d in batch_encodings], dim=0)
                    for key in batch_encodings[0].keys()
                }
            )
            batch_responses = responses[
                batch_start : batch_start + self.forward_batch_size
            ]
            batch_response_texts = response_texts[
                batch_start : batch_start + self.forward_batch_size
            ]
            batch_samples = samples[batch_start : batch_start + self.forward_batch_size]
            vision_inputs = {
                k: v
                for k, v in batch.items()
                if k not in ["input_ids", "attention_mask"]
            }

            # Step III Trajectory Generation: Part 1
            # Concatenate query and response token ids to construct query-response sequences,
            # then compute corresponding attention masks and position ids.
            batch_query_responses = torch.cat([batch.input_ids, batch_responses], dim=1)
            masks = (
                batch_query_responses != self.processor.tokenizer.pad_token_id
            ).long()
            position_ids = generation.get_position_ids_from_padding_mask(masks)

            # Step III Trajectory Generation: Part 2
            # Compute logits over full query-response sequences
            # then extract and convert only the response positions to logprobs
            # for both policy and reference models.
            with torch.no_grad():
                logits = self.policy_model(
                    input_ids=batch_query_responses,
                    position_ids=position_ids,
                    attention_mask=masks,
                    use_cache=False,
                    past_key_values=None,
                    **vision_inputs,
                ).logits

            logits = rlhf.truncate_sequence_for_logprobs(logits, context_length)
            logprobs = rlhf.logits_to_logprobs(
                logits,
                batch_responses,
                self.generation_kwargs.temperature,
            )

            del logits
            torch.cuda.empty_cache()

            ref_logits = self.ref_policy_model(
                input_ids=batch_query_responses,
                position_ids=position_ids,
                attention_mask=masks,
                use_cache=False,
                past_key_values=None,
                **vision_inputs,
            ).logits
            ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
            ref_logprobs = rlhf.logits_to_logprobs(
                ref_logits, batch_responses, self.generation_kwargs.temperature
            )

            del ref_logits
            torch.cuda.empty_cache()

            (
                response_padding_masks,
                batch_responses,
            ) = rlhf.truncate_sequence_at_first_stop_token(
                batch_responses,
                self.stop_token_ids,
                self.processor.tokenizer.pad_token_id,
            )

            # Step III Trajectory Generation: Part 3
            # Compute scalar rewards for each response,
            # including accuracy, format, and language components.
            solutions = [sample["solution"] for sample in batch_samples]

            acc_rewards = rlhf.accuracy_reward_fn(
                batch_response_texts, solutions, judge_fn=self.ds.judge
            ).to(self.device)
            format_rewards = rlhf.format_reward_fn(batch_response_texts).to(self.device)
            language_rewards = rlhf.language_reward_fn(batch_response_texts).to(
                self.device
            )

            scores = acc_rewards + format_rewards + language_rewards

            # compute response sequence length for logging
            seq_lens = rlhf.get_unmasked_sequence_lengths(response_padding_masks)

            # A small trick
            # Apply penalties to responses that are too short or too long (do not end with an EOS token).
            if self.penalise_no_eos or self.min_response_length:
                reward_penalty_mask = rlhf.get_reward_penalty_mask(
                    batch_responses,
                    seq_lens,
                    self.stop_token_ids,
                    self.penalise_no_eos,
                    self.min_response_length,
                )
                scores[reward_penalty_mask] = self.reward_penalty

            del batch_responses
            torch.cuda.empty_cache()

            # mask out all the invalid values in the trajectory due to padding tokens
            logprobs[response_padding_masks] = 1.0
            ref_logprobs[response_padding_masks] = 1.0

            trajectories.append(
                rlhf.Trajectory(
                    query_responses=batch_query_responses,
                    logprobs=logprobs,
                    ref_logprobs=ref_logprobs,
                    masks=masks,
                    position_ids=position_ids,
                    response_padding_masks=response_padding_masks,
                    scores=scores,
                    acc_rewards=acc_rewards,
                    format_rewards=format_rewards,
                    language_rewards=language_rewards,
                    seq_lens=seq_lens,
                )
            )

        return rlhf.Trajectory(*map(torch.cat, zip(*trajectories)))

    def train(self) -> None:
        training.cleanup_before_training()

        if self.compile:
            utils.log_rank_zero(
                "NOTE: torch.compile is enabled and model is compiled in first forward."
                "Expect a relatively slow first iteration."
            )

        self.optimizer.zero_grad()

        pbar = tqdm(
            total=self.total_steps,
            initial=self.steps_run,
            disable=not self.is_rank_zero,
            dynamic_ncols=True,
        )
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            self.sampler.set_epoch(curr_epoch)
            acc_cnt = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            total_cnt = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            for encodings, vllm_inputs, samples in self.dataloader:
                # Step I: Data Flow
                # The dataloader yields inputs in two formats:
                # (1) `encodings` contain preprocessed vision and text inputs via hf_processor.
                # (2) `vllm_inputs` store the same data as VLLM-compatible dictionary inputs.
                encodings = [b.to(self.device, self.dtype) for b in encodings]
                num_tokens = (encodings[0].input_ids.numel()) * len(samples)
                context_length = encodings[0].input_ids.shape[1]

                # Step II: Response Collection
                # Response collection via vLLM. vllm_inputs are gathered across ranks,
                # responses are generated on rank 0, then broadcasted to all processes.
                # response_token_ids are collected and padded for the next step.
                gathered_inputs = [None for _ in range(self.world_size)]
                dist.all_gather_object(gathered_inputs, vllm_inputs)
                all_inputs = list(chain.from_iterable(gathered_inputs))  # type: ignore
                if self.is_rank_zero:
                    all_outputs = self.llm.generate(
                        all_inputs, self.generation_kwargs, use_tqdm=False
                    )
                else:
                    all_outputs = [None] * len(all_inputs)

                dist.barrier()
                dist.broadcast_object_list(all_outputs, src=0)
                rank_slice = slice(
                    self.rank * len(vllm_inputs),
                    (self.rank + 1) * len(vllm_inputs),
                )
                outputs = all_outputs[rank_slice]
                assert outputs is not None, f"Get no outputs in rank {self.rank}"

                response_list, response_texts = [], []
                for output in outputs:
                    response_texts.append(output.outputs[0].text)
                    response_token_ids = torch.tensor(
                        output.outputs[0].token_ids,
                        device=self.device,
                    ).unsqueeze(dim=0)
                    response_list.append(response_token_ids)

                responses = pad_sequence(
                    response_list,
                    dim=1,
                    padding_value=self.processor.tokenizer.eos_token_id,
                )
                dist.barrier()

                t0_traj = time.perf_counter()
                # Step III: Trajectory Generation
                trajectory = self.generate_trajectory_vllm(
                    encodings,
                    samples,
                    responses,
                    response_texts,
                )
                acc_cnt += (trajectory.acc_rewards == 1).sum()
                total_cnt += trajectory.acc_rewards.shape[0]
                traj_time = time.perf_counter() - t0_traj

                # Step IV Policy Update: Part 1
                # Compute token-level KL rewards and assign final outcome scores
                # to the last valid token in each sequence.
                with torch.no_grad():
                    rewards, kl, kl_rewards = rlhf.get_rewards_from_ref(
                        trajectory.scores,
                        trajectory.logprobs,
                        trajectory.ref_logprobs,
                        self.kl_reward_coeff,
                        trajectory.seq_lens,
                    )

                # Step IV Policy Update: Part 2
                # Estimate advantages and returns using the final rewards and response masks.
                advantages, returns = rlhf.estimate_advantages(
                    rewards,
                    self.gamma,
                    masks=~trajectory.response_padding_masks,
                )

                t0_ppo = time.perf_counter()
                rl_stats = []

                for _ in range(self.ppo_epochs):
                    # Step IV Policy Update: Part 3
                    # First, the batch is shuffled and divided into PPO mini-batches.
                    # Then, each mini-batch is further split into backward batches for gradient accumulation.
                    batch_idxs = torch.randperm(self.batch_size, device=self.device)
                    for i in range(0, self.batch_size, self.ppo_batch_size):
                        mini_batch_idxs = batch_idxs[i : i + self.ppo_batch_size]

                        batch_ppo_stats: list[rlhf.PPOStats] = []
                        for j in range(
                            0, self.ppo_batch_size, self.ppo_backward_batch_size
                        ):
                            backward_batch_idxs = mini_batch_idxs[
                                j : j + self.ppo_backward_batch_size
                            ]
                            backward_samples = [
                                samples[idx.item()] for idx in backward_batch_idxs
                            ]
                            # Re-collect vision inputs for the backward batch.
                            # Required because the batch order has been shuffled during chunking.
                            backward_vision_inputs = collate_vision_inputs(
                                backward_samples, processor=self.processor
                            )
                            backward_vision_inputs.to(self.device, self.dtype)

                            batch_trajectory = rlhf.Trajectory(
                                *map(
                                    partial(
                                        torch.index_select,
                                        dim=0,
                                        index=backward_batch_idxs,
                                    ),
                                    trajectory,
                                )
                            )
                            # Step IV Policy Update: Part 4
                            # Compute PPO loss and perform a backward pass for the current backward batch.
                            batch_ppo_stats.append(
                                self.ppo_step(
                                    batch_trajectory,
                                    advantages[backward_batch_idxs],
                                    context_length,
                                    backward_vision_inputs,
                                )
                            )
                            del batch_trajectory

                        rl_stats.append(
                            rlhf.PPOStats(*map(sum, zip(*batch_ppo_stats)))  # type: ignore
                        )

                        if self.clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.policy_model.parameters(),
                                max_norm=float(self.clip_grad_norm),
                            )
                            # If sharded, collect the DTensor here
                            if isinstance(grad_norm, DTensor):
                                grad_norm = grad_norm.full_tensor()

                        dist.barrier()
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        dist.barrier()

                        self.global_step += 1

                        # Step the learning rate scheduler
                        if self.enable_lr_scheduler and self.lr_scheduler is not None:
                            self.lr_scheduler.step()

                rl_stats = rlhf.PPOStats(*map(torch.stack, zip(*rl_stats)))
                rl_time = time.perf_counter() - t0_ppo

                # profit
                self.steps_run += 1
                if self.steps_run % self.log_every_n_steps == 0:
                    extra_metrics = {}
                    extra_metrics["training/lr"] = training.get_lr(self.optimizer)
                    if grad_norm is not None:
                        extra_metrics["training/grad_norm"] = grad_norm.item()
                    extra_metrics[
                        "training/temperature"
                    ] = self.generation_kwargs.temperature

                    self.log_metrics(
                        trajectory,
                        rl_stats,
                        kl,
                        kl_rewards,
                        num_tokens / traj_time,
                        num_tokens / rl_time,
                        **extra_metrics,
                    )

                    self.log_reflection_analysis(
                        response_texts,
                        is_correct=(trajectory.acc_rewards == 1.0).tolist(),
                    )
                self.cleanup_after_step(
                    trajectory, rl_stats, advantages, returns, kl, kl_rewards
                )
                pbar.update(1)
                self._sync_weight_to_vllm(self.policy_model)
                if self.steps_run == self.total_steps:
                    break

            # Evaluation
            self.epochs_run += 1
            dist.all_reduce(acc_cnt, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_cnt, op=dist.ReduceOp.SUM)
            utils.log_rank_zero(
                f"Training Set Epoch Accuracy: {acc_cnt / total_cnt:.3f}-{acc_cnt}/{total_cnt}"
            )
            if self.is_rank_zero:
                # Log the first 10 response texts from rank 0 for observing output patterns
                self.metric_logger.log_table(
                    f"outputs/Epoch {self.epochs_run}",
                    response_texts[:10],
                    step=self.epochs_run,
                )
                # Metrics: training set accuracy
                self.metric_logger.log(
                    "acc/Training Set Epoch Accuracy",
                    acc_cnt / total_cnt,
                    step=self.epochs_run,
                    key="Epoch",
                )

                # Metrics: validation & test set accuracy
                validation_acc_low_temp_pass1 = self.eval(
                    dataloader=self.validation_dataloader,
                    eval_name="Validation",
                    passk=1,
                    generation_strategy="low",
                    epoch=curr_epoch,
                )
                validation_acc_medium_temp_pass1 = self.eval(
                    dataloader=self.validation_dataloader,
                    eval_name="Validation",
                    passk=1,
                    generation_strategy="medium",
                    epoch=curr_epoch,
                )
                validation_acc_high_temp_pass8 = self.eval(
                    dataloader=self.validation_dataloader,
                    eval_name="Validation",
                    passk=8,
                    generation_strategy="high",
                    epoch=curr_epoch,
                )
                test_acc_low_temp_pass1 = self.eval(
                    dataloader=self.test_dataloader,
                    eval_name="Test",
                    passk=1,
                    generation_strategy="low",
                    epoch=curr_epoch,
                )
                test_acc_medium_temp_pass1 = self.eval(
                    dataloader=self.test_dataloader,
                    eval_name="Test",
                    passk=1,
                    generation_strategy="medium",
                    epoch=curr_epoch,
                )
                test_acc_high_temp_pass8 = self.eval(
                    dataloader=self.test_dataloader,
                    eval_name="Test",
                    passk=8,
                    generation_strategy="high",
                    epoch=curr_epoch,
                )
                self.metric_logger.log(
                    "acc/Validation Set Epoch Accuracy pass@1 Temperature=0.01",
                    validation_acc_low_temp_pass1,
                    step=self.epochs_run,
                    key="Epoch",
                )
                self.metric_logger.log(
                    "acc/Validation Set Epoch Accuracy pass@1 Temperature=0.6",
                    validation_acc_medium_temp_pass1,
                    step=self.epochs_run,
                    key="Epoch",
                )
                self.metric_logger.log(
                    "acc/Validation Set Epoch Accuracy pass@8 Temperature=1.0",
                    validation_acc_high_temp_pass8,
                    step=self.epochs_run,
                    key="Epoch",
                )
                self.metric_logger.log(
                    "acc/Test Set Epoch Accuracy pass@1 Temperature=0.01",
                    test_acc_low_temp_pass1,
                    step=self.epochs_run,
                    key="Epoch",
                )
                self.metric_logger.log(
                    "acc/Test Set Epoch Accuracy pass@1 Temperature=0.6",
                    test_acc_medium_temp_pass1,
                    step=self.epochs_run,
                    key="Epoch",
                )
                self.metric_logger.log(
                    "acc/Test Set Epoch Accuracy pass@8 Temperature=1.0",
                    test_acc_high_temp_pass8,
                    step=self.epochs_run,
                    key="Epoch",
                )

            # Save checkpoint at current epoch
            if (self.save_every_n_epochs > 0) and (
                curr_epoch + 1
            ) % self.save_every_n_epochs == 0:
                self.save_checkpoint(epoch=curr_epoch)

    def log_reflection_analysis(self, texts: list[str], is_correct: list[bool]):
        reflection_words = [
            "re-check",
            "re-evaluate",
            "re-examine",
            "re-think",
            "recheck",
            "reevaluate",
            "reexamine",
            "reevaluation",
            "rethink",
            "check again",
            "think again",
            "try again",
            "verify",
            "wait",
            "yet",
        ]

        # Convert all text to lowercase for easier matching
        texts_lower = [t.lower() for t in texts]
        total_count = len(texts_lower)

        # Identify whether each text contains reflection words for ratio computation
        has_reflection = [
            any(word in text for word in reflection_words) for text in texts_lower
        ]

        reflection_correct_texts = [
            texts[i] for i in range(len(texts)) if has_reflection[i] and is_correct[i]
        ]

        # Count total, correct, incorrect, and reflection-included samples
        total_correct = sum(is_correct)
        total_incorrect = total_count - total_correct
        reflection_count = sum(has_reflection)

        # 1. Ratio of responses that contain at least one reflection word
        reflection_ratio = reflection_count / total_count if total_count else 0.0

        # 2. Among correct responses, ratio that contain reflection words
        if total_correct > 0:
            correct_with_reflection_count = sum(
                has_reflection[i] for i in range(total_count) if is_correct[i]
            )
            reflection_ratio_in_correct_answers = (
                correct_with_reflection_count / total_correct
            )
        else:
            reflection_ratio_in_correct_answers = 0.0

        # 3. Among incorrect responses, ratio that contain reflection words
        if total_incorrect > 0:
            incorrect_with_reflection_count = sum(
                has_reflection[i] for i in range(total_count) if not is_correct[i]
            )
            reflection_ratio_in_incorrect_answers = (
                incorrect_with_reflection_count / total_incorrect
            )
        else:
            reflection_ratio_in_incorrect_answers = 0.0

        # 4. Among responses with reflection words, ratio that are correct
        if reflection_count > 0:
            correct_in_reflection_texts_count = sum(
                is_correct[i] for i in range(total_count) if has_reflection[i]
            )
            correct_ratio_in_reflection_texts = (
                correct_in_reflection_texts_count / reflection_count
            )
        else:
            correct_ratio_in_reflection_texts = 0.0

        # 5. Among responses without reflection words, ratio that are correct
        no_reflection_count = total_count - reflection_count
        if no_reflection_count > 0:
            correct_in_no_reflection_texts_count = sum(
                is_correct[i] for i in range(total_count) if not has_reflection[i]
            )
            correct_ratio_in_no_reflection_texts = (
                correct_in_no_reflection_texts_count / no_reflection_count
            )
        else:
            correct_ratio_in_no_reflection_texts = 0.0

        # (A) Aggregate all computed statistics
        analysis_dict = {
            "reflection_analysis/reflection_ratio": reflection_ratio,
            "reflection_analysis/reflection_ratio_in_correct_answers": reflection_ratio_in_correct_answers,
            "reflection_analysis/reflection_ratio_in_incorrect_answers": reflection_ratio_in_incorrect_answers,
            "reflection_analysis/correct_ratio_in_reflection_texts": correct_ratio_in_reflection_texts,
            "reflection_analysis/correct_ratio_in_no_reflection_texts": correct_ratio_in_no_reflection_texts,
        }
        # (B) Count total occurrences of each reflection word (accumulated across texts)
        reflection_word_frequency = {
            f"reflection_words/{word}": sum(text.count(word) for text in texts_lower)
            for word in reflection_words
        }

        gathered_analysis_dict = [None for _ in range(self.world_size)]

        dist.all_gather_object(gathered_analysis_dict, analysis_dict)

        analysis_dict = pd.DataFrame(gathered_analysis_dict).mean().to_dict()  # type: ignore

        gathered_reflection_word_frequency = [None for _ in range(self.world_size)]

        dist.all_gather_object(
            gathered_reflection_word_frequency, reflection_word_frequency
        )

        reflection_word_frequency = pd.DataFrame(gathered_reflection_word_frequency).sum().to_dict()  # type: ignore

        if self.is_rank_zero:
            self.metric_logger.log_dict(analysis_dict, step=self.steps_run)
            self.metric_logger.log_dict(reflection_word_frequency, step=self.steps_run)
            if len(reflection_correct_texts) >= 1:
                self.metric_logger.log_table(
                    f"reflection_outputs/Reflection-Correct {self.steps_run}",
                    [reflection_correct_texts[0]],
                    step=self.steps_run,
                )

    @torch.inference_mode()
    def eval(
        self,
        dataloader: DataLoader,
        eval_name: str,
        passk: int,
        generation_strategy: str,
        epoch: int,
    ):
        if generation_strategy == "high":
            eval_generation_kwargs = SamplingParams(
                temperature=1.0,
                max_tokens=2048,
                top_p=1.0,
            )
        elif generation_strategy == "medium":
            eval_generation_kwargs = SamplingParams(
                temperature=0.6,
                max_tokens=2048,
                top_p=1.0,
            )
        elif generation_strategy == "low":
            eval_generation_kwargs = SamplingParams(
                temperature=0.01,
                max_tokens=2048,
                top_p=0.001,
            )
        else:
            raise ValueError(
                f"Invalid evaluation generation strategy: {generation_strategy}"
            )

        acc_cnt = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        total_cnt = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        pbar = tqdm(
            total=len(dataloader),
            disable=not self.is_rank_zero,
            dynamic_ncols=True,
        )
        results = []
        for batch, samples in dataloader:
            correct_per_batch = torch.zeros(len(batch), dtype=torch.bool)
            precitions_passk = [[] for _ in range(len(samples))]
            for _ in range(passk):
                outputs = self.llm.generate(
                    batch, sampling_params=eval_generation_kwargs, use_tqdm=False
                )
                output_texts = [o.outputs[0].text for o in outputs]

                [
                    pred_passk.append(pred)
                    for pred, pred_passk in zip(output_texts, precitions_passk)
                ]
                correct_per_pass = torch.tensor(
                    [
                        self.ds.judge(pred, sample["solution"])
                        for pred, sample in zip(output_texts, samples)
                    ]
                )
                correct_per_batch = correct_per_batch | correct_per_pass

            [sample.pop("images", None) for sample in samples]
            [sample.pop("prompt", None) for sample in samples]
            result = [
                dict(correct=correct, **sample, predictions=pred_passk)
                for correct, pred_passk, sample in zip(
                    correct_per_batch.tolist(),
                    precitions_passk,
                    samples,
                )
            ]
            results += result

            acc_per_batch = torch.sum(correct_per_batch).to(self.device)
            total_per_batch = torch.tensor(len(samples)).float().to(self.device)

            acc_cnt += acc_per_batch
            total_cnt += total_per_batch

            pbar.set_postfix(accuracy=f"{acc_cnt / total_cnt:.3f}")
            pbar.update(1)

        utils.log_rank_zero(
            f"{eval_name}-{generation_strategy} pass@{passk}: {acc_cnt / total_cnt:.3f}-{acc_cnt}/{total_cnt}"
        )
        if self.save_eval_files:
            model_name_dataset_name = f"{self.output_dir.name}-RL-Epoch-{epoch}"
            eval_output_path = (
                Path("outputs")
                / f"rleval_pass@{passk}_{model_name_dataset_name}_{eval_name}_{generation_strategy}_{acc_cnt / total_cnt:.3f}.jsonl"
            )
            utils.save_jsonl(results, eval_output_path, "w")
        return acc_cnt / total_cnt

    def ppo_step(
        self,
        trajectory: rlhf.Trajectory,
        advantages: torch.Tensor,
        context_length: int,
        vision_inputs: dict[str, torch.Tensor],
    ) -> rlhf.PPOStats:
        torch.cuda.empty_cache()

        # Step IV Policy Update: Part 4.1
        # Compute (updated) policy model logprobs (only for response tokens) from the current parameters.
        pi_logits = self.policy_model(
            input_ids=trajectory.query_responses,
            position_ids=trajectory.position_ids,
            attention_mask=trajectory.masks,
            use_cache=False,
            past_key_values=None,
            **vision_inputs,
        ).logits

        pi_logits = rlhf.truncate_sequence_for_logprobs(pi_logits, context_length)
        pi_logprobs = rlhf.logits_to_logprobs(
            pi_logits,
            trajectory.query_responses[:, context_length:],
            self.generation_kwargs.temperature,
        )
        pi_logprobs[trajectory.response_padding_masks] = 1.0

        # Step IV Policy Update: Part 4.2
        # Compute PPO loss and auxiliary metrics (KL, entropy, etc.), then backpropagate.
        loss, policy_loss, kl_loss, entropy, ratios, clipfrac = self.loss_fn(
            trajectory.logprobs,
            pi_logprobs,
            trajectory.ref_logprobs,
            advantages,
            padding_masks=~trajectory.response_padding_masks,
        )
        del pi_logits
        torch.cuda.empty_cache()

        loss /= self.gradient_accumulation_steps
        loss.backward()

        with torch.no_grad():
            approx_policy_kls = (
                0.5 * (pi_logprobs - trajectory.logprobs).pow(2)
            ).mean()

        return rlhf.PPOStats(
            loss,
            policy_loss / self.gradient_accumulation_steps,
            kl_loss / self.gradient_accumulation_steps,
            entropy / self.gradient_accumulation_steps,
            ratios / self.gradient_accumulation_steps,
            clipfrac / self.gradient_accumulation_steps,
            approx_policy_kls / self.gradient_accumulation_steps,
        )

    def log_metrics(
        self,
        trajectory: rlhf.Trajectory,
        ppo_stats: rlhf.PPOStats,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
        tokens_per_second_trajectory: torch.Tensor,
        tokens_per_second_loss: torch.Tensor,
        **extra_metrics,
    ) -> dict[str, torch.Tensor]:
        log_dict = {
            "rews/scores": trajectory.scores.mean().item(),
            "rews/acc_rewards": trajectory.acc_rewards.mean().item(),
            "rews/format_rewards": trajectory.format_rewards.mean().item(),
            "rews/language_rewards": trajectory.language_rewards.mean().item(),
            "rews/rlhf_reward": trajectory.scores.mean().item()
            + kl_rewards.sum(1).mean().item(),
            "rews/kl_reward": kl_rewards.sum(1).mean().item(),
            "loss/loss": ppo_stats.loss.mean().item(),
            "loss/policy_loss": ppo_stats.policy_loss.mean().item(),
            "loss/kl_loss": ppo_stats.kl_loss.mean().item(),
            "training/num_stop_tokens": trajectory.response_padding_masks.any(-1)
            .sum()
            .item(),
            "training/kl": kl.sum(1).mean().item(),
            "training/clipfrac": ppo_stats.clipfrac.mean().item(),
            "training/entropy": ppo_stats.entropy.mean().item(),
            "training/ratios": ppo_stats.ratios.mean().item(),
            "training/approx_policy_kl": ppo_stats.approx_policy_kls.mean().item(),
            "response/response_lengths": trajectory.seq_lens.float().mean().item(),
            "response/length_clip_ratio": (
                trajectory.seq_lens >= (self.generation_kwargs.max_tokens - 1)
            )
            .float()
            .mean()
            .item(),
            "speed/tokens_per_second_per_gpu_trajectory": tokens_per_second_trajectory,
            "speed/tokens_per_second_per_gpu_ppo": tokens_per_second_loss,
            "generation_steps": self.steps_run,
            "gradient_steps": self.global_step,
            **extra_metrics,
        }

        gathered_log_dict = [None for _ in range(self.world_size)]

        dist.all_gather_object(gathered_log_dict, log_dict)

        log_dict = pd.DataFrame(gathered_log_dict).mean().to_dict()  # type: ignore

        if self.is_rank_zero:
            self.metric_logger.log_dict(log_dict, step=self.steps_run)
        return log_dict

    def cleanup_after_step(
        self,
        trajectory: rlhf.Trajectory,
        ppo_stats: rlhf.PPOStats,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
    ) -> None:
        """
        Cleanup tensors after each ppo step to free up memory.
        """
        # there shouldn't be any floating references to the individual tensors at the this point, so gc can do its thing
        for v in trajectory:
            del v
        del trajectory
        for v in ppo_stats:
            del v
        del ppo_stats
        del advantages
        del returns
        del kl
        del kl_rewards

    def save_checkpoint(self, epoch: int):
        save_path = f"{self.output_dir}-RL-Epoch-{epoch}"
        utils.log_rank_zero(f"Saving ckpts to {save_path}")
        cpu_state_dict = training.gather_cpu_state_dict(
            self.policy_model, self.is_rank_zero, self.device
        )
        if self.is_rank_zero:
            self.policy_model.save_pretrained(save_path, state_dict=cpu_state_dict)
            self.processor.save_pretrained(save_path)

            config_path = f"{save_path}/config.json"
            with open(config_path, "r") as file:
                config = json.load(file)
                if "architectures" in config:
                    config["architectures"] = [
                        element.replace("FSDP", "")
                        for element in config["architectures"]
                    ]
            with open(config_path, "w") as file:
                json.dump(config, file, indent=4)

    def cleanup(self) -> None:
        if self.is_rank_zero:
            self.metric_logger.close()

    def _sync_weight_to_vllm(self, model: torch.nn.Module):
        cpu_state_dict = training.gather_cpu_state_dict(
            model, self.is_rank_zero, self.device
        )

        if self.is_rank_zero:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model  # type: ignore
            llm_model.load_weights(cpu_state_dict.items())


@hydra.main(version_base=None, config_path="configs", config_name="full_ppo")
def main(cfg: DictConfig):
    recipe = PPOFullFinetuneRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(main())
