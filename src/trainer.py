"""
Huggingface trainer
"""

from transformers.trainer import *
from trl import SFTTrainer


class DeepSpeedRemoveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            if f"global_step{state.global_step}" in os.listdir(checkpoint_path):
                shutil.rmtree(
                    os.path.join(checkpoint_path, f"global_step{state.global_step}")
                )


class SFTTrainerNoDeepspeedSave(SFTTrainer):
    """same as SFTTrainer, just skip deepspeed save_checkpoint"""

    # NOTE: transformers>4.35에서는 아래의 checkpoint 부분 수정 필요

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        # Save optimizer and scheduler
        if self.fsdp or self.is_fsdp_enabled:
            if self.is_fsdp_enabled:
                save_fsdp_optimizer(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    self.optimizer,
                    self.model,
                    output_dir,
                )
            else:
                # FSDP has a different interface for saving optimizer states.
                # Needs to be called on all ranks to gather all states.
                # full_optim_state_dict will be deprecated after Pytorch 2.2!
                full_osd = self.model.__class__.full_optim_state_dict(
                    self.model, self.optimizer
                )
                torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(
                self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME)
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
        elif (
            self.args.should_save
            and not self.is_deepspeed_enabled
            and not (self.fsdp or self.is_fsdp_enabled)
        ):
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(
                self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME)
            )

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_tpu_available()
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )
            reissue_pt_warnings(caught_warnings)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        if is_torch_npu_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                rng_states["npu"] = torch.npu.random.get_rng_state_all()
            else:
                rng_states["npu"] = torch.npu.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(
                rng_states,
                os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"),
            )

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif self.fsdp is not None or self.is_fsdp_enabled:
            state_dict = self.model.state_dict() if not self.is_fsdp_enabled else {}
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if self.is_fsdp_enabled:
                # remove the dummy state_dict
                remove_dummy_checkpoint(
                    self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
                )
                save_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    self.model,
                    output_dir,
                )

        elif self.is_deepspeed_enabled:
            # this takes care of everything as long as we aren't under zero3
            if version.parse(accelerate_version) <= version.parse("0.20.3"):
                raise ValueError("Install Accelerate from main branch")
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(
                    self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
                )
                # self.model_wrapped.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
