"""
Custom training entry point for Approach B recurrent memory agent.

Reuses verl's TaskRunner but replaces RayPPOTrainer with RecurrentRayPPOTrainer.

Usage:
    python3 -m onepred.main_ppo_recurrent [hydra overrides...]
"""

import os
import hydra
from verl.trainer.main_ppo import TaskRunner, run_ppo, create_rl_dataset, create_rl_sampler
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.utils.device import auto_set_device

from onepred.recurrent_ray_trainer import RecurrentRayPPOTrainer

import ray

# Resolve verl's config path for hydra
import verl.trainer.config as _verl_config_module
_VERL_CONFIG_PATH = os.path.dirname(_verl_config_module.__file__)


class RecurrentTaskRunner(TaskRunner):
    """TaskRunner that uses RecurrentRayPPOTrainer instead of RayPPOTrainer."""

    def run(self, config):
        import os
        import socket
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        from verl.utils.config import validate_config
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.dataset.rl_dataset import collate_fn

        print(f"RecurrentTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Use RecurrentRayPPOTrainer instead of RayPPOTrainer
        trainer = RecurrentRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path=_VERL_CONFIG_PATH, config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    task_runner_class = ray.remote(num_cpus=1)(RecurrentTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
