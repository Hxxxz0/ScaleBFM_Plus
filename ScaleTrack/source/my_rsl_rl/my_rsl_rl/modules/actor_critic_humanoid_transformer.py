from __future__ import annotations

import math
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from my_rsl_rl.networks import MLP, HumanoidTransformer, TaskEmbedder

class ActorCriticHumanoidTransformer(nn.Module):

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        std_clamp_range: tuple[float] | list[float] = [0.001, 1.0],
        use_transformer_critic: bool = True,
        embedding_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 4,
        reduced_task_dim: int | None = None,
        task_embedder_hidden_dims: list[int] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups

        num_context = obs['policy'].shape[-2]
        num_actor_prop_obs = obs['policy'].shape[-1]
        num_actor_task_obs = obs['policy_task'].shape[-1]
        num_critic_prop_obs = obs['critic'].shape[-1]
        num_critic_task_obs = obs['critic_task'].shape[-1]
        
        if "mode" in obs:
            assert "mode_mapping" in obs, f"You have to ensure that the mapping is defined between mode and task observation."
            assert obs["mode_mapping"].shape[-1] == num_actor_task_obs, f"The mode mapping does not have the aligned dimension as task observation."
            print(f"Mode Strategy Activated!")
            num_mode_obs = obs['mode'].shape[-1]
            num_actor_task_obs += num_mode_obs

        self.state_dependent_std = state_dependent_std
        self.std_clamp_range = std_clamp_range

        self.actor = HumanoidTransformer(
            prop_obs_dim=num_actor_prop_obs,
            action_dim=num_actions,
            output_dim=2*num_actions if self.state_dependent_std else num_actions,
            embed_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
        )

        self.actor_task_embedder = TaskEmbedder(
            task_obs_dim=num_actor_task_obs,
            embedding_dim=embedding_dim,
            reduced_task_dim=reduced_task_dim,
            hidden_dims=task_embedder_hidden_dims
        )

        print(f"Actor: {self.actor}")
        print(f"Actor Task Embedder: {self.actor_task_embedder}")

        # Critic
        self.use_transformer_critic = use_transformer_critic
        if use_transformer_critic:
            self.critic = HumanoidTransformer(
                prop_obs_dim=num_critic_prop_obs,
                action_dim=num_actions,
                output_dim=1,
                embed_dim=embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers,
            )
            self.critic_task_embedder = TaskEmbedder(
                task_obs_dim=num_critic_task_obs,
                embedding_dim=embedding_dim,
                reduced_task_dim=None, # Critic does not use low-rank
                hidden_dims=task_embedder_hidden_dims
            )
            print(f"Critic Task Embedder: {self.critic_task_embedder}")
        else:
            self.critic = MLP(num_critic_prop_obs*num_context+num_actions*(num_context-1)+obs["critic_task"].flatten(start_dim=1).shape[-1], 1, critic_hidden_dims, activation)
        
        print(f"Critic: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            head_init_val = torch.zeros(2*num_actions)
            if self.noise_std_type == "scalar":
                head_init_val[num_actions:] = init_noise_std
                self.actor.init_weights(head_init_val)
            elif self.noise_std_type == "log":
                head_init_val[num_actions:] = torch.log(torch.tensor(init_noise_std + 1e-7))
                self.actor.init_weights(head_init_val)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            self.actor.init_weights()
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.actor_task_embedder.init_weights()

        if use_transformer_critic:
            self.critic.init_weights()
            self.critic_task_embedder.init_weights()

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, prop_obs: TensorDict, task_obs: TensorDict, action_obs: TensorDict) -> None:
        task_tokens = self.actor_task_embedder(task_obs)
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(prop_obs, action_obs, task_tokens)
            if self.noise_std_type == "scalar":
                mean, std = torch.chunk(mean_and_std, 2, dim=-1)
                std = torch.clamp(std, min=self.std_clamp_range[0], max=self.std_clamp_range[1])
            elif self.noise_std_type == "log":
                mean, log_std = torch.chunk(mean_and_std, 2, dim=-1)
                log_std = torch.clamp(log_std, min=self.std_clamp_range[0], max=self.std_clamp_range[1])
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(prop_obs, action_obs, task_tokens)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_prop_obs, actor_task_obs, actor_action_obs = self.get_actor_obs(obs)
        self._update_distribution(actor_prop_obs, actor_task_obs, actor_action_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_prop_obs, actor_task_obs, actor_action_obs = self.get_actor_obs(obs, inference=True)
        actor_task_tokens = self.actor_task_embedder(actor_task_obs)
        if self.state_dependent_std:
            mean_and_std = self.actor(actor_prop_obs, actor_action_obs, actor_task_tokens)
            mean, _ = torch.chunk(mean_and_std, 2, dim=-1)
        else:
            mean = self.actor(actor_prop_obs, actor_action_obs, actor_task_tokens)
        return mean

    def actor_mean_from_parts(
        self,
        prop_obs: torch.Tensor,
        task_obs: torch.Tensor,
        action_obs: torch.Tensor,
        mode: torch.Tensor,
        mode_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic action of the training-mode (masked) actor from separate, differentiable inputs.

        Same computation as ``get_actor_obs`` + ``act_inference`` with ``inference=False``, but without in-place
        operations so it can be back-propagated through (prop_obs (N, C, P), task_obs (N, F, T), action_obs
        (N, C, A), mode (N, M), mode_mapping (N, T)).
        """
        task_obs = task_obs * mode_mapping.unsqueeze(-2)
        task_obs = torch.cat([task_obs, mode.unsqueeze(-2).expand(-1, task_obs.shape[-2], -1)], dim=-1)
        out = self.actor(prop_obs, action_obs, self.actor_task_embedder(task_obs))
        if self.state_dependent_std:
            out = torch.chunk(out, 2, dim=-1)[0]
        return out

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        prop_obs, task_obs, action_obs = self.get_critic_obs(obs)
        if self.use_transformer_critic:
            task_tokens = self.critic_task_embedder(task_obs)
            value = self.critic(prop_obs, action_obs, task_tokens)
        else:
            context = torch.cat([
                prop_obs.reshape(prop_obs.shape[0], -1),
                task_obs.reshape(prop_obs.shape[0], -1),
                action_obs[:,1:].reshape(prop_obs.shape[0], -1)
            ], dim=-1)
            value = self.critic(context)
        return value

    def get_actor_obs(self, obs: TensorDict, inference: bool = False):
        prop_obs = obs['policy']
        if prop_obs.dim() == 2:
            prop_obs = prop_obs.unsqueeze(-2)
        task_obs = obs['policy_task']
        if task_obs.dim() == 2:
            task_obs = task_obs.unsqueeze(-2)
        action_obs = obs["action"]
        if action_obs.dim() == 2:
            action_obs = action_obs.unsqueeze(-2)

        if 'mode' in obs:

            mode_mapping = obs['mode_mapping']
            if mode_mapping.dim() == 2:
                    mode_mapping = mode_mapping.unsqueeze(-2).repeat(1, task_obs.shape[-2], 1)
            mode_obs = obs["mode"]
            if mode_obs.dim() == 2:
                mode_obs = mode_obs.unsqueeze(-2).repeat(1, task_obs.shape[-2], 1)
            
            if not inference:
                task_obs *= mode_mapping
            else:
                mode_obs = torch.ones_like(mode_obs, device=mode_obs.device)

            task_obs = torch.cat([task_obs, mode_obs], dim=-1)

        return prop_obs, task_obs, action_obs

    def get_critic_obs(self, obs: TensorDict):
        prop_obs = obs['critic']
        if prop_obs.dim() == 2:
            prop_obs = prop_obs.unsqueeze(-2)
        task_obs = obs['critic_task']
        if task_obs.dim() == 2:
            task_obs = task_obs.unsqueeze(-2)
        action_obs = obs["action"]
        if action_obs.dim() == 2:
            action_obs = action_obs.unsqueeze(-2)
        return prop_obs, task_obs, action_obs

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        pass
    
    def synchronize_normalizers(self, gpu_world_size: int, gpu_global_rank: int) -> None:
        """Synchronize observation normalizer statistics across all ranks."""
        pass

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
    
    @property
    def actor_parameters(self):
        params = list(self.actor.parameters())
        params += list(self.actor_task_embedder.parameters())

        if not self.state_dependent_std:
            if self.noise_std_type == "scalar":
                params.append(self.std)
            elif self.noise_std_type == "log":
                params.append(self.log_std)
        
        return params
    
    @property
    def critic_parameters(self):
        params = list(self.critic.parameters())
        
        if self.use_transformer_critic:
            params += list(self.critic_task_embedder.parameters())

        return params