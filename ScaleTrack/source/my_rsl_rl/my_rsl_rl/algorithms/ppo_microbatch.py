"""Bound PPO activation memory while preserving full optimizer minibatches and FP32 precision."""

import torch
from torch import nn


def adapt_learning_rate(alg, kl_mean):
    if alg.is_multi_gpu:
        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
        kl_mean /= alg.gpu_world_size
    if alg.gpu_global_rank == 0:
        if kl_mean > alg.desired_kl * 2.0:
            alg.actor_learning_rate = max(1e-5, alg.actor_learning_rate / 1.5)
            alg.critic_learning_rate = max(1e-5, alg.critic_learning_rate / 1.5)
        elif kl_mean < alg.desired_kl / 2.0 and kl_mean > 0.0:
            alg.actor_learning_rate = min(1e-2, alg.actor_learning_rate * 1.5)
            alg.critic_learning_rate = min(1e-2, alg.critic_learning_rate * 1.5)
    if alg.is_multi_gpu:
        lrs = torch.tensor([alg.actor_learning_rate, alg.critic_learning_rate], device=alg.device)
        torch.distributed.broadcast(lrs, src=0)
        alg.actor_learning_rate, alg.critic_learning_rate = lrs.tolist()
    for optimizer, lr in ((alg.actor_optimizer, alg.actor_learning_rate),
                          (alg.critic_optimizer, alg.critic_learning_rate)):
        for group in optimizer.param_groups:
            group["lr"] = lr


def update(alg):
    totals = torch.zeros(3, device=alg.device)
    generator = alg.storage.mini_batch_generator(alg.num_mini_batches, alg.num_learning_epochs)
    updates = 0
    for batch in generator:
        n = batch[0].batch_size[0]
        if not n:
            raise ValueError("empty PPO optimizer minibatch")
        alg.actor_optimizer.zero_grad()
        alg.critic_optimizer.zero_grad()
        kl_mean = torch.zeros((), device=alg.device)
        for lo in range(0, n, alg.forward_micro_batch_size):
            hi = min(n, lo + alg.forward_micro_batch_size)
            obs, actions, old_values, advantages, returns, old_log_prob, old_mu, old_sigma = (
                item[lo:hi] for item in batch
            )
            alg.policy.act(obs)
            log_prob = alg.policy.get_actions_log_prob(actions)
            value = alg.policy.evaluate(obs)
            entropy = alg.policy.entropy.mean()
            scale = (hi - lo) / n
            if alg.desired_kl is not None and alg.schedule == "adaptive":
                with torch.no_grad():
                    mu, sigma = alg.policy.action_mean, alg.policy.action_std
                    kl = (torch.log(sigma / old_sigma + 1e-5)
                          + (old_sigma.square() + (old_mu - mu).square()) / (2.0 * sigma.square()) - 0.5)
                    kl_mean += kl.sum(-1).mean() * scale
            ratio = torch.exp(log_prob - old_log_prob.squeeze(-1))
            adv = advantages.squeeze(-1)
            surrogate = torch.maximum(-adv * ratio,
                                      -adv * ratio.clamp(1.0 - alg.clip_param, 1.0 + alg.clip_param)).mean()
            if alg.use_clipped_value_loss:
                clipped = old_values + (value - old_values).clamp(-alg.clip_param, alg.clip_param)
                value_loss = torch.maximum((value - returns).square(), (clipped - returns).square()).mean()
            else:
                value_loss = (returns - value).square().mean()
            loss = surrogate + alg.value_loss_coef * value_loss - alg.entropy_coef * entropy
            (loss * scale).backward()
            totals += torch.stack((value_loss.detach(), surrogate.detach(), entropy.detach())) * scale
        # LR adaptation, synchronization, clipping and both Adam steps occur once per original minibatch.
        if alg.desired_kl is not None and alg.schedule == "adaptive":
            adapt_learning_rate(alg, kl_mean)
        if alg.is_multi_gpu:
            alg.reduce_parameters()
        nn.utils.clip_grad_norm_(alg.policy.parameters(), alg.max_grad_norm)
        alg.actor_optimizer.step()
        alg.critic_optimizer.step()
        updates += 1
    if updates != alg.num_learning_epochs * alg.num_mini_batches:
        raise ValueError("PPO storage yielded an unexpected optimizer step count")
    alg.storage.clear()
    return dict(zip(("value_function", "surrogate", "entropy"), (totals / updates).tolist()))
