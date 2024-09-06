import time
import warnings
from typing import Any, Callable, Dict, Optional, Union
from random import sample
import gymnasium as gym
from tianshou.data import (
    Batch,
    to_numpy,
)
from tianshou.data import ReplayBuffer, Collector
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy

from .config.config import *


class EnvCollector(Collector):
    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
        env_func: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)
        self.env_func = env_func

    def collect(
            self,
            n_step: Optional[int] = None,
            n_episode: Optional[int] = None,
            random: bool = False,
            random_way: str = None,
            render: Optional[float] = None,
            no_grad: bool = True,
            gym_reset_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            sample_ids = sample(range(self.env_num), min(n_episode, self.env_num))
            self.data = self.data[sample_ids]
            # self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        step_time = []
        step_energy = []
        episode_time = []
        episode_energy = []
        act_distribution = []


        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                mask = [1 for _ in range(NUM_RESOURCE_CLUSTER)]
                if random_way == "remote":
                    mask[0] = 0
                elif random_way == "local":
                    mask[1:] = [0] * len(mask[1:])
                mask = np.array(mask, dtype=np.int8)
                try:
                    act_sample = [
                        self._action_space[i].sample(mask) for i in ready_env_ids
                    ]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample(mask) for _ in ready_env_ids]
                if random_way == "heft":
                    act_sample = [self.env.workers[env].env.min_available_time_resource for env in ready_env_ids]
                if random_way == "greedy":
                    def get_res(env):
                        task_len = len(env.hypergraph.tasks) // NUM_RESOURCE_CLUSTER
                        return min(env.current_task.sorted_id // task_len, NUM_RESOURCE_CLUSTER - 1)
                    act_sample = [get_res(self.env.workers[env].env) for env in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,  # type: ignore
                ready_env_ids
            )
            done = np.logical_or(terminated, truncated)

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                        act=self.data.act,
                    )
                )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(
                    env_ind_local, env_ind_global, gym_reset_kwargs
                )
                for i in env_ind_local:
                    self._reset_state(i)

                env_infos = info[env_ind_local]
                for env_info in env_infos:
                    step_time.append(np.mean(env_info["step_time"]))
                    step_energy.append(np.mean(env_info["step_energy"]))
                    act_distribution.append(env_info["act_distribution"])
                    episode_time.append(env_info["episode_time"])
                    episode_energy.append(env_info["episode_energy"])

                if self.env_func:
                    self.env_func(env_infos)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        act_distribution = np.sum(np.array(act_distribution), axis=0) / len(act_distribution)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(
                map(
                    np.concatenate,
                    [episode_rews, episode_lens, episode_start_indices]
                )
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
            "time/st": np.mean(step_time),
            "energy/st": np.mean(step_energy),
            "time/ep": np.mean(episode_time),
            "energy/ep": np.mean(episode_energy),
            "act/ep": act_distribution
        }