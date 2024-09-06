import os

import gymnasium as gym
import torch
from tianshou.data import VectorReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net, ActorCritic

from environment.config.config import *
from environment.env_collect import EnvCollector
from environment.offloading_env import make_env

from environment.config import config
from utils.utils import is_float, get_mean
from utils.utils import get_file_paths, get_gpu

utilization_rates = []
end_times = []
makespans = []


def collect_test(policy, envs, name=config.GNN_TYPE, random=False, env_func=None):
    # _env = get_env_from_vector_envs(envs)
    # _env.reset()
    # policy.eval()
    if env_func:
        utilization_rates.append([])
        end_times.append([])
    collector = EnvCollector(policy, envs, VectorReplayBuffer(BUFFER_SIZE, len(envs)), env_func=env_func)
    collector_stats = collector.collect(random=random, random_way=name, n_episode=BATCH_SIZE)
    print(f"{name}:")
    for k, v in collector_stats.items():
        if not k.endswith("s"):
            print(f"{k}: {v}", end=' ')
    print()
    if env_func:
        mean_utilization_rate = get_mean(utilization_rates[-1])
        mean_end_time = get_mean(end_times[-1])
        print(f"Mean utilization rate: {mean_utilization_rate}")
        print(f"Mean end time: {mean_end_time}")
        print(f"Makespan: {makespans[-1]}")

def init_env(env, gnn_type=GNN_TYPE):
    use_graph_state = USE_GRAPH_STATE
    if isinstance(env, gym.Env):
        setattr(env, "use_graph_state", use_graph_state)
        setattr(env, "_init_state", None)
        env.reset()
    elif isinstance(env, BaseVectorEnv):
        env.set_env_attr("use_graph_state", use_graph_state)
        env.set_env_attr("_init_state", None)
        env.set_env_attr("_gnn_type", gnn_type)
        env.set_env_attr("_HG", None)

def set_env_attr(env, key, value):
    if isinstance(env, gym.Env):
        setattr(env, key, value)
    elif isinstance(env, BaseVectorEnv):
        env.set_env_attr(key, value)

def env_func(env_infos):
    utilizations = []
    _end_times = []
    sum_makespan = 0
    for env_info in env_infos:
        acts = env_info["acts"]
        exec_time = env_info["exec_time"]
        end_time = list(env_info["end_time"].values())
        makespan = env_info["episode_time"]
        sum_makespan += makespan
        assert makespan == max(end_time), f"makespan: {makespan} != end_time: {max(end_time)}"
        utilization = [0] * NUM_RESOURCE_CLUSTER
        mx_end_time = env_info["curr_time"]

        for task_id, task_exec_time in exec_time.items():
            utilization[acts[task_id]] += task_exec_time
        for i in range(NUM_RESOURCE_CLUSTER):
            utilization[i] /= mx_end_time
        utilizations.append(utilization)
        _end_times.append(sorted(end_time))
    utilization_rates[-1].append(get_mean(utilizations))
    end_times[-1].append(get_mean(_end_times))
    makespans.append(sum_makespan / len(env_infos))


if __name__ == "__main__":


    training_paths, test_paths = get_file_paths()

    env, train_envs, test_envs = make_env("HG", training_paths, test_paths, GNN_TYPE, use_graph_state=USE_GRAPH_STATE,
                                          device=get_gpu(), use_cache=False)

    actor = Net(env.observation_space.shape, hidden_sizes=[128, 64], device=get_gpu())
    critic = Net(env.observation_space.shape, hidden_sizes=[128, 64], device=get_gpu())
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
    dist = torch.distributions.Categorical

    policy = PPOPolicy(actor=actor, critic=critic, optim=optim, dist_fn=dist, action_space=env.action_space, observation_space=env.observation_space.shape)

    collect_test(policy, train_envs, name="heft", random=True, env_func=env_func)

    collect_test(policy, train_envs, name="greedy", random=True, env_func=env_func)

    collect_test(policy, train_envs, name="remote", random=True, env_func=env_func)

    collect_test(policy, train_envs, name="local", random=True, env_func=env_func)

    collect_test(policy, train_envs, name="random", random=True, env_func=env_func)


