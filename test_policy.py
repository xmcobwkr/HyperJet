import os

import gymnasium as gym
import torch
from tianshou.data import VectorReplayBuffer
from tianshou.env import BaseVectorEnv

from environment.config.config import *
from environment.env_collect import EnvCollector
from environment.hypergraph import Hypergraph
from environment.offloading_env import make_env

from environment.config import config
from utils.utils import is_float, get_mean

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
        # mean_utilization_rate = [0 for _ in range(NUM_RESOURCE_CLUSTER)]
        # for utilization_rate in utilization_rates[-1]:
        #     for i in range(NUM_RESOURCE_CLUSTER):
        #         mean_utilization_rate[i] += utilization_rate[i]
        # for i in range(NUM_RESOURCE_CLUSTER):
        #     mean_utilization_rate[i] /= len(utilization_rates[-1])
        mean_utilization_rate = get_mean(utilization_rates[-1])
        mean_end_time = get_mean(end_times[-1])
        print(f"Mean utilization rate: {mean_utilization_rate}")
        print(f"Mean end time: {mean_end_time}")
        print(f"Makespan: {makespans[-1]}")

    # print(f"{name}: {collector_stats}")

def load_policy(_policy, path, file_name="policy.pth") -> None:
    file_path = os.path.join(path, file_name)
    print(file_path)
    _policy.load_state_dict(torch.load(file_path))

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

def deadline(env_infos):
    for env_info in env_infos:
        end_time = env_info["end_time"]

    return True

if __name__ == "__main__":
    from test import get_env_from_vector_envs, get_file_paths, get_gpu, get_policy

    training_paths, test_paths = get_file_paths()

    env, train_envs, test_envs = make_env("HG", training_paths, test_paths, GNN_TYPE, use_graph_state=USE_GRAPH_STATE,
                                          device=get_gpu(), use_cache=False)
    env.reset()
    def net_test(path, gnn_type=GNN_TYPE):
        config.GNN_TYPE = gnn_type
        init_env(env, gnn_type)
        init_env(train_envs, gnn_type)
        _policy = get_policy(env, gnn_type)
        load_policy(_policy, path)
        collect_test(_policy, train_envs, name=gnn_type, env_func=env_func)

    # mlp
    # net_test("./log/default/HG/ppo/HGNN/Jun05-050404-Seq2Seq-[0 2]-task40-edge700", "HGNN")

    def test_weight():
        file_dir = '/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN'
        # file_dirs = [
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul01-055603-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.2",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun21-082625-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.4"
        #
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul02-130841-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-1.0",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul02-130841-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.8",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul02-130841-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.6",
        #
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul02-130805-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.2",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul02-130805-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.4",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul02-130805-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.6",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul02-130805-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.8",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul02-130805-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-1.0",
        #
        #
        #
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-015749-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.4",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-015749-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-1.0",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-015749-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-1.0",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-015749-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.8",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-015749-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.2",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-015749-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.6",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-015749-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.2",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-015749-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.6",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-015749-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.8",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-020727-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.2",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-020730-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.4",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-020738-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.8",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-020738-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-1.0",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-020748-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.2",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-020752-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.6",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun26-020752-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.4",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun27-103508-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.6",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun27-103508-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.8",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun27-103508-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.2",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun27-103508-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.4",
        #     # "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun28-061705-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-1.0"
        # ]
        # file_dirs = [
        #     # "/data/huangkang/hypergraph_offloading/log/std/hg20_seq2seq_1234-1.0-1.0",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-023851-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-1.0",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-023856-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.8",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-023901-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.6",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-023907-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.4",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-023909-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.2",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-1.0",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.2",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.4",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.6",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.8",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-1.0",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.4",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.2",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.6",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.2",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.8",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-062929-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-1.0",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-132331-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.6",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-132331-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.6",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-132331-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-1.0",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-132331-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.4",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-132331-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.8",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-132331-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.8",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-132331-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.4",
        #     "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul04-132331-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.2",
        # ]

        file_dirs = [
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035526-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-1.0",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035531-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.8",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035533-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.6",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035617-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.2",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035617-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-1.0-0.4",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035633-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.8",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035633-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.6",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035638-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.4",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035641-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-0.2",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035651-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.8-1.0",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035657-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.4",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035657-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.2",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035701-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.6",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035708-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-0.8",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-035724-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.6-1.0",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-070248-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-1.0",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-070248-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.4",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-070248-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.2",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-070248-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.6",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-070248-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.4-0.8",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-072257-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.2",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-072257-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.8",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-072257-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.4",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-072257-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-1.0",
            "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul05-072257-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.6",
        ]


        mapp = {}
        for dir in file_dirs:
            dir_arr = dir.split('-')
            if len(dir_arr) > 2 and is_float(dir_arr[-1]) and is_float(dir_arr[-2]):
                a, b = float(dir_arr[-2]), float(dir_arr[-1])
                mapp[(a, b)] = dir
        for k, v in mapp.items():
            file_name = v
            set_env_attr(train_envs, "_time_weight", k[0])
            set_env_attr(train_envs, "_energy_weight", k[1])
            print(f"time_weight: {k[0]}, energy_weight: {k[1]}")
            net_test(file_name, "HGNN")

    hypergraphs = train_envs.get_env_attr("hypergraph")
    for hg in hypergraphs:
        dag_edge = 0
        for edge in hg.edges:
            nodes = len(edge.nodes)
            if hasattr(edge, "source") and edge.source != -1:
                dag_edge += 1
            else:
                dag_edge += nodes * (nodes - 1)
        print(len(hg.tasks), len(hg.edges), dag_edge)
    exit(0)


    # test_weight()
    # net_test('/data/huangkang/hypergraph_offloading/log/exp2/HG/ppo/HGNN/Jun17-092515-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini', "HGNN")
    #
    # # gcn
    # net_test("./log/default/HG/ppo/GCN/Mar29-011605", "GCN")
    #
    # # hgnn
    # net_test("./log/default/HG/ppo/HGNN/Mar28-123144", "HGNN")
    # net_test("./log/default/HG/ppo/HGNN/Jul17-070006-Seq2Seq-[0 3]-task30-edge420-km1_rKaHyPar_sea20.ini-1.0-1.0", "HGNN")
    # net_test("/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jul19-051446-Seq2Seq-[0 1 2 3]-task50-edge144-km1_rKaHyPar_sea20.ini-1.0-1.0",
    #          "HGNN")
    # net_test("./log/default/HG/ppo/GCN/Jul17-162906-Seq2Seq-[0]-task40-edge60-km1_rKaHyPar_sea20.ini-1.0-1.0", "GCN")
    # net_test("/data/huangkang/hypergraph_offloading/log/std/hg20_seq2seq_1234-1.0-1.0", "HGNN")

    # net_test("./log/default/HG/ppo/GCN/Jul17-162906-Seq2Seq-[0]-task40-edge60-km1_rKaHyPar_sea20.ini-1.0-1.0", "GCN")
    net_test("/data/huangkang/hypergraph_offloading/log/default/HG/ppo/Seq2Seq/Jul10-121223-Seq2Seq-[0]-task40-edge600-km1_rKaHyPar_sea20.ini-1.0-1.0", None)
    #


    policy = get_policy(env)

    collect_test(policy, train_envs, name="heft", random=True, env_func=env_func)

    # random
    collect_test(policy, train_envs, name="greedy", random=True, env_func=env_func)

    # remote
    collect_test(policy, train_envs, name="remote", random=True, env_func=env_func)
    # local
    collect_test(policy, train_envs, name="local", random=True, env_func=env_func)
    # random
    collect_test(policy, train_envs, name="random", random=True, env_func=env_func)


