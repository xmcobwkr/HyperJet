import heapq
import os
import pickle
from os import path

import dill as dill
import gymnasium as gym
import torch
from gymnasium.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv

from environment.hypergraph import Hypergraph, HypergraphData
from environment.resource import Resources, resources
from .config.config import *
from utils.utils import *

from utils.utils import CustomException


def _get_transmission_time(task):
    return int(task.input_data_size) / task.bandwidth_up, int(task.output_data_size) / task.bandwidth_dl

class OffloadingEnvironment(gym.Env):
    """
        创建卸载的强化学习环境，环境需要制定状态，动作，奖励以及价值
        需要的为输入的超图集合（包含了相应的节点，超边属性信息），资源池环境
        状态需要超图的数据集以及当前的卸载情况
        构造函数参数：
            资源池环境：包含了资源池相关信息的环境
            超图样本：包含了状态中所需要的一个样本
    """

    def __init__(self, resources: Resources, hypergraph: Hypergraph, gnn_type=None, device=None, use_graph_state=False,
                 time_weight=REWARD_WEIGHT["time"], energy_weight=REWARD_WEIGHT["energy"]):
        super(OffloadingEnvironment, self).__init__()
        # 初始化环境
        # self._min_exec_time = None
        # self._min_total_time = None
        # self._max_total_time = None
        # self._min_time = None
        # self._max_energy = None
        # self._max_exec_time = None
        self._dl_time = {}
        self._up_time = {}
        self._dl_energy = {}
        self._up_energy = {}
        self._HG = None
        # self._min_energy = None
        # self._max_time = None
        self._init_state = None
        self._state = None
        self.use_graph_state = use_graph_state
        self._resources = resources
        self._hypergraph = hypergraph
        self._gnn_type = gnn_type
        self._device = device
        self._current_task = None
        self._reward = 0
        self._num_steps = 0
        self._terminated = False
        self._global_clock = 0
        self._global_energy = 0
        self._num_offloading = 0
        self._num_offloading_steps = 0
        self.reset()
        self._observation_space = Box(shape=self.state.shape, low=0, high=1)
        self._action_space = Discrete(NUM_RESOURCE_CLUSTER)
        self._step_time = None
        self._step_energy = None
        self._acts = None
        self._act_distribution = None
        self._time_weight = time_weight
        self._energy_weight = energy_weight


    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def state(self):
        """返回观察到的状态"""
        return self._state

    @property
    def init_state(self):
        """初始化的状态，每次从起始节点卸载都会变初始化的状态"""
        # 状态由节点数量由节点数量 * (超边数量 + 资源池数量 * 3（是否卸载对应资源池，时延，能耗） + 2(能耗，执行时间))，对于还没有卸载的节点的后2个元素应该使用掩码0覆盖
        # 对于超边就是是否连接，对于资源池是如果卸载该节点时对应的时延和能耗，超边和资源池是初始化的状态就有，后2个则是需要通过action不断更新的
        # 节点数量和超边数量都是固定的，多出删减，不够补全零信息的节点

        if self._init_state is None:
            total_edges = NUM_HG * NUM_HG_EDGES if self.use_graph_state else 0

            n = NUM_HG * NUM_HG_TASKS
            m = total_edges + NUM_RESOURCE_CLUSTER * 2 + 8
            _state = np.zeros((n, m))
            incidence_matrix = self.hypergraph.incidence_matrix
            p, _= incidence_matrix.shape
            # if self.use_graph_state:
            #     _state[:min(n, p), :min(m, q)] = incidence_matrix[:min(n, p), :min(m, q)]

            for i in range(min(n, p)):
                _task = self.hypergraph.task_sequence[i]
                for j in range(0, NUM_RESOURCE_CLUSTER):
                    _state[i, j + total_edges] = self.resources.get_time(_task, j)
                    _state[i, j + total_edges + NUM_RESOURCE_CLUSTER + 4] = self.resources.get_energy(_task, j)
                _state[i, total_edges + NUM_RESOURCE_CLUSTER] = self._up_time[_task.id] + self._dl_time[_task.id]
                _state[i, total_edges + NUM_RESOURCE_CLUSTER*2 + 4] = self._up_energy[_task.id] + self._dl_energy[_task.id]
            self._init_state = torch.tensor(_state, dtype=torch.float32)

        return self._init_state

    @property
    def reward(self):
        """获取当前执行动作后的即时奖励"""
        return self._reward

    @property
    def hypergraph(self):
        return self._hypergraph

    @property
    def resources(self):
        return self._resources

    @property
    def terminated(self):
        return self._terminated

    @property
    def current_task(self):
        return self._current_task

    @property
    def start_time(self):
        return self._start_time[self._current_task.id]

    @property
    def end_time(self):
        return self._end_time[self._current_task.id]

    @property
    def exec_time(self):
        return self._exec_time[self._current_task.id]

    @property
    def exec_energy(self):
        return self._exec_energy[self._current_task.id]

    @property
    def upload_time(self):
        return self._up_time[self._current_task.id]

    @property
    def download_time(self):
        return self._dl_time[self._current_task.id]

    @property
    def upload_energy(self):
        return self._up_energy[self._current_task.id]

    @property
    def download_energy(self):
        return self._dl_energy[self._current_task.id]

    @property
    def energy(self):
        return self._energy[self._current_task.id]

    # @property
    # def max_exec_time(self):
    #     return self._max_exec_time
    #
    # @property
    # def min_exec_time(self):
    #     return self._min_exec_time
    #
    # @property
    # def min_total_time(self):
    #     return self._min_total_time
    #
    # @property
    # def max_total_time(self):
    #     return self._max_total_time
    #
    # @property
    # def max_energy(self):
    #     return self._max_energy
    #
    # @property
    # def min_energy(self):
    #     return self._min_energy

    @property
    def HG(self):
        if self._HG is None and self._gnn_type is not None:
            _edge_list = self.hypergraph.undirected_edge_list
            def check_edge(edges):
                # 去除超边中不在裁剪范围内的节点
                edge_list = []
                for edge in edges:
                    new_edge = []
                    for node in edge:
                        if node < self._state.shape[0]:
                            new_edge.append(node)
                    if len(new_edge) > 1:
                        edge_list.append(new_edge)
                return edge_list

            edge_list = check_edge(_edge_list)
            from dhg import Hypergraph, Graph
            self._HG = Hypergraph(self._state.shape[0], edge_list)
            self._HG.to(self._device)
            if self._gnn_type == "GCN":
                self._HG = Graph.from_hypergraph_clique(self._HG, weighted=True)
                self._HG.to(self._device)
            elif self._gnn_type == "DHGNN+":
                source_edge_list, target_edge_list = self.hypergraph.directed_edge_list
                self._HG = (self._HG, Hypergraph(self._state.shape[0], source_edge_list),
                            Hypergraph(self._state.shape[0], target_edge_list))
                for v in self._HG:
                    v.to(self._device)

        return self._HG

    @property
    def info(self):
        return {"num_steps": self._num_steps, "HG": self.HG, "curr_time": self._global_clock,
                "step_time": self._step_time, "step_energy": self._step_energy, "episode_time": self._global_clock,
                "episode_energy": self._global_energy, "act_distribution": self._act_distribution, "acts": self._acts,
                "id": self.current_task.topsort_id, "end_time": self._end_time, "exec_time": self._exec_time}

    @property
    def min_available_time_resource(self):
        """贪心的选择"""
        return np.argmin(self._resource_available_time)
        # last_column = [row[-1] for row in self._resource_available_time]
        # return last_column.index(min(last_column))


    def reset(self, seed=None, options=None):
        """
        :return: 初始化的状态，以及相关的info
        """
        # self._max_time = {task.id: self.resources.get_max_time(task)[0] for task in self.hypergraph.tasks}
        # self._min_time = {task.id: self.resources.get_min_time(task)[0] for task in self.hypergraph.tasks}
        # self._max_energy = max(self.resources.get_max_energy(task)[0] for task in self.hypergraph.tasks)
        # self._min_energy = min(self.resources.get_min_energy(task)[0] for task in self.hypergraph.tasks)
        for task in self.hypergraph.tasks:
            _id = task.id
            self._up_time[_id], self._dl_time[_id], self._up_energy[_id], self._dl_energy[_id] = get_transmission_info(task)
        # self._max_exec_time = max(self._max_time.values())
        # self._min_exec_time = min(self._min_time.values())
        # self._max_total_time = max({ self._max_time[task.id] + self._up_time[task.id] + self._dl_time[task.id]
        #                              for task in self.hypergraph.tasks})
        # self._min_total_time = min({ self._min_time[task.id] + self._up_time[task.id] + self._dl_time[task.id]
        #                              for task in self.hypergraph.tasks})
        
        self._state, self._reward = self._get_next_obs()
        self._global_clock = self._global_energy = 0
        max_obs = np.ones(self.state.shape)
        for x in max_obs:
            x[-NUM_RESOURCE_CLUSTER*2-8:-1] = MAX_TIME
            x[-NUM_RESOURCE_CLUSTER-4:-1] = MAX_ENERGY

        self._observation_space = Box(shape=self.state.shape, low=np.zeros(self.state.shape), high=max_obs)
        self._action_space = Discrete(NUM_RESOURCE_CLUSTER)
        return self.state, self.info

    def step(self, action):
        assert not self._terminated, "One episodic has terminated"
        self._state, self._reward = self._get_next_obs(action)
        self._num_steps += 1
        self._num_offloading_steps += 1
        if self._num_offloading_steps >= len(self.hypergraph.tasks):
            # print(self.state, self.reward, self.terminated, info)
            # print(f"一轮已经结束, 已经做出的决策数量：{self._num_steps}, 目前的卸载的总时间:{self._global_clock}, "
            #       f"目前的节点{self.current_task.id}, 总时延{self.end_time}, 总能耗{sum(self._energy.values())}")
            self._terminated = True
        """gymnasium相较于gym需要多返回一个truncated表示是否人为截断，这里始终为False"""
        return self.state, self.reward, self.terminated, False, self.info

    def update_time_and_energy_weight(self, time_weight, energy_weight):
        self._time_weight = time_weight
        self._energy_weight = energy_weight



    def _offloading_reset(self):
        self._num_offloading += 1
        self._num_offloading_steps = 0
        # self._history_resource_available_time = [{task.id: -1 for task in self.hypergraph.tasks} for _ in range(self.resources.num_resources)]
        self._resource_available_time = [0 for _ in range(self.resources.num_resources)]
        self._start_time = {task.id: 0 for task in self.hypergraph.tasks}
        self._exec_time = {task.id: 0 for task in self.hypergraph.tasks}
        self._exec_energy = {task.id: 0 for task in self.hypergraph.tasks}
        self._end_time = {task.id: 0 for task in self.hypergraph.tasks}
        self._energy = {task.id: 0 for task in self.hypergraph.tasks}
        self._current_task, self._terminated = None, False
        self._hypergraph.heap_reset()
        self._step_time = [0] * self.state.shape[0]
        self._step_energy = [0] * self.state.shape[0]
        self._act_distribution = [0] * NUM_RESOURCE_CLUSTER
        self._acts = {task.id: 0 for task in self.hypergraph.tasks}


    def _offloading_computation(self, task, action):
        """
        根据动作计算卸载产生的时延，能耗
        :param actions: 动作
        """
        hypergraph, resources = self.hypergraph, self.resources
        # 获取时延，能耗
        exec_time = resources.get_time(task, action)
        exec_energy = resources.get_energy(task, action)
        self._exec_time[task.id] = exec_time
        self._exec_energy[task.id] = exec_energy

        self._end_time[task.id] = max(self.start_time, self._resource_available_time[action]) + exec_time + \
                                  (self.upload_time + self.download_time if action != 0 else 0)
        self._energy[task.id] = max(self._energy.values()) + exec_energy + (self.upload_energy + self.download_energy if action != 0 else 0)

        self._resource_available_time[action] = max(self._resource_available_time[action], self.end_time)
        # self._history_resource_available_time[action][task.id] = self.end_time
        for next_task_id in hypergraph.next(task.id):
            self._start_time[next_task_id] = max(self._start_time[next_task_id], self.end_time)
            hypergraph.update_node(next_task_id, self.end_time)

    def _get_next_obs(self, action=None):
        """
        根据当前的状态采取动作得到下一个状态，即时奖励
        :param action: 动作
        :return: 下一个状态，即时奖励
        """
        hypergraph, resources = self.hypergraph, self.resources

        if action is None or self._terminated:
            """无动作表示环境的初始化，停止了一轮游戏说明已经完成了所有任务的卸载"""
            self._state = self.init_state
            self._offloading_reset()
            self._current_task, self._terminated = hypergraph.next_node(self.current_task)
            self._acts[self.current_task.id] = 0
            return self._state, 0

        task_topsort_id = self.current_task.topsort_id
        if task_topsort_id + 1 == self.state.shape[0]:
            """需要再判断是否超过了state的固定节点数量，超出需要抛弃"""
            self._terminated = True

        self._offloading_computation(self.current_task, action)

        # 计算除了task的最晚完成时间
        # max_end_time_exc = max(
        #     [0 if task.id == self.current_task.id else self._end_time[task.id] for task in self.hypergraph.tasks])
        max_end_time_exc = max([self._end_time[task_id] for task_id in self.hypergraph.cirti_prev(self.current_task.id)]) if \
            len(self.hypergraph.cirti_prev(self.current_task.id)) > 0 else 0

        max_energy_exc = max(
            [0 if task.id == self.current_task.id else self._energy[task.id] for task in self.hypergraph.tasks])
        # max_energy_exc = max([self._energy[task_id] for task_id in self.hypergraph.prev(self.current_task.id)]) if \
        #     len(self.hypergraph.prev(self.current_task.id)) > 0 else 0

        # print(f"任务{self.current_task.id}的完成时间", self.end_time, max_end_time_exc,
        #       self._max_time[self.current_task.id],
        #       self.energy / self._max_energy[self.current_task.id])

        # 由于我们希望卸载时间和能耗越小越好，所以reward与时间和能耗成反比，即时奖励定义为两次状态的增益
        # add_time = min_max_normalization(max(self.end_time - max_end_time_exc, 0), self.max_total_time,
        #                                  self.min_total_time)
        add_time = self.end_time - max_end_time_exc
        add_energy = self.energy - max_energy_exc
        _reward = self._time_weight * TIME_BENCHMARK_DIMENSION_WEIGHT * add_time + self._energy_weight * ENERGY_BENCHMARK_DIMENSION_WEIGHT * add_energy
        if not self.hypergraph.is_critical_path(self.current_task.id):
            _reward *= CRITICAL_PATH_WEIGHT
        self._step_time[task_topsort_id] = add_time
        self._step_energy[task_topsort_id] = add_energy
        self._global_clock = max(self.end_time, self._global_clock)
        self._global_energy = max(self.energy, self._global_energy)
        self._act_distribution[action] += 1
        self._acts[self.current_task.id] = action

        _state = self._state
        for task in self.hypergraph.tasks:
           _state[task.topsort_id, -NUM_RESOURCE_CLUSTER-7:-NUM_RESOURCE_CLUSTER-4] = torch.tensor([self._exec_time[task.id], self._start_time[task.id], self._end_time[task.id]],  dtype=torch.float64)
           _state[task.topsort_id, -3:] = torch.tensor([self._exec_energy[task.id], self._energy[task.id], task.id == self._current_task.id],  dtype=torch.float64)

        if self.use_graph_state:
            incidence_matrix = self.hypergraph.incidence_matrix
            m = min(incidence_matrix.shape[1], NUM_HG * NUM_HG_EDGES)
            _state[task_topsort_id, :m] = torch.tensor(self.hypergraph.incidence_matrix[task_topsort_id, :m], dtype=torch.float64)

        self._current_task, self._terminated = hypergraph.next_node(self.current_task)
        self._acts[self.current_task.id] = 0
        # fixed_idx = NUM_HG * NUM_HG_EDGES + action * 3 if self.use_graph_state else action * 3
        # _state[task_topsort_id, fixed_idx] = 1
        # _state[task_topsort_id, -2:] = torch.tensor((add_time, add_energy), dtype=torch.float64)
        # print(add_time, add_energy, _reward)
        return _state, _reward

    def render(self, mode=""):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)


def make_env(task_name, training_paths=None, test_paths=None, gnn_type=None, use_graph_state=False,
             device="cpu", use_cache=False, time_weight=REWARD_WEIGHT["time"], energy_weight=REWARD_WEIGHT["energy"]):
    """创建训练集，测试集的环境，为节省内存，资源池采取单例模式，资源池只读
    :return: 单个环境样例（用于获取动作空间，状态空间），训练集环境，测试集环境
    """

    def _select_env(hypergraph: Hypergraph, time_weight=REWARD_WEIGHT["time"], energy_weight=REWARD_WEIGHT["energy"]):
        env = OffloadingEnvironment(resources, hypergraph, gnn_type, device, use_graph_state, time_weight=time_weight,
                                    energy_weight=energy_weight) if task_name == 'HG' else gym.make(task_name)
        if task_name == "HG":
            env.seed(SEED)
        return env

    root = path.dirname(path.dirname(path.abspath(__file__)))

    example_path = os.path.join(root, EXAMPLE_PATH)
    example_hypergraph = Hypergraph(example_path)
    env = _select_env(example_hypergraph)

    train_envs, test_envs = None, None
    train_cache_path = "./cache/train.pkl"
    test_cache_path = "./cache/test.pkl"
    if use_cache:
        if os.path.exists(train_cache_path):
            with open(train_cache_path, "rb") as file:
                # 反序列化
                serialized_file = pickle.load(file)
                train_envs = dill.loads(serialized_file)
        if os.path.exists(test_cache_path):
            with open(test_cache_path, "rb") as file:
                serialized_file = pickle.load(file)
                test_envs = dill.loads(serialized_file)

    if training_paths and train_envs is None:
        train_hgs = HypergraphData(training_paths)
        """需要注意，这里必须使用默认参数_hypergraph捕获临时变量，否则创建的多个环境都将会指向最后一个引用"""
        train_envs = DummyVectorEnv(
            [lambda _hypergraph=hypergraph: _select_env(hypergraph=_hypergraph, time_weight=time_weight, energy_weight=energy_weight) for hypergraph in train_hgs])
        with open(train_cache_path, "wb") as file:
            # 序列化
            serialized_file = dill.dumps(train_envs)
            pickle.dump(serialized_file, file)

    if test_paths and test_envs is None:
        # print(test_paths)
        test_hgs = HypergraphData(test_paths)
        test_envs = DummyVectorEnv(
            [lambda _hypergraph=hypergraph: _select_env(hypergraph=_hypergraph) for hypergraph in test_hgs.hypergraphs])
        with open(test_cache_path, "wb") as file:
            serialized_file = dill.dumps(train_envs)
            pickle.dump(serialized_file, file)

    return env, train_envs, test_envs
