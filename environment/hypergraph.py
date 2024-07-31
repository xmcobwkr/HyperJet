import heapq
import json
import os
import random
from collections import deque, namedtuple
from types import SimpleNamespace
from tqdm import tqdm
import kahypar
import heapq

import numpy as np

from environment.resource import Resources, resources
from utils.utils import CustomException, topsort_with_time_and_energy, DisablePrint
from .config.config import *


class HypergraphData():
    """
        超图数据集，包含了所有的超图数据，读取超图文件并将NUM_HG超图任务合并为一个超图样本作为输入数据的一部分
    """

    def __init__(self, paths):
        if isinstance(paths, str):
            paths = [paths]

        if not isinstance(paths, list):
            raise CustomException("超图类的构造参数必须是文件路径或者是文件路径的列表")

        for path in paths:
            if not os.path.exists(path):
                raise CustomException(f"超图文件路径{path}不存在")

        hypergraphs = []
        for i in tqdm(range(0, len(paths), NUM_HG), desc="读取超图中...", unit="iter"):
            hypergraph_list = []
            for j in range(i, min(i + NUM_HG, len(paths))):
                hypergraph_list.append(Hypergraph(path=paths[j]))
            # 将NUM_HG个超图合并在一起作为一个样本
            union_hypergraph = Hypergraph(hypergraphs=hypergraph_list)
            # 生成超边， 拓展超图
            union_hypergraph.generate_hyperedges(partitioning_k=NUM_RESOURCE_CLUSTER)
            hypergraphs.append(union_hypergraph)
        self._hypergraphs = hypergraphs

        self.index = 0

    @property
    def hypergraphs(self):
        return self._hypergraphs

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self):
            result = self.hypergraphs[self.index]
            self.index += 1
            return result
        else:
            self.index = 0
            raise StopIteration

    def __len__(self):
        return len(self.hypergraphs)


class Hypergraph():
    def __init__(self, path=None, hypergraphs=None):
        self._degs = None
        self._vis = None
        self._heap = []
        if path is None and hypergraphs is None:
            raise CustomException(f"错误的超图初始化参数")
        if path is not None:
            if not os.path.exists(path):
                raise CustomException(f"超图文件路径{path}不存在")
            self._parse_hypergraph_from_path(path)
        else:
            if isinstance(hypergraphs, Hypergraph):
                hypergraphs = [hypergraphs]

            if not isinstance(hypergraphs, list):
                raise CustomException(f"错误的超图初始化参数")

            for hypergraph in hypergraphs:
                if not isinstance(hypergraph, Hypergraph):
                    raise CustomException(f"错误的超图初始化参数")

            self._parse_hypergraph_from_hypergraph(hypergraphs)

        self._parse_sequence_from_hypergraph()

    @property
    def tasks(self):
        return list(self._id2task.values())

    @property
    def edges(self):
        return list(self._id2edge.values())

    @property
    def task_sequence(self):
        return self._task_sequence

    @property
    def incidence_matrix(self):
        return self._incidence_matrix

    @property
    def source_incidence_matrix(self):
        return self._source_incidence_matrix

    @property
    def target_incidence_matrix(self):
        return self._target_incidence_matrix

    @property
    def undirected_edge_list(self):
        return self._undirected_edge_list

    @property
    def directed_edge_list(self):
        return self._source_edge_list, self._target_edge_list

    def id2topsort_id(self, task_id):
        return self._id2task[task_id].topsort_id

    def is_critical_path(self, task_id):
        return self._is_critical_path[task_id]

    def next(self, task_id):
        return self._next_nodes[task_id]

    def prev(self, task_id):
        return self._prev_nodes[task_id]

    def cirti_prev(self, task_id):
        """关键路径上的上一个节点，由元组列表组成，当这个节点在关键路径上时第二个元素为1，否则为0"""
        return self._pre[task_id]



    def heap_reset(self):
        self._heap = []
        self._degs = {task.id: task.deg for task in self.tasks}
        self._start_time = {task.id: 0 for task in self.tasks}
        heapq.heapify(self._heap)
        for task in self.tasks:
            if task.deg == 0:
                """入度为0的点可以随时启动"""
                heapq.heappush(self._heap, [0, task.topsort_id])
    def update_node(self, task_id, start_time):
        self._start_time[task_id] = max(self._start_time[task_id], start_time)
        self._degs[task_id] -= 1
        if self._degs[task_id] == 0:
            heapq.heappush(self._heap, [self._start_time[task_id], self.id2topsort_id(task_id)])


    def next_node(self, task=None):
        """
        查询下一个任务
        :param task: 当前的任务
        :return: 下一个任务以及是否已经完成一次全部任务的卸载
        """
        if task is None:
            if len(self._heap) == 0:
                self.heap_reset()
            assert len(self._heap), "堆未初始化"
            task_topsort_id = heapq.heappop(self._heap)[1]
            return self.task_sequence[task_topsort_id], False
        if len(self._heap) == 0:
            return self.task_sequence[0], True
        task_topsort_id = heapq.heappop(self._heap)[1]
        return self.task_sequence[task_topsort_id], False

    def _get_next_nodes(self):
        """
        获取所有节点的后继节点
        :return:
        """
        _next_nodes = {task.id: [] for task in self.tasks}
        _prev_nodes = {task.id: [] for task in self.tasks}
        for edge in self.edges:
            if edge.type == 1:
                _next_nodes[edge.source].append(edge.target)
                _prev_nodes[edge.target].append(edge.source)
        self._next_nodes = _next_nodes
        self._prev_nodes = _prev_nodes

    def _parse_hypergraph_from_path(self, path: str):
        """
        :param path: 超图的json文件
        :return:
        """
        with open(path, 'r') as f:
            data_dict = json.load(f)
            data = SimpleNamespace(**data_dict)
            _id2task = {}
            _id2edge = {}

            for _node in data.nodes:
                node = SimpleNamespace(**_node)
                # 暂时固定为1
                # node.task_complexity = list(TASK_COMPLEXITIES.values())[int(node.task_complexity)]
                node.task_complexity = list(TASK_COMPLEXITIES.values())[int(node.task_complexity)]
                node.task_constant = TASK_CONSTANTS[node.task_constant]
                up_bandwidth = getattr(data, 'up_bandwidth', None)
                down_bandwidth = getattr(data, 'down_bandwidth', None)
                if up_bandwidth and down_bandwidth:
                    node.bandwidth_up = up_bandwidth
                    node.bandwidth_dl = down_bandwidth
                else:
                    node.bandwidth_up = UPLOAD_BANDWIDTHS[data.bandwidth_type]
                    node.bandwidth_dl = DOWNLOAD_BANDWIDTHS[data.bandwidth_type]
                node.load_energy_coefficient = LOAD_ENERGY[data.bandwidth_type]
                node.num_operation = node.task_complexity(node.input_data_size // BASE_INT) * node.task_constant
                node.deg = 0
                if node.init_resource_id == -1:
                    node.init_resource_id = random.choice(range(NUM_RESOURCE_CLUSTER))
                node.id = int(node.id)
                _id2task[node.id] = node
            edge_id = 1
            for _edge in data.edges:
                edge = SimpleNamespace(**_edge)
                edge.id = edge_id
                _id2edge[edge_id] = edge
                if edge.type == 1:
                    _id2task[edge.target].deg += 1
                edge_id += 1
            self._id2task = _id2task
            self._id2edge = _id2edge
            self._sorted_task = sorted(self.tasks, key=lambda x: x.num_operation)
            for i, task in enumerate(self._sorted_task):
                task.sorted_id = i

    def _parse_hypergraph_from_hypergraph(self, hypergraphs):
        """
        合并超图，直接基于编号合并
        :param hypergraphs: 超图的列表
        :return:
        """

        _id2task = {}
        _id2edge = {}
        for id, hypergraph in enumerate(hypergraphs):
            for task in hypergraph.tasks:
                task.id = f'hg{id + 1}_{task.id}'
                _id2task[task.id] = task

            for edge in hypergraph.edges:
                for i, node in enumerate(edge.nodes):
                    edge.nodes[i] = f'hg{id + 1}_{node}'
                edge.source = f'hg{id + 1}_{edge.source}'
                edge.target = f'hg{id + 1}_{edge.target}'
                edge.id = f'hg{id + 1}_{edge.id}'
                _id2edge[edge.id] = edge

        self._id2task = _id2task
        self._id2edge = _id2edge
        self._sorted_task = sorted(self.tasks, key=lambda x: x.num_operation)
        for i, task in enumerate(self._sorted_task):
            task.sorted_id = i


    def _parse_sequence_from_hypergraph(self):
        """
        通过拓扑排序将超图任务解析为序列
        :return:
        """
        self._get_next_nodes()
        # 拓扑排序
        max_rew, min_rew, _task_sequence = topsort_with_time_and_energy(self._id2task, self._next_nodes, resources)
        self._pre = {task.id: [] for task in self.tasks}
        self._is_critical_path = {task.id: max_rew[task.id] == min_rew[task.id] for task in self.tasks}
        for src, adj in self._next_nodes.items():
            for dst in adj:
                if self.is_critical_path(dst) and not self.is_critical_path(src):
                    continue
                self._pre[dst].append(src)
        for i, task_id in enumerate(_task_sequence):
            self._id2task[task_id].topsort_id = i
        self._task_sequence = [self._id2task[task_id] for task_id in _task_sequence]

        # q = deque()
        # _task_sequence = []
        # degs = {task.id: task.deg for task in self._id2task.values()}
        # for task in self._id2task.values():
        #     if task.deg == 0:
        #         q.append(task)
        # cnt = 0
        # while len(q) > 0:
        #     task = q.popleft()
        #     self._id2task[task.id].topsort_id = cnt
        #     _task_sequence.append(task)
        #     for next_task_id in self.next(task.id):
        #         degs[next_task_id] -= 1
        #         if degs[next_task_id] == 0:
        #             q.append(self._id2task[next_task_id])
        #     cnt += 1
        #
        # # 不出现环
        # assert cnt == len(degs)
        # self._task_sequence = _task_sequence

        # 根据edge_list生成HG
        self._undirected_edge_list = [[self.id2topsort_id(node) for node in edge.nodes] for edge in self.edges]
        self._source_edge_list = [[self.id2topsort_id(edge.source)] for edge in self.edges if edge.type == 1]
        self._target_edge_list = [[self.id2topsort_id(edge.target)] for edge in self.edges if edge.type == 1]

        self._update_incidence_matrix()

    def _append_edge(self, edge, in_matrix=True):
        self._id2edge[edge.id] = edge
        if in_matrix:
            self._undirected_edge_list.append([self.id2topsort_id(node) for node in edge.nodes])
            if edge.type == 1:
                self._source_edge_list.append(edge.source)
                self._target_edge_list.append(edge.target)
            else:
                self._source_edge_list.append([self.id2topsort_id(node) for node in edge.nodes])
                self._source_edge_list.append([self.id2topsort_id(node) for node in edge.nodes])

    def _update_incidence_matrix(self):
        _incidence_matrix = np.zeros((len(self.tasks), len(self.edges)))
        _source_incidence_matrix = np.zeros((len(self.tasks), len(self.edges)))
        _target_incidence_matrix = np.zeros((len(self.tasks), len(self.edges)))
        for edge_id, edge in enumerate(self.edges):
            for node in edge.nodes:
                idx = self.id2topsort_id(node)
                _incidence_matrix[idx][edge_id] = 1
                if edge.type == 0:
                    _source_incidence_matrix[idx][edge_id] = _target_incidence_matrix[idx][edge_id] = 1
            if edge.type == 1:
                _source_incidence_matrix[self.id2topsort_id(edge.source)][edge_id] =\
                    _target_incidence_matrix[self.id2topsort_id(edge.target)][edge_id] = 1

        self._incidence_matrix = _incidence_matrix
        self._source_incidence_matrix = _source_incidence_matrix
        self._target_incidence_matrix = _target_incidence_matrix

    def get_k_neighborhood(self, task_id, k):
        res = {task_id}
        if k == 0:
            return res
        for next_task_id in self.prev(task_id):
            res.union(self.get_k_neighborhood(next_task_id, k - 1))
        return res

    def generate_hyperedges(self, neighborhood_k=2, kmeans_k=NUM_EDGE_CLUSTER, partitioning_k=NUM_EDGE_CLUSTER, partitioning_config=PARTITIONING_CONFIG):
        """生成无向超边
        1. 通过k邻域元素建立超边
        2. 通过相近的num_operation建立超边
        3. 通过超图分区建立超边
        """
        edges_1, edges_2, edges_3 = [], [], []

        if USE_HG_EDGE[1] or USE_HG_EDGE[-1]:
            edge_idx = 1
            for task in self.tasks:
                new_edge = {"nodes": list(self.get_k_neighborhood(task.id, neighborhood_k)), "type": 0, "e_weight": 0,
                            "id": f"{neighborhood_k}-neighborhood-{edge_idx}"}
                edge_idx += 1
                # self._append_edge(SimpleNamespace(**new_edge), USE_HG_EDGE[1])
                edges_1.append(SimpleNamespace(**new_edge))

        if USE_HG_EDGE[2] or USE_HG_EDGE[-1]:
            from sklearn.cluster import KMeans
            X = np.array([[task.bandwidth_up, task.bandwidth_dl, task.load_energy_coefficient, task.num_operation] for task in self.tasks])
            # 创建KMeans对象并进行聚类
            kmeans = KMeans(n_clusters=kmeans_k)
            kmeans.fit(X)
            labels = {label_id: [] for label_id in range(kmeans_k)}
            kmeans_labels = kmeans.labels_
            for i, task in enumerate(self.tasks):
                labels[kmeans_labels[i]].append(task.id)
            edge_idx = 1
            for tasks in labels.values():
                if len(tasks) > 0:
                    new_edge = {"nodes": tasks, "type": 0, "e_weight": 0,
                                "id": f"{kmeans_k}-kmeans-{edge_idx}"}
                    edge_idx += 1
                    # self._append_edge(SimpleNamespace(**new_edge), USE_HG_EDGE[2])
                    edges_2.append(SimpleNamespace(**new_edge))

        if USE_HG_EDGE[-1]:
            with DisablePrint():
                context = kahypar.Context()
                context.loadINIconfiguration(f"./environment/config/kahypar-config/{partitioning_config}")
                hyperedge_indices = [0]
                hyperedges = []
                # for i, edge in enumerate(self.edges):
                #     for node in edge.nodes:
                #         hyperedges.append(self.id2topsort_id(node))
                #     hyperedge_indices.append(len(hyperedges))
                for edge in edges_1:
                    tmp = []
                    for node in edge.nodes:
                        tmp.append(self.id2topsort_id(node))
                    tmp = sorted(tmp)
                    hyperedges.extend(tmp)
                    hyperedge_indices.append(len(hyperedges))
                for edge in edges_2:
                    tmp = []
                    for node in edge.nodes:
                        tmp.append(self.id2topsort_id(node))
                    tmp = sorted(tmp)
                    hyperedges.extend(tmp)
                    hyperedge_indices.append(len(hyperedges))

                # ibm01 = kahypar.createHypergraphFromFile(mydir+"/ISPD98_ibm01.hgr",2)
                hypergraph = kahypar.Hypergraph(len(self.tasks), len(edges_1) + len(edges_2), hyperedge_indices, hyperedges, partitioning_k)

                context.setK(partitioning_k)
                context.setEpsilon(0.03)
                kahypar.partition(hypergraph, context)

                block_dict = {block_id: [] for block_id in range(partitioning_k + 1)}
                for task in self.task_sequence:
                    block_id = hypergraph.blockID(task.topsort_id)
                    block_dict[block_id].append(task.id)
                edge_idx = 1
                for tasks in block_dict.values():
                    if len(tasks) > 0:
                        new_edge = {"nodes": tasks, "type": 0, "e_weight": 0,
                                    "id": f"{partitioning_k}-partitioning-{edge_idx}"}
                        edge_idx += 1
                        # self._append_edge(SimpleNamespace(**new_edge))
                        edges_3.append(SimpleNamespace(**new_edge))

        if USE_HG_EDGE[1] and edges_1:
            for edge in edges_1:
                self._append_edge(edge)

        if USE_HG_EDGE[2] and edges_2:
            for edge in edges_2:
                self._append_edge(edge)

        if USE_HG_EDGE[3] and edges_3:
            for edge in edges_3:
                self._append_edge(edge)

        if GRAPH_STATE_TYPE == "HG":
            self._update_incidence_matrix()

