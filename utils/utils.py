import torch
import numpy as np
from scipy.interpolate import interp1d
import sys
import os

from environment.config.config import MS2S, CUDA_VISIBLE_DEVICES, TIME_BENCHMARK_DIMENSION_WEIGHT, \
    ENERGY_BENCHMARK_DIMENSION_WEIGHT


def get_gpu():
    return f'cuda:{CUDA_VISIBLE_DEVICES}' if torch.cuda.is_available() else 'cpu'


class CustomException(Exception):
    def __init__(self, message="发生了自定义异常"):
        self.message = message
        super().__init__(self.message)


def get_transmission_time(task):
    return int(task.input_data_size) / task.bandwidth_up / MS2S, int(task.output_data_size) / task.bandwidth_dl / MS2S


def get_transmission_info(task):
    upload_time, download_time = get_transmission_time(task)
    upload_energy, download_energy = upload_time * task.load_energy_coefficient, download_time * task.load_energy_coefficient
    return upload_time, download_time, upload_energy, download_energy


def min_max_normalization(x, max_value, min_value):
    # 最大最小归一化
    return (x - min_value) / (max_value - min_value)


def topsort_with_time_and_energy(id2tasks, edges, resources):
    import heapq
    heap, task_sequence = [], []
    heapq.heapify(heap)
    tasks = id2tasks.values()
    deg = {task.id: 0 for task in tasks}
    for adj in edges.values():
        for v in adj:
            deg[v] += 1
    vis = {task.id: False for task in tasks}
    max_rew = {task.id: 0 for task in tasks}
    for task in tasks:
        if deg[task.id] == 0:
            max_rew[task.id] = - TIME_BENCHMARK_DIMENSION_WEIGHT * resources.get_time(task, 0) - \
                               ENERGY_BENCHMARK_DIMENSION_WEIGHT * resources.get_energy(task, 0)
            heapq.heappush(heap, (max_rew[task.id], task.id))

    while len(heap) > 0:
        rew, u = heapq.heappop(heap)
        if vis[u]:
            continue
        vis[u] = True
        task_sequence.append(u)
        for v in edges[u]:
            deg[v] -= 1
            add_rew = - TIME_BENCHMARK_DIMENSION_WEIGHT * resources.get_time(id2tasks[v], 0) - \
                      ENERGY_BENCHMARK_DIMENSION_WEIGHT * resources.get_energy(id2tasks[v], 0)
            max_rew[v] = max(max_rew[v], max_rew[u] + add_rew)
            if deg[v] == 0:
                heapq.heappush(heap, (max_rew[v], v))

    assert len(task_sequence) == len(tasks), "出现环"

    # 反向求最晚完成时间
    heap = []
    heapq.heapify(heap)
    rev_edges = {task.id: [] for task in tasks}
    for src, adj in edges.items():
        for dst in adj:
            rev_edges[dst].append(src)
    deg = {task.id: 0 for task in tasks}
    for adj in rev_edges.values():
        for v in adj:
            deg[v] += 1
    mx_rew = max(max_rew.values())
    min_rew = {task.id: mx_rew for task in tasks}
    for task in tasks:
        if deg[task.id] == 0:
            heapq.heappush(heap, (-mx_rew, task.id))
    while len(heap) > 0:
        rew, u = heapq.heappop(heap)
        add_rew = - TIME_BENCHMARK_DIMENSION_WEIGHT * resources.get_time(id2tasks[u], 0) - \
                  ENERGY_BENCHMARK_DIMENSION_WEIGHT * resources.get_energy(id2tasks[u], 0)
        for v in rev_edges[u]:
            deg[v] -= 1
            min_rew[v] = min(min_rew[v], min_rew[u] - add_rew)
            if deg[v] == 0:
                heapq.heappush(heap, (- min_rew[v], v))
    return max_rew, min_rew, task_sequence

# 定义DisablePrint类
class DisablePrint:
    def __enter__(self):
        pass
        # self._original_stdout = sys.stdout
        # self._original_stderr = sys.stderr
        # self._original_stdout_fd = os.dup(1)
        # self._original_stderr_fd = os.dup(2)
        # sys.stdout = open(os.devnull, 'w')
        # sys.stderr = open(os.devnull, 'w')
        # os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        # os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        # os.dup2(self._original_stdout_fd, 1)
        # os.dup2(self._original_stderr_fd, 2)
        # sys.stdout.close()
        # sys.stderr.close()
        # sys.stdout = self._original_stdout
        # sys.stderr = self._original_stderr


def test_exp():
    # 示例
    # try:
    #     raise CustomException("这是一个自定义异常的消息")
    # except CustomException as ce:
    #     print(f"捕获到自定义异常: {ce}")
    def normalize_list(input_list):
        # 找到列表中的最大值和最小值
        max_val = max(input_list)
        min_val = min(input_list)

        # 对列表中的每个值进行归一化
        normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]

        return normalized_list

    # 示例
    original_list = [2, 5, 8, 12, 4]
    normalized_list = normalize_list(original_list)

    print("Original List:", original_list)
    print("Normalized List:", normalized_list)


def interpolate_subarrays(arr, target_length):
    """
    对二维数组的每个子数组进行插值，使得它们的长度都变为target_length。

    Parameters:
    arr (list of list of float): 输入的二维数组
    target_length (int): 目标长度

    Returns:
    np.ndarray: 插值后的二维数组
    """
    interpolated_arr = []

    for subarray in arr:
        # 获取当前子数组的长度
        current_length = len(subarray)

        # 如果当前子数组的长度已经是目标长度，则无需插值
        if current_length == target_length:
            interpolated_arr.append(subarray)
            continue

        # 创建一个表示当前子数组位置的数组
        current_positions = np.linspace(0, 1, current_length)
        target_positions = np.linspace(0, 1, target_length)

        # 使用线性插值方法
        interpolator = interp1d(current_positions, subarray, kind='linear')
        interpolated_subarray = interpolator(target_positions)

        interpolated_arr.append(interpolated_subarray)

    return np.array(interpolated_arr)

def get_mean(data):
    # 二维数组求平均值
    mx_len = max([len(v) for v in data])
    data = interpolate_subarrays(data, mx_len)
    mean_data = [0] * len(data[0])
    for i in range(len(data)):
        for j in range(len(mean_data)):
            mean_data[j] += data[i][j]
    for i in range(len(mean_data)):
        mean_data[i] /= len(data)
    return mean_data


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
