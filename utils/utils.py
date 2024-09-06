from scipy.interpolate import interp1d
import sys
import os

from environment.config.config import *


def get_gpu():
    return f'cuda:{CUDA_VISIBLE_DEVICES}' if torch.cuda.is_available() else 'cpu'


class CustomException(Exception):
    def __init__(self, message="Custom exception occurred"):
        self.message = message
        super().__init__(self.message)


def get_transmission_time(task):
    return int(task.input_data_size) / task.bandwidth_up / MS2S, int(task.output_data_size) / task.bandwidth_dl / MS2S


def get_transmission_info(task):
    upload_time, download_time = get_transmission_time(task)
    upload_energy, download_energy = upload_time * task.load_energy_coefficient, download_time * task.load_energy_coefficient
    return upload_time, download_time, upload_energy, download_energy

def get_file_paths(max_training_paths=NUM_HG * NUM_TRAIN_HG, max_test_paths=NUM_HG * NUM_TEST_HG, min_task_num=NUM_HG_TASKS-5,
                   max_task_num=NUM_HG_TASKS):
    training_paths = []
    test_paths = []
    import re
    paths = []
    for filename in os.listdir(DATASET_PATH):
        match = re.search(r'_(\d+)_(\d+)', filename)
        if match:
            node_num = int(match.group(1))
            edge_num = int(match.group(2))
            if filename.endswith('.json') and min_task_num < node_num <= max_task_num and min_task_num <= edge_num <= max_task_num + 10:
                file_path = os.path.join(DATASET_PATH, filename)
                paths.append((node_num, edge_num, file_path))
                if len(paths) >= max_training_paths + max_test_paths:
                    break

    paths = sorted(paths, key=lambda x: (-x[0], -x[1]))

    for i, (node_num, edge_num, file_path) in enumerate(paths):
        if i < max_training_paths:
            training_paths.append(file_path)
        else:
            test_paths.append(file_path)
    return training_paths, test_paths


def min_max_normalization(x, max_value, min_value):
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

    assert len(task_sequence) == len(tasks), "Appearance of cycles"

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

class DisablePrint:
    def __enter__(self):
        pass
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._original_stdout_fd = os.dup(1)
        self._original_stderr_fd = os.dup(2)
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        os.dup2(self._original_stdout_fd, 1)
        os.dup2(self._original_stderr_fd, 2)
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class RedirectOutput:
    def __init__(self, stdout_path, stderr_path):
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.stdout_old = sys.stdout
        self.stderr_old = sys.stderr

    def __enter__(self):
        sys.stdout = open(self.stdout_path, 'w')
        sys.stderr = open(self.stderr_path, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.stdout_old
        sys.stderr = self.stderr_old



def test_exp():
    def normalize_list(input_list):
        max_val = max(input_list)
        min_val = min(input_list)
        normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]

        return normalized_list

    original_list = [2, 5, 8, 12, 4]
    normalized_list = normalize_list(original_list)

    print("Original List:", original_list)
    print("Normalized List:", normalized_list)


def interpolate_subarrays(arr, target_length):
    """
    Interpolate each subarray of the two-dimensional array so that their lengths become target_1ength.

    Parameters:
    arr (list of list of float): Input two-dimensional array
    target_length (int): target length

    Returns:
    np.ndarray: Interpolated two-dimensional array
    """
    interpolated_arr = []

    for subarray in arr:
        current_length = len(subarray)
        if current_length == target_length:
            interpolated_arr.append(subarray)
            continue
        current_positions = np.linspace(0, 1, current_length)
        target_positions = np.linspace(0, 1, target_length)
        interpolator = interp1d(current_positions, subarray, kind='linear')
        interpolated_subarray = interpolator(target_positions)

        interpolated_arr.append(interpolated_subarray)

    return np.array(interpolated_arr)

def get_mean(data):
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
