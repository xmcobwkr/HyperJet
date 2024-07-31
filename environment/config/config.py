from importlib.resources import Resource

from math import log

import numpy as np
import torch

SEED = 3407
CUDA_VISIBLE_DEVICES = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# 一轮游戏执行十次卸载
# NUM_OFFLOADING_COUNT = 10

LOG = True

# example的位置
EXAMPLE_PATH = "data/hypergraph_example.json"

# 数据集位置
DATASET_PATH = "./data/unclassified_data"

# 050501|-050449|050421-|050404-|-050345-|Jun13-142617

NUM_HG_TASKS = 10
NUM_HG_EDGES = 0
# 假设每个场景下有2个用户提交算力需求，即十个NUM_HG需要合并在一块
NUM_HG = 2
# 最大的数据量 50mb bits
MAX_DATA_SIZE = 50 * 1024 * 1024 * 8

# USE_GRAPH_STATE = GNN_TYPE not in ["HGNN+", "GCN", "DHGNN+", "HGNN"]
# 是否使用图状态
USE_GRAPH_STATE = False
USE_HG_EDGE = [True, True, True, True]
HG_EDGE_NUM = [NUM_HG_TASKS + 10, NUM_HG_TASKS, NUM_HG_TASKS // 4, NUM_HG_TASKS // 4]
for k, v in zip(USE_HG_EDGE, HG_EDGE_NUM):
    NUM_HG_EDGES += v if k else 0
# 图状态的类型
GRAPH_STATE_TYPE = "HG"
# 属于正式实验
EXP = None

# 训练相关
EMBED_SIZE = 64
BATCH_SIZE = 16
EPOCH = 300
NUM_TRAIN_HG = 128
NUM_TEST_HG = 128
# 没一次训练会所需要的step，训练会不停的执行train step直到总的step超过这个数，设定为1其实就相当于每次只进行一次train_step
STEP_PER_EPOCH=NUM_HG * NUM_HG_TASKS * NUM_TRAIN_HG
# STEP_PER_EPOCH=1
# 每次的train_step会使用训练的collect，这个是训练的collect会使用的环境数量
EPISODE_PER_COLLECT = BATCH_SIZE
REPEAT_PER_COLLECT = 1
# 每一次的train step会拿test中的前episode_per_test个环境来测试
EPISODE_PER_TEST = BATCH_SIZE
# 早停的耐心设置
PATIENCE = 200

# must in ["GCN", "HGNN", "HGNN+", None]
GNN_TYPE = None

# BASE_POLICY = "/data/huangkang/hypergraph_offloading/log/default/HG/ppo/HGNN/Jun21-082625-Seq2Seq-[0 1 2 3]-task40-edge1200-km1_rKaHyPar_sea20.ini-0.2-0.4"

BASE_POLICY = None

# must in ["Seq2Seq", "MLP", "Transformer"]
NORMAL_NET = ["Seq2Seq"]

PARTITIONING_CONFIG = "km1_rKaHyPar_sea20.ini"

SELF_ATTENTION = False

# 是否使用上次的环境参数，如果使用上次的环境参数，那么就不会重新生成环境，后续关于环境的设定将无效
USE_CACHE = False

USE_HEFT = False

# 实验后缀
NORMAL_NET_STR = '-'.join(NORMAL_NET)
EXPERIMENT_SUFFIX = f"-{NORMAL_NET_STR}-{np.where(np.array(USE_HG_EDGE))[0]}-task{NUM_HG * NUM_HG_TASKS}-edge{NUM_HG_EDGES * NUM_HG}-{PARTITIONING_CONFIG}"

# 策略相关
GAMMA = 0.95
MAX_GRAD_NORM = 0.4
GAE_LAMBDA = 0.95
REW_NORM = 1
DUAL_CLIP = None
VALUE_CLIP = 1
NORM_ADV = 1
VF_COEF = 0.25
ENT_COEF = 0
EPS_CLIP = 0.2
# 评估时是否使用确定性动作而不是随机化的动作，确定性就是选择概率最高的
DETERMINISTIC_EVAL = True
# 回放buffer的大小，存储多少帧
BUFFER_SIZE = 20000

# 任务的执行复杂度, 定位7个等级
TASK_COMPLEXITIES = {"O(1)": lambda n: 1, "O(n)": lambda n: n,
                     "O(nlogn)": lambda n: n * log(n), "O(nlogn^2)": lambda n: n * log(n)**2,
                     "O(n^2)": lambda n: n**2, "O(n^2logn)": lambda n: n**2 * log(n),
                     "O(n^3)": lambda n: n**3, "O(2^n)": lambda n: 2**n}
TASK_COMPLEXITIES_RATE = [0, 0.3, 0.7, 0, 0, 0, 0, 0]
# 常数
TASK_CONSTANTS = list(range(1, 11))
TASK_CONSTANTS_RATE = [0.1] * 10
# 需要执行的基本操作的计算方法： NUM_OPERATION = TASK_CONSTANT * TASK_COMPLEXITY(INPUT_DATA_N)

NUM_RESOURCE_CLUSTER = 4
NUM_EDGE_CLUSTER = NUM_HG // 2
# CPU_TYPES = list(zip(range(NUM_RESOURCE_CLUSTER), [0] * NUM_RESOURCE_CLUSTER))

# 上传带宽，分为多个等级，有10Mbps, 20Mbps, 500Mbps, 1Gbps四个等级
UPLOAD_BANDWIDTHS = [10 * 1024 * 1024, 20 * 1024 * 1024, 500 * 1024 * 1024, 1024 * 1024 * 1024]
# 上传时间计算方法： UPLOAD_TIME = INPUT_DATA_SIZE / UPLOAD_BANDWIDTH

# 下载带宽，一般下载带宽会比上传带宽快
DOWNLOAD_BANDWIDTHS = [100 * 1024 * 1024, 50 * 1024 * 1024, 500 * 1024 * 1024, 1024 * 1024 * 1024]
# 下载时间计算方法： DOWNLOAD_TIME = DOWNLOAD_DATA_SIZE /  DOWNLOAD_BANDWIDTH

LOAD_ENERGY = [10, 20, 30, 40]

# 选中对应带宽的概率
BANDWIDTHS_RATE = [0.35, 0.639, 0.01, 0.001]

# 一般来说，单个CPU的频率可以1s执行10^8*32的bit操作
BASE_INT = 32

# 只考虑单核的CPU情况，CPU频率，HZ为单位
CPU_FREQUENCIES = [2.0 * 10**9, 3 * 10**9, 3.6 * 10**9, 3.7 * 10**9, 4.5 * 10**9, 4.9 * 10**9]

LOCAL_CPU_FREQUENCY = 1.0 * 10 ** 9

LOCAL_ENERGY_COEFFICIENT = 20

ENERGY_COEFFICIENT = [20, 30, 40, 40, 40, 50]

# CPU执行时间的计算方法：EXECUTION_TIME = NUM_OPERATION / CPU_FREQUENCY
# CPU执行能耗的计算方法： EXECUTION_ENERGY = NUM_OPERATION * (CPU_FREQUENCY) ^ 2 * ENERGY_COEFFICIENT

# 状态空间为连续，需要限定最大时间，最大能耗
MS2S = 1000

MAX_INPUT_N = MAX_DATA_SIZE / BASE_INT
MAX_EXEC_TIME = MAX_INPUT_N * MAX_INPUT_N * BASE_INT / min(CPU_FREQUENCIES) / MS2S
MAX_TIME = NUM_HG * NUM_HG_TASKS * (MAX_DATA_SIZE / min(UPLOAD_BANDWIDTHS) + MAX_DATA_SIZE / min(DOWNLOAD_BANDWIDTHS) + MAX_EXEC_TIME) / MS2S
MAX_ENERGY = NUM_HG * NUM_HG_TASKS * MAX_EXEC_TIME / MS2S * max(ENERGY_COEFFICIENT)

# reward中时延和能耗的量纲对标权重
TIME_BENCHMARK_DIMENSION_WEIGHT = - 10

ENERGY_BENCHMARK_DIMENSION_WEIGHT = - 1

REWARD_WEIGHT = {
    "time": .2,
    "energy": .4
}
# reward中非关键路径的权重
CRITICAL_PATH_WEIGHT = .5



