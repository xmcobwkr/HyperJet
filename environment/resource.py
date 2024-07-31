import random

from .config.config import *
import numpy as np
import os

class Resources(object):
    """
    构建算力网络资源池
    """

    def __init__(self, cpu_types=None):
        self._reset(cpu_types)

    def _reset(self, cpu_types):
        """
        资源池的初始化主要为CPU的频率，CPU的时钟周期，
        :return: 全部资源池的能耗系数
        """
        resources = [{
            "cpu_frequency": LOCAL_CPU_FREQUENCY,
            "cpu_energy_coefficient": LOCAL_ENERGY_COEFFICIENT
        }]
        if cpu_types is None:
            cpu_types = [x % len(CPU_FREQUENCIES) for x in range(self.num_resources - 1)]
        for i in range(self.num_resources - 1):
            cpu_frequency = CPU_FREQUENCIES[cpu_types[i]]
            cpu_energy_coefficient = ENERGY_COEFFICIENT[cpu_types[i]]
            resources.append({
                "cpu_frequency": cpu_frequency,
                "cpu_energy_coefficient": cpu_energy_coefficient,
            })
        self._resources = resources


    @property
    def num_resources(self):
        return NUM_RESOURCE_CLUSTER

    @property
    def resources(self):
        return self._resources

    def get_time(self, task, resource_id):
        """
        :param task: 任务节点，包含了相关的任务属性
        :param resource_id: 资源池节点
        :return: 执行的时间
        """
        resource = self.resources[resource_id]
        return task.num_operation * BASE_INT / resource["cpu_frequency"] / MS2S

    def get_energy(self, task, resource_id):
        """
        :param task: 任务节点，包含了相关的任务属性
        :param resource_id: 资源池节点
        :return: 执行的能耗
        """
        resource = self.resources[resource_id]
        # return self.get_time(task, resource_id) * resource["cpu_frequency"] ** 3 * resource["cpu_energy_coefficient"]
        return self.get_time(task, resource_id) * resource["cpu_energy_coefficient"]

    def get_max_time(self, task):
        """
        获取卸载到资源池的最大时间
        :param task: 任务节点，包含了相关的任务属性
        :return: （时间，资源池id）
        """
        return max([(self.get_time(task, resource_id), resource_id) for resource_id in range(self.num_resources)])

    def get_max_energy(self, task):
        """
        获取卸载到资源池的最大能耗
        :param task: 任务节点，包含了相关的任务属性
        :return: （能耗，资源池id）
        """
        return max([(self.get_energy(task, resource_id), resource_id) for resource_id in range(self.num_resources)])


    def get_min_time(self, task):
        """
        获取卸载到资源池的最小时间
        :param task: 任务节点，包含了相关的任务属性
        :return: （时间，资源池id）
        """
        return min([(self.get_time(task, resource_id), resource_id) for resource_id in range(self.num_resources)])

    def get_min_energy(self, task):
        """
        获取卸载到资源池的最小能耗
        :param task: 任务节点，包含了相关的任务属性
        :return: （能耗，资源池id）
        """
        return min([(self.get_energy(task, resource_id), resource_id) for resource_id in range(self.num_resources)])

resources = Resources()