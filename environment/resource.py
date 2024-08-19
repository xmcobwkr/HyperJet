import random

from .config.config import *
import numpy as np
import os

class Resources(object):

    def __init__(self, cpu_types=None):
        self._reset(cpu_types)

    def _reset(self, cpu_types):
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
        :param task: Task node, including related task attributes
        :param resource_id: Resource pool node
        :return: Time of execution
        """
        resource = self.resources[resource_id]
        return task.num_operation * BASE_INT / resource["cpu_frequency"] / MS2S

    def get_energy(self, task, resource_id):
        """
        :param task: Task node, including related task attributes
        :param resource_id: Resource pool node
        :return: Energy of execution
        """
        resource = self.resources[resource_id]
        # return self.get_time(task, resource_id) * resource["cpu_frequency"] ** 3 * resource["cpu_energy_coefficient"]
        return self.get_time(task, resource_id) * resource["cpu_energy_coefficient"]

    def get_max_time(self, task):
        """
        :param task: Task node, including related task attributes
        :return: (Time, RP id)
        """
        return max([(self.get_time(task, resource_id), resource_id) for resource_id in range(self.num_resources)])

    def get_max_energy(self, task):
        """
        :param task: Task node, including related task attributes
        :return: (Energy, RP id)
        """
        return max([(self.get_energy(task, resource_id), resource_id) for resource_id in range(self.num_resources)])


    def get_min_time(self, task):
        """
        :param task: Task node, including related task attributes
        :return: (Energy, RP id)
        """
        return min([(self.get_time(task, resource_id), resource_id) for resource_id in range(self.num_resources)])

    def get_min_energy(self, task):
        """
        :param task: Task node, including related task attributes
        :return: (Energy, RP id)
        """
        return min([(self.get_energy(task, resource_id), resource_id) for resource_id in range(self.num_resources)])

resources = Resources()