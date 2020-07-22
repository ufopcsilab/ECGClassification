#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: out-15 of 2019
"""

from .base_database_details import BaseDatabaseDetails
from ..paths import DataPaths


class MITDetails(BaseDatabaseDetails):

    @classmethod
    def get_base_data_path(cls):
        return DataPaths.get_mit_base_data_path()

    @classmethod
    def get_test_records(cls):
        return ['100.mat', '102.mat', '104.mat', '106.mat', '108.mat', '112.mat', '114.mat', '116.mat', '118.mat', '122.mat', '124.mat']

    @classmethod
    def get_train_records(cls):
        return ['101.mat', '103.mat', '105.mat', '107.mat', '109.mat', '111.mat', '113.mat', '115.mat', '117.mat', '119.mat', '121.mat', '123.mat']

    @classmethod
    def get_data_frequency(cls):
        return 360
