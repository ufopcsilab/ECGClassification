#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: out-15 of 2019
"""


class BaseDatabaseDetails(object):
    """
    Class with the methods during the training.
    """

    @classmethod
    def get_base_data_path(cls):
        """
        Return the base data path.
        :return: str.
        """
        raise NotImplementedError()

    @classmethod
    def get_test_records(cls):
        """
        Return a list of the records that will be used during the test phase.
        :return list: a list of records that will be used during the test phase.
        """
        raise NotImplementedError()

    @classmethod
    def get_train_records(cls):
        """
        Return a list of the records that will be used during the train phase.
        :return list: a list of records that will be used during the train phase.
        """
        raise NotImplementedError()

    @classmethod
    def get_data_frequency(cls):
        """
        The data frequency of the ECG signal.
        :return int: the frequency in which the data was acquired.
        """
        raise NotImplementedError()
