#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: out-16 of 2019
"""

from .utils import get_host_name


class DataPaths(object):
    """
    Class responsible to handle all paths used during the training process.
    """

    @classmethod
    def get_mit_base_data_path(cls):
        """
        Get the base data path from the MIT database.
        :return str: the path according to host.
        """
        if get_host_name() == 'g7':
            return '/media/data/Profissional/Doc/2019-QRS/data_save/mit/'
        elif get_host_name() in ['russell', 'horatio']:
            return '/media/share/pedro/data/QRS/mit/'
        else:
            raise Exception('Host name unknown. Define the data path for the current host ({})'.format(get_host_name()))

    @classmethod
    def get_cybhi_base_data_path(cls):
        """
        Get the base data path from the CYBHi database.
        :return str: the path according to host.
        """
        if get_host_name() == 'g7':
            return '/media/data/Profissional/Doc/2019-QRS/data_save/cybhi/'
        elif get_host_name() in ['russell', 'horatio']:
            return '/media/share/pedro/data/QRS/cybhi/'
        else:
            raise Exception('Host name unknown. Define the data path for the current host ({})'.format(get_host_name()))
