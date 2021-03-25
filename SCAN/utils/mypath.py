"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database='', type=''):
        db_names = {'partnet'}
        assert(database in db_names)

        if database == 'partnet':
            if type == 'chair':
                return '../partit_data/0.chair/'
            elif type == 'table':
                return '../partit_data/1.table/'
            elif type == 'bed':
                return '../partit_data/2.bed/'
            elif type == 'bag':
                return '../partit_data/3.bag/'

