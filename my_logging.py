import sys
import os
# import time

class Logger(object):
    def __init__(self, _name, stream=sys.stdout):
        self.terminal = stream

        file_dict=os.path.join('{}.log'.format(_name))
        self.log = open(file_dict, 'w')
        print('start logging ... {}'.format(file_dict))

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
        pass

