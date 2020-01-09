from time import time
from datetime import datetime


class TimeExpector:
    def __init__(self):
        self._last_time_length = None
        self.tick_tock_check = False
        self.t_start = None

    @staticmethod
    def time_unit_creator(l_time):
        time_unit = 'seconds'
        if l_time >= 60:
            l_time /= 60
            time_unit = 'minutes'
        else:
            return l_time, time_unit

        if l_time >= 60:
            l_time /= 60
            time_unit = 'hours'
        else:
            return l_time, time_unit

        if l_time >= 24:
            l_time /= 24
            time_unit = 'days'
        else:
            return l_time, time_unit

        if l_time >= 30:
            l_time /= 30
            time_unit = 'month'
            return l_time, time_unit
        else:
            return l_time, time_unit

    def tick(self, iteration_left=None, reset=False):
        if self.tick_tock_check:
            print('[WARNING] calling "tick" without a "tock"...')

        if reset:
            self._last_time_length = None
        self.t_start = time()

        if self._last_time_length is not None:
            expectation = self.t_start
            if iteration_left is not None:
                expectation +=  self._last_time_length * iteration_left
            else:
                expectation +=  self._last_time_length
            datestr = datetime.fromtimestamp(expectation).strftime("%Y-%m-%d %H:%M:%S")
            print('[expecting to finish at %s]' % datestr)
        self.tick_tock_check = True

    def tock(self):
        if not self.tick_tock_check:
            print('[WARNING] calling "tock" without a "tick"...')

        self.tick_tock_check = False
        length = time() - self.t_start

        if self._last_time_length is None:
            self._last_time_length = length
        else:
            self._last_time_length = .9 * self._last_time_length + .1 * length

        length, length_unit = self.time_unit_creator(length)
        print('[operation finished at %s  -  took %d %s]' % (
            datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S"), length, length_unit))
