from time import time
from datetime import datetime


class TimeExpector:
    def __init__(self):
        self._last_time_length = None
        self._last_macro_time_length = None
        self.tick_tock_check = False
        self.t_start = None
        self.macro_start = None

    def reset(self):
        self._last_time_length = None
        self._last_macro_time_length = None
        self.tick_tock_check = False
        self.t_start = None
        self.macro_start = None

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

    def macro_tick(self):
        self.macro_start = time()

    def macro_tock(self):
        if self.macro_start is None:
            self.macro_tick()
            return ''

        length = time() - self.macro_start
        if self._last_macro_time_length is None:
            self._last_macro_time_length = length
        else:
            self._last_macro_time_length = .9 * self._last_macro_time_length + .1 * length

        self.macro_tick()

        length, length_unit = self.time_unit_creator(length)
        return 'took %d %s' % (length, length_unit)

    def tick(self):
        if self.tick_tock_check:
            print('[WARNING] calling "tick" without a "tock"...')
        self.t_start = time()
        self.tick_tock_check = True

    def tock(self):
        if not self.tick_tock_check:
            print('[WARNING] calling "tock" without a "tick"...')
            return ''

        self.tick_tock_check = False
        length = time() - self.t_start
        if self._last_time_length is None:
            self._last_time_length = length
        else:
            self._last_time_length = .995 * self._last_time_length + .005 * length

        return datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S")

    def expectation(self, iteration_left, sub_iteration_left, sub_iteration_count):
        if self._last_macro_time_length is not None:
            expectation = self._last_macro_time_length * iteration_left + self._last_time_length * sub_iteration_left
        elif self._last_time_length is not None:
            expectation = self._last_time_length * (iteration_left * sub_iteration_count + sub_iteration_left)
        else:
            return ''
        length, length_unit = self.time_unit_creator(expectation)
        return '%d %s remaining' % (length, length_unit)
