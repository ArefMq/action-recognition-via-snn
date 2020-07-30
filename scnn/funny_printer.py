from random import choice, random
from math import fabs

EMOJI_PACK = {
    'foot_slow': [
        '🚶‍♀️',
        '🚶',
        '🚶‍♂️',
    ],

    'foot_fast': [
        '🏃‍♀️',
        '🏃',
        '🏃‍♂️',
        '🚴‍♀️',
        '🚴',
        '🚴‍♂️'
    ],

    'animal_slow': [
        '🐢',
    ],
    'animal_fast': [
        '🐎',
    ],
    'race_slow': [
        '🏍',
    ],

    'race_fast': [
        '🏎',
    ],

    'car_slow': [
        '🚗',
        '🚕',
        '🚙',
        '🚌',
        '🚗',
        '🚕',
        '🚙',
        '🚌',
        '🚗',
        '🚕',
        '🚙',
        '🚌',
        '🚗',
        '🚕',
        '🚙',
        '🚌',
        '🚗',
        '🚎',
        '🚒',
        '🚐',
        '🚚',
        '🚛',
        '🚜',
        '🚲',
        '🛵',
        '🏍',
        '🛺',
    ],

    'car_fast': [
        '🚗',
        '🚗',
        '🚕',
        '🚙',
        '🚗',
        '🚗',
        '🚕',
        '🚙',
        '🚗',
        '🚗',
        '🚕',
        '🚙',
        '🚓',
        '🚑',
        '🚒',
        '🏍',
    ],

    'air_slow': [
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🛸',
    ],

    'air_fast': [
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🚁',
        '🛸',
    ],

    'boat_slow': [
        '🚣‍♀️',
        '🚣',
        '🚣‍♂️',
        '🏊‍♀️',
        '🏊',
        '🏊‍♂️',
        '⛵️',
        '🚣‍♀️',
        '🚣',
        '🚣‍♂️',
        '🏊‍♀️',
        '🏊',
        '🏊‍♂️',
        '⛵️',
        '⛵️',
        '⛵️',
        '⛵️',
        '🚤',
        '🛥',
    ],

    'boat_fast': [
        '🚤',
        '🛥',
        '🚤',
    ],

    'sky_x': [
        '☁️',
        '🌥',
        '⛅️',
        '🌧',
        '🌨',
    ],

    'sea_x': [
        '🌊',
    ],

    'race_x': [
        '💯',
    ],

    'city_x': [
        '🏠',
        '🏡',
        '🏘',
        '🏚',
        '🏦',
        '🏥',
        '🏤',
        '🏣',
        '🏬',
        '🏢',
        '🏭',
        '🏨',
        '🏪',
        '🏫',
        '🏩',
        '💒',
        '🏛',
        '⛪️',
        '🕌',
        '🕍',
    ],

    'safari_x': [
        '⛰',
        '🏔',
        '🗻',
        '🏕',
        '🌵',
    ],

    'sky_y': [
        '🌤',
        '☀️',
        '🌦',
        '⛈',
        '🌩',
    ],

    'sea_y': [
        '🌊',
        '🏖',
        '🏝',
    ],

    'city_y': [
        '🏠',
        '🏡',
        '🏘',
        '🏚',
        '🏦',
        '🏥',
        '🏤',
        '🏣',
        '🏬',
        '🏢',
        '🏭',
        '🏨',
        '🏪',
        '🏫',
        '🏩',
        '💒',
        '🏛',
        '⛪️',
        '🕌',
        '🕍',
    ],

    'park_y': [
        '🗽',
        '🏰',
        '🏯',
        '🏟',
        '🌲',
        '🌳',
        '🌴',
        '🎄',

        '🌹',
        '🌷',
        '🌻',
        '🍄',
    ],

    'safari_y': [
        '🏜',
        '🌋',
        '⛺️',
    ],

    'race_y': [
        '🏁',
    ],

    'park_x': [
        '🌲',
        '🌳',
        '🌴',
        '🎄',
    ],
}

RELATIONS = [
    {'foot': ['city', 'park', 'safari']},
    {'car': ['city', 'park']},
    {'foot': ['city', 'park']},
    {'car': ['city']},
    {'foot': ['city', 'park']},
    {'car': ['city']},
    {'foot': ['city', 'park']},
    {'car': ['city']},
    {'foot': ['city', 'park']},
    {'car': ['city']},
    {'foot': ['city', 'park']},
    {'car': ['city']},
    {'foot': ['city', 'park']},
    {'car': ['city']},
    {'foot': ['city', 'park']},
    {'car': ['city']},
    {'foot': ['city', 'park']},
    {'car': ['city']},
    {'air': ['sky']},
    {'boat': ['sea']},
    {'air': ['sky']},
    {'boat': ['sea']},
    {'air': ['sky']},
    {'boat': ['sea']},
    {'animal': ['safari', 'park']},
    {'race': ['race']},
]


class FunPack:
    def __init__(self):
        self.fu = None
        self.fp = None
        self.pattern_a = 0
        self.init_fp()

    @staticmethod
    def gamma(x, z):
        return z / (z + fabs(x))

    def r(self, i, c):
        if self.pattern_a > 0:
            self.pattern_a -= 1
            return ' '

        if i < self.gamma(c, 0.2):
            self.pattern_a = 3
            return 'y'

        if i < self.gamma(c, 0.6):
            self.pattern_a = 2
            return 'x'
        return ' '

    def init_fp(self, l=40):
        r = choice(RELATIONS)
        c = choice(list(r.values())[0])
        r = list(r.keys())[0]

        self.fu = {
            'x': choice(EMOJI_PACK[c + '_x']),
            'y': choice(EMOJI_PACK[c + '_y']),
            '>': choice(EMOJI_PACK[r + '_slow']),
            '>>': choice(EMOJI_PACK[r + '_fast'])
        }

        fun_pattern = ''.join([self.r(random(), c-l/2) for c in range(l)])
        self.fp = list(''.join(fun_pattern).replace('x', self.fu['x']).replace('y', self.fu['y']))

    def funnify(self, i, c, p):

        lfp = len(self.fp)

        if c == '.' or c == '-' or c == '=':
            x = self.fp[i % lfp]
            if x == ' ' and c == '-':
                return '.'
            elif x == ' ' and c == '=':
                return '-'
            else:
                return x
        if c == '>':
            if p == '.':
                return self.fu['>']
            elif p == '-':
                return self.fu['>>']
        return c
