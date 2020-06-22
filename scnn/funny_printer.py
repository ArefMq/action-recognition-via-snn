from random import choice

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
        '⛪️'
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

FUN_PATTERN = list('     x       x      y        x           x                     y       x         x')


class FunPack:
    def __init__(self):
        self.fu = None
        self.fp = None
        self.init_fp()

    def init_fp(self):
        r = choice(RELATIONS)
        c = choice(list(r.values())[0])
        r = list(r.keys())[0]

        self.fu = {
            'x': choice(EMOJI_PACK[c + '_x']),
            'y': choice(EMOJI_PACK[c + '_y']),
            '>': choice(EMOJI_PACK[r + '_slow']),
            '>>': choice(EMOJI_PACK[r + '_fast'])
        }
        self.fp = list(''.join(FUN_PATTERN).replace('x', self.fu['x']).replace('y', self.fu['y']))

    def funnify(self, i, c, p):

        lfp = len(self.fp)

        if c == '.' or c == '-' or c == '=':
            x = self.fp[i % lfp]
            if x == ' ' and c == '-':
                return '.'
            else:
                return x
        if c == '>':
            if p == '.':
                return self.fu['>']
            elif p == '-':
                return self.fu['>>']
        return c
