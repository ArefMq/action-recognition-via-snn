
from enum import Enum


class TimeReduction(Enum):
    NoTimeReduction = None
    SpikeRate = "SpikeRate"
    SpikeTime = "SpikeTime"
    MemRecMax = "MemRecMax"
    MemRecMean = "MemRecMean"

class PollingReduction(Enum):
    MaxSpikeRate = "MaxSpikeRate"
    AvgSpikeRate = "AvgSpikeRate"
    SpikeTime = "SpikeTime"