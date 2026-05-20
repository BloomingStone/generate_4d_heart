import math
from typing import ClassVar
from dataclasses import dataclass

DEFAULT_BINS = 1_000_000

@dataclass(frozen=True)
class CardiacPhase:
    """
    Cardiac phase class, representing the phase value range in the cardiac cycle: [0, 1), 
    where 0 indicates the beginning of the cycle and close to 1 indicates the end of the cycle
    """
    tick: int           # tick \in [0, bins-1], the index of the phase in the cardiac cycle
    bins: ClassVar[int] = DEFAULT_BINS   # the total number of bins to discretize the cardiac cycle, default is 10000 for high precision
    
    def __post_init__(self):
        assert isinstance(self.tick, int) and 0 <= self.tick < self.bins, "tick must be an integer in the range of [0, bins)"
    
    @staticmethod
    def from_index(index: int, total_phases: int) -> "CardiacPhase":
        """
        Create a CardiacPhase object from the index of the phase in the cardiac cycle
        """
        assert isinstance(total_phases, int) and total_phases > 0, "total_phases must be a positive integer"
        assert isinstance(index, int) and index >= 0 and index < total_phases, "index must be a non-negative integer less than total_phases"
        
        tick = index * CardiacPhase.bins // total_phases
        return CardiacPhase(tick=tick)
    
    
    @staticmethod
    def from_time(t: float, cardiac_cycle_time: float) -> "CardiacPhase":
        """
        Create a CardiacPhase object from the time of the phase in the cardiac cycle
        when t = 0, phase = 0
        """
        assert isinstance(cardiac_cycle_time, (int, float)) and cardiac_cycle_time > 0, "cardiac_cycle_time must be a positive number"
        assert isinstance(t, (int, float)) and t >= 0, "t must be a positive number"
        phase = (t % cardiac_cycle_time) / cardiac_cycle_time
        tick = round(phase * CardiacPhase.bins) % CardiacPhase.bins
        if tick == CardiacPhase.bins:
            tick = 0
        return CardiacPhase(tick=tick)

    def lower_index(self, total_phases: int) -> int:
        assert total_phases > 0
        return (self.tick * total_phases) // self.bins
    
    def upper_index(self, total_phases: int, overflow_to_loop: bool = True) -> int:
        assert total_phases > 0
        floor = self.lower_index(total_phases)
        ceil = floor + 1
        if overflow_to_loop:
            ceil = ceil % total_phases
        else:
            ceil = min(ceil, total_phases - 1)
        return ceil

    def __float__(self):
        return self.tick / self.bins
    
    def __format__(self, format_spec):
        return format(float(self), format_spec)
    
    def __repr__(self):
        return f"CardiacPhase({float(self):.4f})"
    
    def __str__(self):
        return f"{float(self):.4f}"
    
    def to_str(self, precision: int = 4, has_decimal_point: bool = True) -> str:
        if has_decimal_point:
            return f"{self:.{precision}f}"
        else:
            return f"{float(self):.{precision}f}".replace('.', '_')
    
    def __add__(self, other: "CardiacPhase"):
        if isinstance(other, CardiacPhase):
            tick = (self.tick + other.tick) % self.bins
        else:
            return NotImplemented
        
        return CardiacPhase(tick=tick)
    
    def __sub__(self, other: "CardiacPhase"):
        if isinstance(other, CardiacPhase):
            tick = (self.tick - other.tick) % self.bins
        else:
            return NotImplemented
        
        return CardiacPhase(tick=tick)