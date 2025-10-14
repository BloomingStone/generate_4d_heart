import math
from dataclasses import dataclass

@dataclass
class CardiacPhase:
    """
    Cardiac phase class, representing the phase value range in the cardiac cycle: [0, 1), 
    where 0 indicates the beginning of the cycle and close to 1 indicates the end of the cycle
    """
    phase: float
    
    def __post_init__(self):
        if not isinstance(self.phase, (int, float)) or self.phase < 0 or self.phase >= 1:
            raise ValueError("phase must be a number in the range of [0, 1)")
    
    @staticmethod
    def from_index(index: int, total_phases: int) -> "CardiacPhase":
        """
        Create a CardiacPhase object from the index of the phase in the cardiac cycle
        """
        assert isinstance(total_phases, int) and total_phases > 0, "total_phases must be a positive integer"
        assert isinstance(index, int) and index >= 0 and index < total_phases, "index must be a non-negative integer less than total_phases"
        return CardiacPhase(index / total_phases)
    
    
    @staticmethod
    def from_time(t: float, cardiac_cycle_time: float) -> "CardiacPhase":
        """
        Create a CardiacPhase object from the time of the phase in the cardiac cycle
        when t = 0, phase = 0
        """
        assert isinstance(cardiac_cycle_time, (int, float)) and cardiac_cycle_time > 0, "cardiac_cycle_time must be a positive number"
        assert isinstance(t, (int, float)) and t >= 0, "t must be a positive number"
        return CardiacPhase((t % cardiac_cycle_time) / cardiac_cycle_time)
    
    
    def closest_index_floor(self, total_phases: int) -> int:
        """
        Return the closest index to the phase value, rounding down to the nearest integer
        """
        assert isinstance(total_phases, int) and total_phases > 0, "total_phases must be a positive integer"
        
        return math.floor(self.phase * total_phases)
    
    def closest_index_ceil(self, total_phases: int) -> int:
        """
        Return the closest index to the phase value, rounding down to the nearest integer,
        the index in [0, total_phases) ^ N
        """
        assert isinstance(total_phases, int) and total_phases > 0, "total_phases must be a positive integer"
        
        res = math.ceil(self.phase * total_phases)
        if res == total_phases:
            res = 0
        
        return res


    def __float__(self):
        return self.phase
    
    def __repr__(self):
        return f"CardiacPhase({self.phase:.4f})"
    
    def __eq__(self, other):
        if isinstance(other, CardiacPhase):
            return abs(self.phase - other.phase) < 1e-5
        elif isinstance(other, (int, float)):
            return abs(self.phase - other) < 1e-5
        return False
    
    def __add__(self, other):
        if isinstance(other, CardiacPhase):
            new_phase = self.phase + other.phase
        elif isinstance(other, (int, float)):
            new_phase = self.phase + other
        else:
            return NotImplemented
        
        new_phase = new_phase % 1.0  # make sure it's in the range of [0, 1)
        return CardiacPhase(new_phase)
    
    def __sub__(self, other):
        if isinstance(other, CardiacPhase):
            diff = self.phase - other.phase
        elif isinstance(other, (int, float)):
            diff = self.phase - other
        else:
            return NotImplemented
        
        diff = diff % 1.0   # # make sure it's in the range of [0, 1)
        return CardiacPhase(diff)