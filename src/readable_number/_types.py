"""
Used types.
"""

from typing import Union
from .basic_class import BasicClass

Number = Union[int, float]
NewTypes = BasicClass
CalculationSupportsTypes = Union[Number, BasicClass]
SupportsTypes = Union[Number, str, NewTypes]
