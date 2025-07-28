"""
Integer

The defination of the `Integer` class.
"""

from typing import Any, Generator, Optional, Sequence, Union

from ._error_helper import (
    invalid_type,
    invalid_value,
)
from ._types import CalculationSupportsTypes, NewTypes
from .basic_class import BasicClass


# pylint: disable=import-outside-toplevel, protected-access, unused-argument


class Integer(BasicClass, int):
    """
    A class to represent an Integer.
    """

    # ====================
    # initialization
    # ====================

    __match_args__ = ("value",)

    def __new__(cls, value: Union[int, float, str]) -> "Integer":
        """
        Creates a new `Integer` object.

        Args:
            value (Union[int, float, str]): The value of the Integer.

        Returns:
            A new `Integer` object.
        """

        try:
            return int.__new__(cls, value)
        except TypeError as e:
            raise TypeError(
                invalid_type(
                    "value",
                    value,
                    expected="an Integer, float, or a string "
                    "representing an Integer",
                )
            ) from e
        except ValueError as e:
            raise ValueError(
                invalid_value(
                    "value",
                    value,
                    expected="an Integer, float, or a string "
                    "representing an Integer",
                )
            ) from e

    def __init__(self, value: Union[int, float, str]) -> None:
        """
        Initializes the Integer object.

        Args:
            value (Union[int, float, str]): The value of the Integer.
        """
        self.__value = int(value)
        self._force_do_hit_count = 0

    # ====================
    # public functions
    # ====================

    def get_factors(
        self, _circular_refs: Optional[set[NewTypes]] = None
    ) -> Generator["Integer", None, None]:
        """
        Gets the factors of the Integer.

        Args:
            _circular_refs (Optional[set[NewTypes]], optional):
                A set to keep track of circular references.
                Defaults to None.

        Yields:
            Generator[Integer, None, None]: A generator of factors.
        """

        num = int(self)

        if num == 0:
            return
        if num < 0:
            yield Integer(-1)
            num = -num

        yield Integer(1)
        for i in range(2, num + 1):
            if num % i == 0:
                yield Integer(i)

    def simplify(self, _circular_refs: Optional[set[NewTypes]] = None) -> None:
        """
        Simplifies the Integer.
        """

    def simplify_without_change(self) -> "Integer":
        """
        Simplifies the Integer without changing it.
        """
        return self.copy()

    @classmethod
    def from_auto(cls, item: Union[int, str], *args, **kwargs) -> "Integer":
        """
        Creates a new Integer object from an item.

        Args:
            item (Union[int, str]): The item to create
                the Integer object from.

        Returns:
            Integer: A new Integer object.
        """
        return cls(item)

    def copy(
        self,
        *,
        copy_unknown_num: bool = False,
        try_deep_copy: bool = False,
        force_deep_copy: bool = False,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> "Integer":
        """
        Copies the Integer object.

        Args:
            try_deep_copy (bool, optional):
                Whether to try to use deep copy.
                Defaults to False.
            force_deep_copy (bool, optional):
                Whether to force deep copy.
                Defaults to False.
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            Integer: A new Integer object.
        """
        return Integer(int(self))

    # ====================
    # Supports for UnknownNum
    # ====================

    def get_unknowns(
        self,
        _circular_refs: Optional[set[NewTypes]] = None,
    ) -> set[NewTypes]:
        """
        Returns the UnknownNum objects in the container.
        """
        return set()

    def get_coefficient_of_unknowns(
        self,
        unknown_nums: Sequence[Union[NewTypes, str]],
        _do_simplify: bool = True,
        _circular_refs: Optional[set[NewTypes]] = None,
    ) -> list[NewTypes]:
        """
        Returns the coefficients of the UnknownNum objects in the container.
        If the object does not contain the UnknownNum object,
        the coefficient will be 0.
        NOTE: Now only support simple equation.
        """
        return [Integer(0)] * len(unknown_nums)

    def contain_unknown_num(
        self, _circular_refs: Optional[set[NewTypes]] = None
    ) -> bool:
        """
        Returns True if the object contains an unknown number,
        otherwise False.
        """
        return False

    def set_values(
        self,
        values: Optional[
            dict[Union[NewTypes, str], Optional[CalculationSupportsTypes]]
        ] = None,
        _circular_refs: Optional[set[NewTypes]] = None,
    ) -> None:
        """
        Sets the values of the UnknownNum object.
        """

    # ====================
    # represent
    # ====================

    # def __str__(self) -> str:
    #     return self.to_string()

    def to_string(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        **kwargs: Any,
    ) -> str:
        return str(int(self))

    # def __repr__(self) -> str:
    #     return self.do_repr()

    def do_repr(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        _force_do: bool = False,
    ) -> str:
        return f"Integer({int(self)})"

    def to_latex(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        **kwargs: Any,
    ) -> str:
        return str(int(self))

    # ====================
    # calculate
    # ====================

    def do_int(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> int:
        return self.__value

    def do_float(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        _force_do: bool = False,
        **kwargs: Any,
    ) -> float:
        return float(self.__value)

    def do_abs(
        self, *, _circular_refs: Optional[dict[NewTypes, int]] = None
    ) -> "Integer":
        return Integer(abs(int(self)))

    def do_add(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:

        if isinstance(other, (int, Integer)):
            return Integer(int(self) + int(other))

        if isinstance(other, float):
            from .fraction import Fraction

            return Fraction.from_float(float(self) + other)

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `+` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Integer(1)

        if isinstance(other, BasicClass):
            return other.do_add(self, _circular_refs=_circular_refs)

        raise TypeError(
            invalid_type(
                "other",
                other,
                expected="an Integer, float, or BasicClass object",
            )
        )

    def do_mul(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:

        if isinstance(other, (int, Integer)):
            return Integer(int(self) * int(other))

        if isinstance(other, float):
            from .fraction import Fraction

            return Fraction.from_float(float(self) * other)

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `*` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Integer(1)

        if isinstance(other, BasicClass):
            return other.do_mul(self, _circular_refs=_circular_refs)

        raise TypeError(
            invalid_type(
                "other",
                other,
                expected="an Integer, float, or BasicClass object",
            )
        )

    def do_truediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:

        if isinstance(other, (int, Integer)):
            res = int(self) / int(other)
            if res.is_integer():
                return Integer(res)

            from .fraction import Fraction

            return Fraction(self, other)

        if isinstance(other, float):
            from .fraction import Fraction

            return Fraction(self, other)

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `/` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Integer(1)

        if isinstance(other, BasicClass):
            return other.do_rtruediv(self, _circular_refs=_circular_refs)

        raise TypeError(
            invalid_type(
                "other",
                other,
                expected="an Integer, float, or BasicClass object",
            )
        )

    def do_rtruediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:

        if isinstance(other, (int, Integer)):
            res = int(other) / int(self)
            if isinstance(res, int) or res.is_integer():
                return Integer(res)

            from .fraction import Fraction

            return Fraction(other, self)

        if isinstance(other, float):
            from .fraction import Fraction

            return Fraction(other, self)

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `/` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Integer(1)

        if isinstance(other, BasicClass):
            return other.do_truediv(self, _circular_refs=_circular_refs)

        raise TypeError(
            invalid_type(
                "other",
                other,
                expected="an Integer, float, or BasicClass object",
            )
        )

    def __pow__(  # type: ignore
        self,
        other: CalculationSupportsTypes,
    ) -> NewTypes:
        from .power import Power

        return Power(self, other)

    def __rpow__(  # type: ignore
        self,
        other: CalculationSupportsTypes,
    ) -> NewTypes:
        from .power import Power

        return Power(other, self)

    # ====================
    # check
    # ====================

    def do_exactly_eq(
        self,
        other: object,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> bool:
        if not isinstance(other, int):
            return False
        return int(self) == int(other)

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, float(self)))
