"""
Fraction

The definition of the `Fraction` class.
"""

from enum import StrEnum, auto
from math import isnan, isinf
from typing import Any, Generator, Optional, Sequence, Union

from ._error_helper import (
    invalid_type,
    invalid_value,
)
from ._types import CalculationSupportsTypes, NewTypes, Number, SupportsTypes
from .basic_class import BasicClass
from .integer import Integer


# pylint: disable=import-outside-toplevel, protected-access


class LatexMode(StrEnum):
    """
    The Latex mode for the `to_latex` method.
    """

    AUTO = auto()
    FRAC = auto()  # same as NORMAL
    CFRAC = auto()
    TFRAC = auto()
    NORMAL = FRAC


class Fraction(BasicClass):
    """
    A class to represent a fraction.
    """

    # Whether to simplify the fraction after calculation.
    simplify_after_calculation = False

    # Whether the separator is between the space.
    # If True, the separator is " / ",
    # If False, the separator is "/".
    use_space_separator = False

    # Whether to optimize the recursive representation of the fraction.
    # For example, if a fraction is represented as `(...)/(...)`,
    # it will be optimized to `...` if this attribute is True.
    optimize_recursive_repr = False

    # ====================
    # initialization
    # ====================

    __match_args__ = ("numerator", "denominator")

    def __init__(
        self,
        numerator: SupportsTypes,
        denominator: SupportsTypes,
        *,
        simplify: bool = True,
    ) -> None:
        """
        The initialization of a Fraction object.
        NOTE: Will not copy the numerator and denominator.

        Args:
            numerator (SupportsTypes)
            denominator (SupportsTypes)
            simplify (bool, optional): Whether to simplify the fraction
                after handling the numerator and denominator.
                Defaults to True.

        Raises:
            ValueError: When the denominator is 0.
            FROM `_init_args_handler`:
                TypeError: When the numerator or denominator's type
                    is not in Supportedtypes.
                ValueError: When the str cannot be converted to a float.
        """

        numerator, denominator = self._init_args_handler(
            numerator, denominator
        )

        if denominator.value_eq(0):
            raise ValueError(
                invalid_value("denominator", denominator, expected="not 0")
            )

        self._numerator = numerator
        self._denominator = denominator
        self._force_do_hit_count = 0

        if simplify:
            self.simplify()

    @property
    def numerator(self) -> NewTypes:
        """
        Returns the numerator of the fraction.
        """
        return self._numerator

    @numerator.setter
    def numerator(self, value: SupportsTypes) -> None:
        """
        Sets the numerator of the fraction.
        """

        numerator, denominator = self._init_args_handler(
            value, self._denominator
        )

        if self._constants("COPY_WHEN_SETTING_ATTR", "bool"):
            self._numerator = numerator.copy(try_deep_copy=True)
            self._denominator = denominator.copy(try_deep_copy=True)
        else:
            self._numerator = numerator
            self._denominator = denominator

        if self._denominator.value_eq(0):  # pragma: no cover
            raise ValueError(
                invalid_value(
                    "denominator", self._denominator, expected="not 0"
                )
            )

    @property
    def denominator(self) -> NewTypes:
        """
        Returns the denominator of the fraction.
        """
        return self._denominator

    @denominator.setter
    def denominator(self, value: SupportsTypes) -> None:
        """
        Sets the denominator of the fraction.
        """

        numerator, denominator = self._init_args_handler(
            self._numerator, value
        )

        if self._constants("COPY_WHEN_SETTING_ATTR", "bool"):
            self._numerator = numerator.copy(try_deep_copy=True)
            self._denominator = denominator.copy(try_deep_copy=True)
        else:
            self._numerator = numerator
            self._denominator = denominator

        if self._denominator.value_eq(0):
            raise ValueError(
                invalid_value(
                    "denominator", self._denominator, expected="not 0"
                )
            )

    def _init_args_handler(
        self, numerator: SupportsTypes, denominator: SupportsTypes
    ) -> tuple[NewTypes, NewTypes]:
        """
        A helper function to handle the initialization arguments.

        Args:
            numerator (SupportsTypes)
            denominator (SupportsTypes)

        Raises:
            TypeError: When the numerator or denominator's type
                is not in Supportedtypes.
            ValueError: When the str cannot be converted to a float.

        Returns:
            tuple[NewTypes, NewTypes]: A tuple of numerator
                and denominator, converted to the appropriate type.
        """

        # shortcut
        if isinstance(numerator, int) and isinstance(denominator, int):
            return Integer(numerator), Integer(denominator)

        if not isinstance(numerator, SupportsTypes):
            raise TypeError(
                invalid_type("numerator", numerator, expected=SupportsTypes)
            )
        if not isinstance(denominator, SupportsTypes):
            raise TypeError(
                invalid_type(
                    "denominator", denominator, expected=SupportsTypes
                )
            )

        if isinstance(numerator, str):
            try:
                numerator = float(numerator)
            except ValueError as e:
                raise ValueError(
                    invalid_value(
                        "numerator",
                        numerator,
                        expected="which can be converted to a float",
                    )
                ) from e

        if isinstance(denominator, str):
            try:
                denominator = float(denominator)
            except ValueError as e:
                raise ValueError(
                    invalid_value(
                        "denominator",
                        denominator,
                        expected="which can be converted to a float",
                    )
                ) from e

        if isinstance(numerator, float) or isinstance(denominator, float):
            if isinf(numerator) or isnan(numerator):
                raise ValueError(
                    invalid_value(
                        "numerator",
                        numerator,
                        expected="which is not an infinity or NaN",
                    )
                )
            if isinf(denominator) or isnan(denominator):
                raise ValueError(
                    invalid_value(
                        "denominator",
                        denominator,
                        expected="which is not an infinity or NaN",
                    )
                )

            max_digits_after_decimal = max(
                (
                    len(str(abs(num)).split(".")[1].rstrip("0"))
                    if isinstance(num, float)
                    else 0
                )
                for num in (numerator, denominator)
            )
            numerator *= 10**max_digits_after_decimal
            denominator *= 10**max_digits_after_decimal
            if isinstance(numerator, float):
                numerator = round(numerator)
            if isinstance(denominator, float):
                denominator = round(denominator)

        # if isinstance(_, NewTypes):
        #     pass

        if isinstance(numerator, int):
            numerator = Integer(numerator)
        if isinstance(denominator, int):
            denominator = Integer(denominator)

        assert not isinstance(numerator, str)
        assert not isinstance(denominator, str)

        return numerator, denominator

    # ====================
    # public functions
    # ====================

    def get_factors(
        self, _circular_refs: Optional[set[NewTypes]] = None
    ) -> Generator[NewTypes, None, None]:
        """
        Gets the factors of the fraction.

        Raises:
            ValueError: When both numerator's and denominator's
                factors are empty.

        Yields:
            Generator[NewTypes, None, None]: A generator of factors.
        """

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if to_return:
            yield Integer(1)
            return

        n_factors = self._numerator.get_factors(_circular_refs)

        has_1 = False
        for n_factor in n_factors:
            for d_factor in self._denominator.get_factors(_circular_refs):
                if n_factor.value_eq(d_factor):
                    if has_1:
                        continue
                    has_1 = True

                yield Fraction(n_factor, d_factor)

    def simplify(self, _circular_refs: Optional[set[NewTypes]] = None) -> None:
        """
        Simplifies the fraction.
        """

        # shortcut
        if self._numerator.value_eq(0):
            # self._numerator = Integer(0)
            self._denominator = Integer(1)
            return

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if to_return:
            return

        self._numerator.simplify(_circular_refs)
        self._denominator.simplify(_circular_refs)

        numerator = self._numerator._optimize()
        denominator = self._denominator._optimize()

        n_factors = set(numerator.get_factors())
        d_factors = set(denominator.get_factors())

        common_factors = self.get_intersection_by_value(n_factors, d_factors)

        gcd = max(common_factors) * (-1 if Integer(-1) in d_factors else 1)
        gcd = gcd._optimize()

        original_numerator = self._numerator / gcd
        original_denominator = self._denominator / gcd

        numerator = original_numerator._optimize()
        denominator = original_denominator._optimize()

        exec_loop = False

        for _ in range(self._constants("SIMPLIFY_LOOP_LIMIT", ">-1")):
            if isinstance(numerator, Fraction):
                exec_loop = True
                denominator = denominator * numerator._denominator
                numerator = numerator._numerator
            else:
                break

        for _ in range(self._constants("SIMPLIFY_LOOP_LIMIT", ">-1")):
            if isinstance(denominator, Fraction):
                exec_loop = True
                numerator = numerator * denominator._denominator
                denominator = denominator._numerator
            else:
                break

        if isinstance(numerator, Fraction) or isinstance(
            denominator, Fraction
        ):  # or not exec_loop
            self._numerator = original_numerator._optimize()
            self._denominator = original_denominator._optimize()
            return

        if exec_loop:
            # If the loop has been executed,
            # we should calculate the gcd again.
            numerator = numerator._optimize()
            denominator = denominator._optimize()

            n_factors = set(numerator.get_factors())
            d_factors = set(denominator.get_factors())

            common_factors = self.get_intersection_by_value(
                n_factors, d_factors
            )

            gcd = max(common_factors) * (-1 if Integer(-1) in d_factors else 1)
            gcd = gcd._optimize()

            numerator = (numerator / gcd)._optimize()
            denominator = (denominator / gcd)._optimize()

        self._numerator = numerator
        self._denominator = denominator

    def simplify_without_change(self) -> "Fraction":
        """
        Returns a new Fraction object
        with simplified numerator and denominator.
        """
        return Fraction(self._numerator, self._denominator, simplify=True)

    @classmethod
    def from_auto(
        cls, item: Union[int, float, str, "Fraction"], *args, **kwargs
    ) -> "Fraction":
        """
        Creates a Fraction object from an integer, float, string
        or another Fraction object (will call `from_int`, `from_float`,
        `from_str` or `copy` method).
        Will set `simplify` to False by default.
        """

        # pylint: disable=unused-variable

        match item:
            case int(x) | Integer(x):
                return cls.from_int(item, *args, **kwargs)
            case float(x):
                return cls.from_float(item, *args, **kwargs)
            case str(x):
                return cls.from_str(item, *args, **kwargs)
            case Fraction(n, d):
                return item.copy(*args, **kwargs)

            case other:
                raise TypeError(
                    invalid_type(
                        "num",
                        item,
                        more_msg="when creating a Fraction from an integer, "
                        "float, string or another Fraction object",
                        expected="int, float, str or Fraction",
                    )
                )

        # pylint: enable=unused-variable

    @classmethod
    def from_int(cls, num: int, *args, **kwargs) -> "Fraction":
        """
        Creates a Fraction object from an integer.
        Will set `simplify` to False by default.
        """

        if not isinstance(num, int):
            raise TypeError(
                invalid_type(
                    "num",
                    num,
                    more_msg="when creating a Fraction from an integer",
                    expected=int,
                )
            )

        kwargs["simplify"] = kwargs.get("simplify", False)
        return cls(num, 1, *args, **kwargs)

    @classmethod
    def from_float(cls, num: float, *args, **kwargs) -> "Fraction":
        """
        Creates a Fraction object from a float.
        Will set `simplify` to False by default.
        """

        if not isinstance(num, float):
            raise TypeError(
                invalid_type(
                    "num",
                    num,
                    more_msg="when creating a Fraction from a float",
                    expected=float,
                )
            )
        if isinf(num) or isnan(num):
            raise ValueError(
                invalid_value(
                    "num",
                    num,
                    expected="which is not an infinity or NaN",
                )
            )

        kwargs["simplify"] = kwargs.get("simplify", False)
        return cls(*(num.as_integer_ratio()), *args, **kwargs)

    @classmethod
    def from_str(cls, s: str, *args, **kwargs) -> "Fraction":
        """
        Create a Fraction object from a string,
        in the form of f"{numerator}/{denominator}".
        (numerator and denominator should be an instance of int or float)

        Args:
            s (str): The string to be converted to a Fraction object.

        Raises:
            TypeError: When the s is not a string.
            ValueError: When the string cannot be converted
                to a Fraction. (i.e. not in the form of
                f"{numerator}/{denominator}")

        Returns:
            Fraction: The Fraction object created from the string.
        """

        if not isinstance(s, str):
            raise TypeError(
                invalid_type(
                    "s",
                    s,
                    more_msg="when creating a Fraction from a string",
                    expected=str,
                )
            )

        try:
            numerator, denominator = map(float, s.split("/"))
        except ValueError as e:
            raise ValueError(
                invalid_value(
                    "s",
                    s,
                    expected="in the form of f'{numerator}/{denominator}'",
                )
            ) from e

        return cls(numerator, denominator, *args, **kwargs)

    # def get_value(self) -> float:
    #     return float(self)

    def copy(
        self,
        *,
        copy_unknown_num: bool = False,
        try_deep_copy: bool = False,
        force_deep_copy: bool = False,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> "Fraction":
        """
        Creates a copy of the Fraction object.
        The parameter `force_deep_copy` is only effective when
        `try_deep_copy` is True.
        When doing a try_deep_copy, if the object's circular reference
        times is greater than the constant MAX_COPY_CR_DEPTH,
        it will return the object itself as the result.
        When doning a force_deep_copy, it will always try to use deepcopy
        to create a copy, until it causes ``RecursionError``,
        and then return the object itself.

        Args:
            copy_unknown_num (bool, optional):
                Whether to copy the UnknownNum object.
                Defaults to False.
            try_deep_copy (bool, optional):
                Whether to try to use deepcopy to create a copy.
                Defaults to False.
            force_deep_copy (bool, optional):
                Only effective when try_deep_copy is True.
                Whether to force to use deepcopy to create a copy.
                Defaults to False.
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            Fraction: The copy of the Fraction object.
        """

        if not try_deep_copy:
            return Fraction(self._numerator, self._denominator, simplify=False)

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_COPY_CR_DEPTH",
        )
        if to_return:
            return self

        # try_deep_copy=True
        if force_deep_copy:
            try:
                # do forced deep copy by not passing _circular_refs
                numerator = self._numerator.copy(
                    copy_unknown_num=copy_unknown_num,
                    try_deep_copy=True,
                    force_deep_copy=True,
                )
            except RecursionError:
                return self
        else:
            numerator = self._numerator.copy(
                copy_unknown_num=copy_unknown_num,
                try_deep_copy=True,
                force_deep_copy=False,
                _circular_refs=_circular_refs,
            )

        if force_deep_copy:
            try:
                # do forced deep copy by not passing _circular_refs
                denominator = self._denominator.copy(
                    copy_unknown_num=copy_unknown_num,
                    try_deep_copy=True,
                    force_deep_copy=True,
                )
            except RecursionError:
                return self
        else:
            denominator = self._denominator.copy(
                copy_unknown_num=copy_unknown_num,
                try_deep_copy=True,
                force_deep_copy=False,
                _circular_refs=_circular_refs,
            )

        return Fraction(numerator, denominator, simplify=False)

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

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if to_return:
            return set()

        return self._numerator.get_unknowns(_circular_refs) | (
            self._denominator.get_unknowns(_circular_refs)
        )

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

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if to_return or self._numerator.value_eq(0) or not unknown_nums:
            return [Integer(0)] * len(unknown_nums)

        if self._denominator.contain_unknown_num():
            raise NotImplementedError(  # pragma: no cover
                "Now only support simple equation. (self."
                "denominator.contain_unknown_num() is True)"
            )

        numerator_coefs = self._numerator.get_coefficient_of_unknowns(
            unknown_nums, False, _circular_refs=_circular_refs
        )
        # denominator_coefs = self._denominator.get_coefficient_of_unknowns(
        #     unknown_nums, _circular_refs=_circular_refs
        # )

        return [
            Fraction(n, self._denominator, simplify=_do_simplify)
            for n in numerator_coefs
        ]

    def contain_unknown_num(
        self, _circular_refs: Optional[set[NewTypes]] = None
    ) -> bool:
        """
        Returns True if the object contains an unknown number,
        otherwise False.
        """

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if to_return:
            return False

        return self._numerator.contain_unknown_num(
            _circular_refs=_circular_refs
        ) or self._denominator.contain_unknown_num(
            _circular_refs=_circular_refs
        )

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

        if values is None or not values:
            return

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if to_return:
            return

        self._numerator.set_values(values, _circular_refs=_circular_refs)
        self._denominator.set_values(values, _circular_refs=_circular_refs)

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
        """
        Represents the Fraction object in string format.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `str()` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_REPR_CR_DEPTH",
        )
        if to_return:
            return "..."

        result = []

        from .power import Power
        from .unknown_num import UnknownNum

        if not isinstance(self._numerator, (Integer, Power, UnknownNum)):
            result.append("(")
            result.append(
                n_str := self._numerator.to_string(
                    _circular_refs=_circular_refs,
                    **kwargs,
                )
            )
            result.append(")")
        else:
            result.append(n_str := str(self._numerator))

        if self.use_space_separator:
            result.append(" / ")
        else:
            result.append("/")

        if not isinstance(self._denominator, (Integer, Power, UnknownNum)):
            result.append("(")
            result.append(
                d_str := self._denominator.to_string(
                    _circular_refs=_circular_refs,
                    **kwargs,
                )
            )
            result.append(")")
        else:
            result.append(d_str := str(self._denominator))

        if self.optimize_recursive_repr:
            if n_str == "..." and d_str == "...":
                return "..."

        return "".join(result)

    # def __repr__(self) -> str:
    #     return self.do_repr()

    def do_repr(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        _force_do: bool = False,
    ) -> str:
        """
        A helper function to represent a Fraction object.
        """

        if not _force_do:
            assert self._force_do_hit_count == 0
            to_return, _circular_refs = self._circular_reference_dict(
                "do not pass `_circular_refs` in wrong type "
                "when calling this method, use `repr()` instead",
                _circular_refs=_circular_refs,
                constant_get_key="MAX_REPR_CR_DEPTH",
            )
            if to_return:
                return "..."
        elif self._force_do_hit_count > self._constants(
            "MAX_FORCE_DO_HIT_COUNT", "<13"
        ):
            self._force_do_hit_count = 0
            return "..."
        else:
            self._force_do_hit_count += 1

        if self is not self._numerator:
            numerator = self._numerator.do_repr(
                _circular_refs=_circular_refs,
                _force_do=_force_do,
            )
            self._numerator._force_do_hit_count = 0
        else:
            numerator = "..."

        if self is not self._denominator:
            denominator = self._denominator.do_repr(
                _circular_refs=_circular_refs, _force_do=_force_do
            )
            self._denominator._force_do_hit_count = 0
        else:
            denominator = "..."

        if self.optimize_recursive_repr:
            if numerator == "..." and denominator == "...":
                return "..."

        return f"Fraction({numerator}, {denominator})"

    def to_latex(
        self,
        *,
        fraction_manual_mode: Optional[LatexMode] = LatexMode.AUTO,
        fraction_auto_mode: bool = False,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Represent the Fraction object in LaTeX format.
        fraction_manual_mode will override fraction_auto_mode.

        Args:
            fraction_manual_mode (Optional[LatexMode], optional):
                The mode to use for LaTeX representation.
                Defaults to LatexMode.AUTO.
            fraction_auto_mode (bool, optional): Defaults to False.
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the fraction_manual_mode is not a string.
            ValueError: When the fraction_manual_mode is not supported.

        Returns:
            str: The LaTeX representation of the Fraction object.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_REPR_CR_DEPTH",
        )
        if to_return:
            return "..."

        if fraction_manual_mode is None:
            fraction_manual_mode = LatexMode.AUTO

        if not isinstance(fraction_manual_mode, LatexMode):
            raise TypeError(
                invalid_type(
                    "fraction_manual_mode",
                    fraction_manual_mode,
                    expected=str,
                )
            )

        # default "\\frac"
        # continued fraction "\\cfrac"
        # tiny fraction "\\tfrac"
        prefix = "\\frac"

        fraction_auto_mode = (
            fraction_auto_mode or fraction_manual_mode == LatexMode.AUTO
        )

        if not isinstance(self._numerator, Integer):
            if fraction_auto_mode:
                prefix = "\\cfrac"
            numerator = self._numerator.to_latex(
                fraction_manual_mode=fraction_manual_mode,
                fraction_auto_mode=fraction_auto_mode,
                _circular_refs=_circular_refs,
                **kwargs,
            )
        else:
            numerator = str(self._numerator)
        if not isinstance(self._denominator, Integer):
            if fraction_auto_mode:
                prefix = "\\cfrac"
            denominator = self._denominator.to_latex(
                fraction_manual_mode=fraction_manual_mode,
                fraction_auto_mode=fraction_auto_mode,
                _circular_refs=_circular_refs,
                **kwargs,
            )
        else:
            denominator = str(self._denominator)

        if self.optimize_recursive_repr:
            if numerator == "..." and denominator == "...":
                return "..."

        # fraction_manual_mode will override fraction_auto_mode
        match fraction_manual_mode:
            case LatexMode.AUTO:
                pass
            case LatexMode.FRAC:
                prefix = "\\frac"
            case LatexMode.CFRAC:
                prefix = "\\cfrac"
            case LatexMode.TFRAC:
                prefix = "\\tfrac"
            case _:  # pragma: no cover
                assert False, invalid_value(
                    "fraction_manual_mode",
                    fraction_manual_mode,
                    more_msg="The fraction_manual_mode should "
                    f"be one of {LatexMode.__members__}",
                )

        return f"{prefix}{{ {numerator} }}{{ {denominator} }}"

    # ====================
    # calculate
    # ====================

    # def __int__(self) -> int:
    #     return int(float(self))

    def do_float(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        _force_do: bool = False,
        **kwargs: Any,
    ) -> float:
        """
        A helper function to convert a Fraction object to a float.
        """

        if not _force_do:
            assert self._force_do_hit_count == 0
            to_return, _circular_refs = self._circular_reference_dict(
                "do not pass `_circular_refs` in wrong type "
                "when calling this method, use `float()` instead",
                _circular_refs=_circular_refs,
                constant_get_key="MAX_CALCULATION_CR_DEPTH",
            )
            if to_return:
                return 1.0
        elif self._force_do_hit_count > self._constants(
            "MAX_FORCE_DO_HIT_COUNT", "<13"
        ):
            self._force_do_hit_count = 0
            return 1.0
        else:
            self._force_do_hit_count += 1

        numerator = self._numerator.do_float(
            _circular_refs=_circular_refs,
            _force_do=_force_do,
            **kwargs,
        )
        self._numerator._force_do_hit_count = 0

        denominator = self._denominator.do_float(
            _circular_refs=_circular_refs,
            _force_do=_force_do,
            **kwargs,
        )
        self._denominator._force_do_hit_count = 0

        return numerator / denominator

    def do_abs(
        self, *, _circular_refs: Optional[dict[NewTypes, int]] = None
    ) -> "Fraction":
        """
        A helper function to calculate the absolute value of a Fraction object.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `abs()` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Fraction(1, 1, simplify=False)

        numerator = self._numerator.do_abs(_circular_refs=_circular_refs)
        denominator = self._denominator.do_abs(_circular_refs=_circular_refs)

        return Fraction(
            numerator, denominator, simplify=self.simplify_after_calculation
        ).copy(try_deep_copy=True)

    def do_add(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to add a Fraction object and
        an anothor object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Fraction, Multinomial]: The result of adding the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `+` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Fraction(1, 1, simplify=False)

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when adding a Fraction object",
                    expected=CalculationSupportsTypes,
                )
            )

        res = None

        # ====================
        # add Number
        # ====================

        if isinstance(other, Number) and (
            self._constants("TRY_KEEP_FRACTION", "bool")
            or isinstance(self._denominator, Integer)
        ):
            numerator1 = self._denominator.do_mul(
                other,
                _circular_refs=_circular_refs,
            )
            numerator = self._numerator.do_add(
                numerator1,
                _circular_refs=_circular_refs,
            )
            res = Fraction(
                numerator,
                self._denominator,
                simplify=self.simplify_after_calculation,
            )

        # ====================
        # add Fraction
        # ====================

        elif isinstance(other, Fraction):
            # numerator = self.numerator * other.denominator
            # ..          + self.denominator * other.numerator
            numerator = self._numerator.do_mul(
                other.denominator,
                _circular_refs=_circular_refs,
            ).do_add(
                self._denominator.do_mul(
                    other.numerator,
                    _circular_refs=_circular_refs,
                )
            )

            # denominator = self.denominator * other.denominator
            denominator = self._denominator.do_mul(
                other.denominator,
                _circular_refs=_circular_refs,
            )

            res = Fraction(
                numerator,
                denominator,
                simplify=self.simplify_after_calculation,
            )

        # ====================
        # add other types
        # ====================

        else:
            from .multinomial import Multinomial

            if isinstance(other, Multinomial):
                res = other.do_add(self, _circular_refs=_circular_refs)
            else:
                # simplify=False because the result cannot be simplified
                # in this case. Also avoid circular calling.
                res = Multinomial((self, other), simplify=False)

        return res.copy(try_deep_copy=True)

    def do_mul(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to multiply a Fraction object and
        an another object that is supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Fraction, Monomial]:
                The result of multiplying the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `*` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Fraction(1, 1, simplify=False)

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when multiplying a Fraction object",
                    expected=CalculationSupportsTypes,
                )
            )

        res = None

        # ====================
        # mul Number
        # ====================

        if isinstance(other, Number):
            numerator = self._numerator.do_mul(
                other,
                _circular_refs=_circular_refs,
            )
            res = Fraction(
                numerator,
                self._denominator,
                simplify=self.simplify_after_calculation,
            )

        # ====================
        # mul Fraction
        # ====================

        elif isinstance(other, Fraction):
            numerator = self._numerator.do_mul(
                other.numerator,
                _circular_refs=_circular_refs,
            )
            denominator = self._denominator.do_mul(
                other.denominator,
                _circular_refs=_circular_refs,
            )

            res = Fraction(
                numerator,
                denominator,
                simplify=self.simplify_after_calculation,
            )

        # ====================
        # mul other types
        # ====================

        else:
            from .monomial import Monomial

            if isinstance(other, Monomial):
                res = other.do_mul(self, _circular_refs=_circular_refs)
            else:
                # simplify=False because the result cannot be simplified
                # in this case. Also avoid circular calling.
                res = Monomial((self, other), simplify=False)

        return res.copy(try_deep_copy=True)

    def do_truediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> "Fraction":
        """
        A helper function to divide a Fraction object and
        an another object that is supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Fraction: The result of dividing the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `/` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Fraction(1, 1, simplify=False)

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when dividing a Fraction object",
                    expected=CalculationSupportsTypes,
                )
            )

        res = None

        # ====================
        # div Number
        # ====================

        if isinstance(other, Number):
            denominator = self._denominator.do_mul(
                other,
                _circular_refs=_circular_refs,
            )
            res = Fraction(
                self._numerator,
                denominator,
                simplify=self.simplify_after_calculation,
            )

        # ====================
        # div Fraction
        # ====================

        elif isinstance(other, Fraction):
            numerator = self._numerator.do_mul(
                other.denominator,
                _circular_refs=_circular_refs,
            )
            denominator = self._denominator.do_mul(
                other.numerator,
                _circular_refs=_circular_refs,
            )

            res = Fraction(
                numerator,
                denominator,
                simplify=self.simplify_after_calculation,
            )

        # ====================
        # div other types
        # ====================

        else:
            res = Fraction(
                self, other, simplify=self.simplify_after_calculation
            )

        return res.copy(try_deep_copy=True)

    def do_rtruediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> "Fraction":
        """
        A helper function to divide an another object and
        a Fraction object that is supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            Fraction: The result of dividing the other object
                and the Fraction object.
        """

        res = self.do_truediv(other, _circular_refs=_circular_refs)
        assert isinstance(res, Fraction)
        res = res.reciprocal()
        return res

    def reciprocal(self) -> "Fraction":
        """
        Returns the reciprocal of the Fraction object.

        Returns:
            Fraction: The reciprocal of the Fraction object.
        """
        return Fraction(
            self._denominator,
            self._numerator,
            simplify=False,
        ).copy(try_deep_copy=True)

    def self_reciprocal(self) -> None:
        """
        Modifies the Fraction object to become its reciprocal.
        """
        self._numerator, self._denominator = self._denominator, self._numerator

    def __pow__(self, other: CalculationSupportsTypes) -> NewTypes:
        """
        Raises the Fraction object to the power of the other object.

        Args:
            other (CalculationSupportsTypes): The exponent to which
                the Fraction object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Power: The result of raising the Fraction object to
                the power of the other object.
        """

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when raising a Fraction object to the power",
                    expected=CalculationSupportsTypes,
                )
            )

        from .power import Power

        if super()._constants("FORCE_FRACTION_AS_BASE", "bool"):
            return Power(self, other, simplify=self.simplify_after_calculation)

        return Fraction(
            Power(
                self._numerator,
                other,
                simplify=self.simplify_after_calculation,
            ),
            Power(
                self._denominator,
                other,
                simplify=self.simplify_after_calculation,
            ),
            simplify=self.simplify_after_calculation,
        ).copy(try_deep_copy=True)

    def __rpow__(self, other: CalculationSupportsTypes) -> NewTypes:
        """
        Raises the other object to the power of the Fraction object.

        Args:
            other (CalculationSupportsTypes): The base that
                is raised to the power of the Fraction object.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Power: The result of raising the other object
                to the power of the Fraction object.
        """

        from .power import Power

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when raising an object to the "
                    "power of a Fraction object",
                    expected=CalculationSupportsTypes,
                )
            )

        return Power(
            other, self, simplify=self.simplify_after_calculation
        ).copy(try_deep_copy=True)

    # ====================
    # check
    # ====================

    def do_exactly_eq(
        self,
        other: object,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> bool:
        """
        Checks if the Fraction object is exactly equal to the other object,
        which means both the numerator and denominator are equal.

        Args:
            other (object): The object to compare with the Fraction object.
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            bool: True if the Fraction object is exactly equal to
                the other object, False otherwise.
        """

        if not isinstance(other, Fraction):
            return False

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `==` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_COMPARISON_CR_DEPTH",
        )
        if to_return:
            return True

        return self.numerator.do_exactly_eq(
            other.numerator, _circular_refs=_circular_refs
        ) and self.denominator.do_exactly_eq(
            other.denominator, _circular_refs=_circular_refs
        )
