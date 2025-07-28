"""
Power

The definition of the `Power` class.
"""

from typing import Any, Generator, Literal, Optional, Sequence, Union

from ._error_helper import (
    invalid_type,
    invalid_value,
)
from ._types import CalculationSupportsTypes, NewTypes, SupportsTypes
from .basic_class import BasicClass
from .integer import Integer


# pylint: disable=import-outside-toplevel, protected-access


class Power(BasicClass):
    """
    A class to represent a power of a number.
    """

    # Whether to simplify the fraction after calculation.
    simplify_after_calculation = False

    # Whether the separator is between the space.
    # If True, the separator is " ^ " for example,
    # if False, the separator is "^".
    use_space_separator = False

    # Whether to optimize the recursive representation of the power.
    # For example, if a power is represented as `(...)^(...)`,
    # it will be optimized to `...` if this attribute is True.
    optimize_recursive_repr = False

    # ====================
    # initialization
    # ====================

    __match_args__ = ("base", "exponent")

    def __init__(
        self,
        base: SupportsTypes,
        exponent: SupportsTypes,
        *,
        separator: Literal["^", "**"] = "^",
        simplify: bool = True,
    ) -> None:
        """
        The initialization of the Power class.

        Args:
            base (SupportsTypes)
            exponent (SupportsTypes)
            separator (Literal["^", "**"], optional): The separator between
                the base and exponent.
                Only "^" and "**" are allowed now.
                Defaults to "^".
            simplify (bool, optional): Whether to simplify the power
                after handling the base and exponent.
                Defaults to True.
        """

        base, exponent, separator = self._init_args_handler(
            base, exponent, separator
        )

        self._base = base.copy(try_deep_copy=True)
        self._exponent = exponent.copy(try_deep_copy=True)
        self._force_do_hit_count = 0
        self._separator = separator

        if simplify:
            self.simplify()

    @property
    def base(self) -> NewTypes:
        """
        Returns the base of the power.
        """
        return self._base

    @base.setter
    def base(self, value: SupportsTypes) -> None:
        """
        Sets the base of the power.
        """

        base, exponent, _ = self._init_args_handler(value, self._exponent, "^")

        self.base = base.copy(try_deep_copy=True)
        self.exponent = exponent.copy(try_deep_copy=True)

    @property
    def exponent(self) -> NewTypes:
        """
        Returns the exponent of the power.
        """
        return self._exponent

    @exponent.setter
    def exponent(self, value: SupportsTypes) -> None:
        """
        Sets the exponent of the power.
        """

        base, exponent, _ = self._init_args_handler(self._base, value, "^")

        self.base = base.copy(try_deep_copy=True)
        self.exponent = exponent.copy(try_deep_copy=True)

    @property
    def separator(self) -> Literal["^", "**"]:
        """
        Returns the separator of the power.
        """
        return self._separator

    @separator.setter
    def separator(self, value: Literal["^", "**"]) -> None:
        """
        Sets the separator of the power.
        """

        if not isinstance(value, str):
            raise TypeError(invalid_type("separator", value, expected="str"))

        value = value.strip()  # type: ignore
        if value not in ["^", "**"]:
            raise ValueError(
                invalid_value(
                    "separator",
                    value,
                    expected="^ or **",
                )
            )

        self._separator = value  # type: Literal["^", "**"]

    def _init_args_handler(
        self,
        base: SupportsTypes,
        exponent: SupportsTypes,
        separator: Literal["^", "**"],
    ) -> tuple[NewTypes, NewTypes, Literal["^", "**"]]:
        """
        A helper function to handle the initialization arguments.

        Args:
            base (SupportsTypes)
            exponent (SupportsTypes)
            separator (Literal["^", "**"])

        Raises:
            TypeError: When the base or exponent's type is not
                in SupportsTypes.
            ValueError: When the str cannot be converted to float.

        Returns:
            tuple[NewTypes, NewTypes, Literal["^", "**"]]:
                A tuple of base, exponent, and separator.
        !       (NOT ENSURED that the base and exponent are in the
        !       appropriate types)
        """

        if not isinstance(separator, str):
            raise TypeError(
                invalid_type("separator", separator, expected="str")
            )

        separator = separator.strip()  # type: ignore
        if separator not in ["^", "**"]:
            raise ValueError(
                invalid_value(
                    "separator",
                    separator,
                    expected="^ or **",
                )
            )

        # short cut
        if isinstance(base, int) and isinstance(exponent, int):
            return Integer(base), Integer(exponent), separator

        if not isinstance(base, SupportsTypes):
            raise TypeError(invalid_type("base", base, expected=SupportsTypes))
        if not isinstance(exponent, SupportsTypes):
            raise TypeError(
                invalid_type("exponent", exponent, expected=SupportsTypes)
            )

        if isinstance(base, str):
            try:
                base = float(base)
            except ValueError as e:
                raise ValueError(
                    invalid_value(
                        "base",
                        base,
                        expected="which can be converted to float",
                    )
                ) from e
            assert isinstance(base, float)

        if isinstance(exponent, str):
            try:
                exponent = float(exponent)
            except ValueError as e:
                raise ValueError(
                    invalid_value(
                        "exponent",
                        exponent,
                        expected="which can be converted to float",
                    )
                ) from e
            assert isinstance(exponent, float)

        if isinstance(base, float):
            from .fraction import Fraction

            base = Fraction.from_float(base)
        if isinstance(exponent, float):
            from .fraction import Fraction

            exponent = Fraction.from_float(exponent)

        # if isinstance(_, NewTypes):
        #     pass

        if isinstance(base, int):
            base = Integer(base)
        if isinstance(exponent, int):
            exponent = Integer(exponent)

        return base, exponent, separator

    # ====================
    # public funcions
    # ====================

    def get_factors(
        self, _circular_refs: Optional[set[NewTypes]] = None
    ) -> Generator[NewTypes, None, None]:
        """
        Gets the factors of the power.

        Note that we cannot use the exponent's factors,
        because of the calculation of the power.
        (e.g. 2**7 = 2**(3+4) = 2**3 * 2**4)
        So we use `range(2, int(self._exponent) + 1)`
        to evaluate the `factors`.

        Yields:
            Generator[NewTypes, None, None]: A generator of factors.
        """

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if (
            to_return
            or self._base.value_eq(0)  # 0 is not included in the generator.
            or self._exponent.value_eq(0)
        ):
            yield Integer(1)
            return

        if self._exponent < 0:
            obj = Power(self._base, -self._exponent, simplify=False)
            yield from obj.get_factors(_circular_refs)
            return

        b_factors = self._base.get_factors(_circular_refs)

        has_1 = False
        for b_factor in b_factors:
            for e_factor in range(1, int(self._exponent) + 1):
                if b_factor.value_eq(1):  # or e_factor.value_eq(0)
                    if not has_1:
                        has_1 = True
                        yield Integer(1)
                    continue

                yield Power(b_factor, e_factor)

    def simplify(self, _circular_refs: Optional[set[NewTypes]] = None) -> None:
        """
        Simplifies the power.
        """

        # short cut
        if self._base.value_eq(0):
            # self._base = Integer(0)
            self._exponent = Integer(0)
            return
        if self._exponent.value_eq(0):
            self._base = Integer(1)
            # self._exponent = Integer(0)
            return

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if to_return:
            return

        self._base.simplify(_circular_refs)
        # exponent cannot be simplified, because of the situation
        # like this: (-3)**(2/4) == ((-3)**2)**(1/4) != (-3)**(1/2)
        # self._exponent.simplify(_circular_refs)

        if self._exponent < 0 and not self._constants(
            "ALLOW_NEGATIVE_EXPONENT", "bool"
        ):
            # if self._constants("FORCE_FRACTION_AS_BASE", "bool"):
            from .fraction import Fraction

            self._base = Fraction(1, self._base)
            exponent = -self._exponent
            if isinstance(exponent, int):
                exponent = Integer(exponent)
            self._exponent = exponent

            # else:
            #     # cannot convert self to Fraction in Power class
            #     # Will be handled in Monomial class
            #     pass

        if self._base > self._constants("EXPECTED_MAX_BASE", ">2"):
            # cannot convert self to Monomial in Power class
            # Will be handled in Monomial class
            pass

        if self._exponent > self._constants("EXPECTED_MAX_EXPONENT", ">2"):
            # cannot convert self to Monomial in Power class
            # Will be handled in Monomial class or Multinomial class
            pass

    def simplify_without_change(self) -> "Power":
        """
        Returns a new Power object
        with simplified base and exponent.
        """
        return Power(self._base, self._exponent, simplify=True)

    @classmethod
    def from_auto(
        cls, item: Union[int, str, "Power"], *args, **kwargs
    ) -> "Power":
        """
        Creates a Power object from an integer, float, string,
        or another Power object (will call `from_int`, `from_str`
        or `copy` method).
        Will set `simplify` to False by default.
        """

        # pylint: disable=unused-variable

        match item:
            case int(x) | Integer(x):
                return cls.from_int(item, *args, **kwargs)
            case str(x):
                return cls.from_str(item, *args, **kwargs)
            case Power(b, e):
                return item.copy(*args, **kwargs)

            case other:
                raise TypeError(
                    invalid_type(
                        "item",
                        item,
                        more_msg="when creating a Power object from "
                        "an integer, float, string, "
                        "or another Power object",
                        expected="int, float, str, or Power object",
                    )
                )

        # pylint: enable=unused-variable

    @classmethod
    def from_int(cls, num: int, *args, **kwargs) -> "Power":
        """
        Creates a Power object from an integer.
        Will set `simplify` to False by default.
        """

        if not isinstance(num, int):
            raise TypeError(
                invalid_type(
                    "num",
                    num,
                    more_msg="when creating a Power object from an integer",
                    expected=int,
                )
            )

        kwargs["simplify"] = kwargs.get("simplify", False)
        return cls(num, 1, *args, **kwargs)

    @classmethod
    def from_str(
        cls, s: str, *args, separator: Literal["^", "**"] = "^", **kwargs
    ) -> "Power":
        """
        Creates a Power object from a string,
        in the form of "{base}{separator}{exponent}".
        (base and exponent should be an instance of int or float)

        Args:
            s (str): The string to be converted to a Power object.

        Raises:
            TypeError: When the s is not a string.
            ValueError: When the string cannot be converted to
                a Power object. (i.e. not in the form of
                f"{base}{separator}{exponent}")

        Returns:
            Power: The Power object created from the string.
        """

        if not isinstance(s, str):
            raise TypeError(
                invalid_type(
                    "s",
                    s,
                    more_msg="when creating a Power object from a string",
                    expected=str,
                )
            )

        if not isinstance(separator, str):
            raise TypeError(
                invalid_type("separator", separator, expected="str")
            )

        separator = separator.strip()  # type: ignore
        if separator not in ["^", "**"]:
            raise ValueError(
                invalid_value(
                    "separator",
                    separator,
                    expected="^ or **",
                )
            )

        try:
            base, exponent = map(float, s.split(separator))
        except ValueError as e:
            raise ValueError(
                invalid_value(
                    "s",
                    s,
                    expected="in the form of f'{base}"
                    f"{separator}{{exponent}}'",
                )
            ) from e

        return cls(base, exponent, *args, separator=separator, **kwargs)

    # def get_value(self) -> float:
    #     return float(self)

    def copy(
        self,
        *,
        copy_unknown_num: bool = False,
        try_deep_copy: bool = False,
        force_deep_copy: bool = False,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> "Power":
        """
        Creates a copy of the Power object.
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
            Power: The copy of the Power object.
        """

        if not try_deep_copy:
            return Power(self._base, self._exponent, simplify=False)

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
                base = self._base.copy(
                    copy_unknown_num=copy_unknown_num,
                    try_deep_copy=True,
                    force_deep_copy=True,
                )
            except RecursionError:
                return self
        else:
            base = self._base.copy(
                copy_unknown_num=copy_unknown_num,
                try_deep_copy=True,
                force_deep_copy=False,
                _circular_refs=_circular_refs,
            )

        if force_deep_copy:
            try:
                # do forced deep copy by not passing _circular_refs
                exponent = self._exponent.copy(
                    copy_unknown_num=copy_unknown_num,
                    try_deep_copy=True,
                    force_deep_copy=True,
                )
            except RecursionError:
                return self
        else:
            exponent = self._exponent.copy(
                copy_unknown_num=copy_unknown_num,
                try_deep_copy=True,
                force_deep_copy=False,
                _circular_refs=_circular_refs,
            )

        return Power(base, exponent, simplify=False)

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

        return self._base.get_unknowns(_circular_refs) | (
            self._exponent.get_unknowns(_circular_refs)
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
        if to_return:
            return [Integer(0)] * len(unknown_nums)

        if not self._exponent.value_eq(1):
            if self.contain_unknown_num():
                raise NotImplementedError(
                    "Now only support simple equation. (self.exponent: "
                    f"{self._exponent.to_string()} is not 1)"
                )

            # base_coefs = self._base.get_coefficient_of_unknowns(
            #     unknown_nums, _circular_refs=_circular_refs
            # )
            # exponent_coefs = self._exponent.get_coefficient_of_unknowns(
            #     unknown_nums, _circular_refs=_circular_refs
            # )

            return [Integer(0)] * len(unknown_nums)

        return self._base.get_coefficient_of_unknowns(
            unknown_nums, _do_simplify, _circular_refs=_circular_refs
        )

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

        return self._base.contain_unknown_num(_circular_refs) or (
            self._exponent.contain_unknown_num(_circular_refs)
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

        self._base.set_values(values, _circular_refs=_circular_refs)
        self._exponent.set_values(values, _circular_refs=_circular_refs)

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
        Represents the Power object in string format.
        If the exponent is a `Fraction` object whose
        numerator is 1, and denominator is a integer that
        is greater than 1, the `Power` object will be
        represented as a radical form.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_REPR_CR_DEPTH",
        )
        if to_return:
            return "..."

        from .unknown_num import UnknownNum

        result = []
        no_parentheses_types = (Integer, Power, UnknownNum)

        # If the exponent is a `Fraction` object whose
        # numerator 1, and denominator is a integer that
        # is greater than 1, the `Power` object will be
        # represented as a radical form.
        if self._constants("TRY_TO_CONVERT_TO_RADICAL", "bool"):
            from .fraction import Fraction

            # 2**(1/2) -> √2
            # Currently only consider conversions
            # when the denominator of exponent is an integer
            # greater than 1.
            if isinstance(self._exponent, Fraction):
                if (
                    # actually an Integer object
                    isinstance(self._exponent.denominator, int)
                    and self._exponent.denominator > 1
                ):
                    if self._exponent.numerator.value_eq(1):
                        if self._exponent.denominator.value_eq(2):
                            result.append("√")
                        else:
                            result.append(
                                f"[{str(self._exponent.denominator)}]√"
                            )
                        if not isinstance(self._base, no_parentheses_types):
                            result.append("(")
                            result.append(
                                self._base.to_string(
                                    _circular_refs=_circular_refs,
                                    **kwargs,
                                )
                            )
                            result.append(")")
                        else:
                            result.append(str(self._base))
                        return "".join(result)

                    # isint(self._exponent.denominator)
                    # and self._exponent.denominator > 1
                    # and self._exponent.numerator != 1:
                    if self._exponent.numerator <= self._constants(
                        "MAX_RADICAL_ROOT_EXPONENT", ">2"
                    ):
                        if self._exponent.denominator.value_eq(2):
                            result.append("(√")
                        else:
                            result.append(
                                f"([{str(self._exponent.denominator)}]√"
                            )
                        if not isinstance(self._base, no_parentheses_types):
                            result.append("(")
                            result.append(
                                self._base.to_string(
                                    _circular_refs=_circular_refs,
                                    **kwargs,
                                )
                            )
                            result.append(")")
                        else:
                            result.append(str(self._base))

                        if self.use_space_separator:
                            sep = f" {self._separator} "
                        else:
                            sep = self._separator
                        result.append(f"){sep}")

                        if not isinstance(
                            self._exponent.numerator, no_parentheses_types
                        ):
                            result.append("(")
                            result.append(
                                self._exponent.numerator.to_string(
                                    _circular_refs=_circular_refs,
                                    **kwargs,
                                )
                            )
                            result.append(")")
                        else:
                            result.append(str(self._exponent.numerator))
                        return "".join(result)

        if not isinstance(self._base, no_parentheses_types):
            result.append("(")
            result.append(
                b_str := self._base.to_string(
                    _circular_refs=_circular_refs,
                    **kwargs,
                )
            )
            result.append(")")
        else:
            result.append(b_str := str(self._base))

        if self.use_space_separator:
            sep = f" {self._separator} "
        else:
            sep = self._separator
        result.append(sep)

        if not isinstance(self._exponent, no_parentheses_types):
            result.append("(")
            result.append(
                e_str := self._exponent.to_string(
                    _circular_refs=_circular_refs,
                    **kwargs,
                )
            )
            result.append(")")
        else:
            result.append(e_str := str(self._exponent))

        if self.optimize_recursive_repr:
            if b_str == "..." and e_str == "...":
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
        A helper function to represent the Power object.
        """

        if not _force_do:
            assert self._force_do_hit_count == 0
            to_return, _circular_refs = self._circular_reference_dict(
                "do not pass `_circular_refs` in wrong type "
                "when calling this method",
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

        base = self._base.do_repr(
            _circular_refs=_circular_refs, _force_do=_force_do
        )
        self._base._force_do_hit_count = 0

        exponent = self._exponent.do_repr(
            _circular_refs=_circular_refs, _force_do=_force_do
        )
        self._exponent._force_do_hit_count = 0

        if self.optimize_recursive_repr:
            if base == "..." and exponent == "...":
                return "..."

        return f"Power({base}, {exponent})"

    def to_latex(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Represent the Power object in LaTeX format.
        manual_mode and auto_mode are not used in Power class.
        If the exponent is a `Fraction` object whose
        numerator is 1, and denominator is a integer that
        is greater than 1, the `Power` object will be
        represented as a radical form.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_REPR_CR_DEPTH",
        )
        if to_return:
            return "..."

        from .unknown_num import UnknownNum

        result = []
        no_parentheses_types = (Integer, Power, UnknownNum)

        # If the exponent is a `Fraction` object whose
        # numerator 1, and denominator is a integer that
        # is greater than 1, the `Power` object will be
        # represented as a radical form.
        if self._constants("TRY_TO_CONVERT_TO_RADICAL", "bool"):
            from .fraction import Fraction

            # 2**(5/3) -> {\sqrt[3]{2}}^5
            # Currently only consider conversions
            # when the denominator of exponent is an integer
            # greater than 1.
            if isinstance(self._exponent, Fraction):
                if (
                    isinstance(self._exponent.denominator, int)
                    and self._exponent.denominator > 1
                ):
                    if self._exponent.numerator.value_eq(1):
                        if self._exponent.denominator.value_eq(2):
                            result.append("\\sqrt")
                        else:
                            result.append(
                                f"\\sqrt[{str(self._exponent.denominator)}]"
                            )
                        result.append("{")
                        if not isinstance(self._base, no_parentheses_types):
                            result.append("\\left( ")
                            result.append(
                                self._base.to_latex(
                                    _circular_refs=_circular_refs,
                                    **kwargs,
                                )
                            )
                            result.append(" \\right)")
                        else:
                            result.append(str(self._base))
                        result.append("}")
                        return "".join(result)

                    # isint(self._exponent.denominator)
                    # and self._exponent.denominator > 1
                    # and self._exponent.numerator != 1:
                    if self._exponent.numerator <= self._constants(
                        "MAX_RADICAL_ROOT_EXPONENT", ">1"
                    ):
                        if self._exponent.denominator.value_eq(2):
                            result.append("{\\sqrt")
                        else:
                            result.append(
                                f"{{\\sqrt[{str(self._exponent.denominator)}]"
                            )
                        result.append("{")
                        if not isinstance(self._base, no_parentheses_types):
                            result.append("\\left( ")
                            result.append(
                                self._base.to_latex(
                                    _circular_refs=_circular_refs,
                                    **kwargs,
                                )
                            )
                            result.append(" \\right)")
                        else:
                            result.append(str(self._base))

                        result.append("}^{")

                        if not isinstance(
                            self._exponent.numerator, no_parentheses_types
                        ):
                            result.append("\\left( ")
                            result.append(
                                self._exponent.numerator.to_latex(
                                    _circular_refs=_circular_refs,
                                    **kwargs,
                                )
                            )
                            result.append(" \\right)")
                        else:
                            result.append(str(self._exponent.numerator))
                        result.append("}")
                        return "".join(result)

        result.append("{")
        if not isinstance(self._base, no_parentheses_types):
            result.append("\\left( ")
            result.append(
                b_str := self._base.to_latex(
                    _circular_refs=_circular_refs,
                    **kwargs,
                )
            )

            result.append(" \\right)")
        else:
            result.append(b_str := str(self._base))

        result.append("}^{")

        if not isinstance(self._exponent, no_parentheses_types):
            result.append("\\left( ")
            result.append(
                e_str := self._exponent.to_latex(
                    _circular_refs=_circular_refs,
                    **kwargs,
                )
            )
            result.append(" \\right)")
        else:
            result.append(e_str := str(self._exponent))
        result.append("}")

        if self.optimize_recursive_repr:
            if b_str == "..." and e_str == "...":
                return "..."

        return "".join(result)

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
        A helper function to convert the Power object to float.
        """

        if not _force_do:
            assert self._force_do_hit_count == 0
            to_return, _circular_refs = self._circular_reference_dict(
                "do not pass `_circular_refs` in wrong type "
                "when calling this method",
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

        base = self._base.do_float(
            _circular_refs=_circular_refs,
            _force_do=_force_do,
            **kwargs,
        )
        self._base._force_do_hit_count = 0

        from .fraction import Fraction

        if isinstance(self._exponent, Fraction):
            numerator = self._exponent.numerator.do_float(
                _circular_refs=_circular_refs,
                _force_do=_force_do,
                **kwargs,
            )
            self._exponent.numerator._force_do_hit_count = 0

            denominator = Fraction(1, self._exponent.denominator).do_float(
                _circular_refs=_circular_refs,
                _force_do=_force_do,
                **kwargs,
            )
            self._exponent.denominator._force_do_hit_count = 0

            return (base**numerator) ** denominator

        exponent = self._exponent.do_float(
            _circular_refs=_circular_refs,
            _force_do=_force_do,
            **kwargs,
        )
        self._exponent._force_do_hit_count = 0

        return base**exponent

    def do_abs(
        self, *, _circular_refs: Optional[dict[NewTypes, int]] = None
    ) -> "Power":
        """
        A helper function to calculate the absolute value of the Power object.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Power(1, 1, simplify=False)

        base = self._base.do_abs(_circular_refs=_circular_refs)

        return Power(
            base,
            self._exponent,
            simplify=self.simplify_after_calculation,
        ).copy(try_deep_copy=True)

    def do_add(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to add a Power object and
        an anothor object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Power, Multinomial, Monomial]:
                The result of adding the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `+` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Power(1, 1, simplify=False)

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when adding a Power object",
                    expected=CalculationSupportsTypes,
                )
            )

        if (
            isinstance(other, Power)
            and self._base.do_exactly_eq(other.base)
            and self._exponent.do_exactly_eq(other.exponent)
        ):
            from .monomial import Monomial

            # simplify=False because the result cannnot be simplified
            # in this case. Also avoid circular calling.
            res = Monomial((2, self), simplify=False)

        else:
            from .multinomial import Multinomial

            if isinstance(other, Multinomial):
                res = other.do_add(self, _circular_refs=_circular_refs)
            else:
                # simplify=False because the result cannnot be simplified
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
        A helper function to multiply a Power object and
        an anothor object that is supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Power, Multinomial, Monomial]:
                The result of multiplying the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `*` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Power(1, 1, simplify=False)

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when multiplying a Power object",
                    expected=CalculationSupportsTypes,
                )
            )

        if isinstance(other, Power):
            if self._base.do_exactly_eq(other.base):
                exponent = self._exponent.do_add(
                    other.exponent, _circular_refs=_circular_refs
                )
                res = Power(
                    self._base,
                    exponent,
                    simplify=self.simplify_after_calculation,
                )
            elif self._exponent.do_exactly_eq(other.exponent):
                base = self._base.do_mul(
                    other.base, _circular_refs=_circular_refs
                )
                res = Power(
                    base,
                    self._exponent,
                    simplify=self.simplify_after_calculation,
                )
            else:
                from .monomial import Monomial

                # simplify=False because the result cannnot be simplified
                # in this case. Also avoid circular calling.
                res = Monomial((self, other), simplify=False)

        else:
            from .monomial import Monomial

            if isinstance(other, Monomial):
                res = other.do_mul(self, _circular_refs=_circular_refs)
            else:
                # simplify=False because the result cannnot be simplified
                # in this case. Also avoid circular calling.
                res = Monomial((self, other), simplify=False)

        return res.copy(try_deep_copy=True)

    def do_truediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to divide a Power object and
        an anothor object that is supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            Union[Fraction, Power]: The result of
                dividing the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `/` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Power(1, 1, simplify=False)

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when dividing a Power object",
                    expected=CalculationSupportsTypes,
                )
            )

        if isinstance(other, Power) and self._base.do_exactly_eq(other.base):
            exponent = self._exponent.do_sub(
                other.exponent, _circular_refs=_circular_refs
            )
            res = Power(
                self._base,
                exponent,
                simplify=self.simplify_after_calculation,
            )

        elif isinstance(other, Integer) and self._base.do_exactly_eq(other):
            exponent = self._exponent.do_sub(1, _circular_refs=_circular_refs)
            res = Power(
                self._base,
                exponent,
                simplify=self.simplify_after_calculation,
            )

        else:
            from .fraction import Fraction

            res = Fraction(
                self, other, simplify=self.simplify_after_calculation
            )

        return res.copy(try_deep_copy=True)

    def do_rtruediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to divide an anothor object and
        a Power object that is supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Fraction, Power]: The result of
                dividing the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `/` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Power(1, 1, simplify=False)

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when dividing a Power object",
                    expected=CalculationSupportsTypes,
                )
            )

        if isinstance(other, Power) and self._base.do_exactly_eq(other.base):
            exponent = other.exponent.do_sub(
                self._exponent, _circular_refs=_circular_refs
            )
            res = Power(
                self._base,
                exponent,
                simplify=self.simplify_after_calculation,
            )

        elif isinstance(other, Integer) and self._base.do_exactly_eq(other):
            exponent = self._exponent.do_rsub(1, _circular_refs=_circular_refs)
            res = Power(
                self._base,
                exponent,
                simplify=self.simplify_after_calculation,
            )

        else:
            from .fraction import Fraction

            res = Fraction(
                other, self, simplify=self.simplify_after_calculation
            )

        return res.copy(try_deep_copy=True)

    def __pow__(self, other: CalculationSupportsTypes) -> "Power":
        """
        Raises the Power object to the power of the other object.

        Args:
            other (CalculationSupportsTypes): The exponent to which
                the Power object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Power: The result of raising the Power object to
                the power of the other object.
        """

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when raising a Power object",
                    expected=CalculationSupportsTypes,
                )
            )

        exponent = self._exponent.do_mul(other)

        return Power(
            self._base,
            exponent,
            simplify=self.simplify_after_calculation,
        ).copy(try_deep_copy=True)

    def __rpow__(self, other: CalculationSupportsTypes) -> "Power":
        """
        Raises the other object to the power of the Power object.

        Args:
            other (CalculationSupportsTypes): The base to which
                the Power object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Power: The result of raising the other object to
                the power of the Power object.
        """

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when raising a Power object",
                    expected=CalculationSupportsTypes,
                )
            )

        return Power(
            other,
            self,
            simplify=self.simplify_after_calculation,
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
        Checks if the Power object is exactly equal to the other object,
        which means both the base and the exponent are equal.

        Args:
            other (object): The object to be compared with.
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            bool: True if the two objects are exactly equal, False otherwise.
        """

        if not isinstance(other, Power):
            return False

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_COMPARISON_CR_DEPTH",
        )
        if to_return:
            return True

        return self._base.do_exactly_eq(
            other.base, _circular_refs=_circular_refs
        ) and self._exponent.do_exactly_eq(
            other.exponent, _circular_refs=_circular_refs
        )
