"""
UnknownNum

The definition of the `UnknownNum` class, which is used to
represent an unknown number.
"""

from enum import Enum, auto
from typing import Any, Generator, Optional, Sequence, Union
from weakref import WeakValueDictionary

from ._error_helper import (
    invalid_type,
    invalid_value,
)
from ._types import CalculationSupportsTypes, NewTypes, Number, SupportsTypes
from .basic_class import BasicClass
from .integer import Integer


# pylint: disable=import-outside-toplevel, protected-access


class UnknownShowMode(Enum):
    """
    The mode to show the unknown number.
    """

    LABEL = auto()
    STRING = auto()
    LATEX = auto()
    VALUE = auto()
    DEFAULT = VALUE


class UnknownNum(BasicClass):
    """
    A class to represent an unknown number.
    """

    # The wrapper string around the string representation of
    # the unknown number if ' ' (space) is found in the string.
    wrapper = ("(", ")")

    # ====================
    # initialization
    # ====================

    __match_args__ = ("label", "value", "string", "latex")

    __instances = WeakValueDictionary()

    def __new__(
        cls,
        label: str = "x",
        *,
        value: Optional[SupportsTypes] = None,
        string: Optional[str] = None,
        latex: Optional[str] = None,
        simplify: bool = False,
    ) -> "UnknownNum":
        """
        The initialization of an UnknownNum object.

        If `label` already exists in the `__instances` dictionary,
        the existing instance will be returned.


        FROM `__init__`:

        The initialization of an UnknownNum object.

        The `value` means the value of the unknown number,
        which can be converted to a number, or an instance of
        `BasicClass`. If `value` is None, the `UnknownNum` object
        cannot be got a value.
        Note that if the value is set, the `string` and `latex`
        will be ignored. And they will use the `value.to_string()`
        and `value.to_latex()` to get the string and latex
        representation of the unknown number.

        If `string` or `latex` is None, the `string` or `latex`
        representation of the unknown number will be same
        as the `label`.

        Args:
            label (str, optional): The label of the unknown number.
                Defaults to "x".
            value (Optional[SupportsTypes], optional): The value of the
                unknown number. Defaults to None.
            string (Optional[str], optional): The string representation
                of the unknown number. Defaults to None.
            latex (Optional[str], optional): The latex representation
                of the unknown number. Defaults to None.
            simplify (bool, optional): Whether to simplify the value
                of the unknown number. Defaults to False.

        Raises:
            TypeError: When any of the arguments is not a string.
            ValueError: When any of the arguments is an empty string.
        """

        if not isinstance(label, str):
            raise TypeError(invalid_type("label", label, expected="a string"))

        if not label:
            raise ValueError(
                invalid_value("label", label, expected="not an empty string")
            )

        if label not in cls.__instances:
            ret = super().__new__(cls)
            cls.__instances[label] = ret
        return cls.__instances[label]

    def __init__(
        self,
        label: str = "x",
        *,
        value: Optional[SupportsTypes] = None,
        string: Optional[str] = None,
        latex: Optional[str] = None,
        simplify: bool = False,
    ) -> None:
        """
        The initialization of an UnknownNum object.

        NOTE: You will overwirte the attributes of the object if
            you call the second time of `__init__` with the same
            `label`.

        The `value` means the value of the unknown number,
        which can be converted to a number, or an instance of
        `BasicClass`. If `value` is None, the `UnknownNum` object
        cannot be got a value.
        Note that if the value is set, the `string` and `latex`
        will be ignored. And they will use the `value.to_string()`
        and `value.to_latex()` to get the string and latex
        representation of the unknown number.

        If `string` or `latex` is None, the `string` or `latex`
        representation of the unknown number will be same
        as the `label`.

        Args:
            label (str, optional): The label of the unknown number.
                Defaults to "x".
            value (Optional[SupportsTypes], optional): The value of the
                unknown number. Defaults to None.
            string (Optional[str], optional): The string representation
                of the unknown number. Defaults to None.
            latex (Optional[str], optional): The latex representation
                of the unknown number. Defaults to None.
            simplify (bool, optional): Whether to simplify the value
                of the unknown number. Defaults to False.

        Raises:
            TypeError: When any of the arguments is not a string.
            ValueError: When any of the arguments is an empty string.
        """

        if not isinstance(label, str):
            raise TypeError(invalid_type("label", label, expected="a string"))

        if not label:
            raise ValueError(
                invalid_value("label", label, expected="not an empty string")
            )

        if string is None:
            string = label
        if latex is None:
            latex = label

        if not isinstance(string, str):
            raise TypeError(
                invalid_type("string", string, expected="a string")
            )

        if not string:
            raise ValueError(
                invalid_value("string", string, expected="not an empty string")
            )

        if not isinstance(latex, str):
            raise TypeError(invalid_type("latex", latex, expected="a string"))

        if not latex:
            raise ValueError(
                invalid_value("latex", latex, expected="not an empty string")
            )

        if value is not None and not isinstance(value, SupportsTypes):
            raise TypeError(
                invalid_type(
                    "value", value, expected="a number or a BasicClass"
                )
            )

        match value:
            case None:
                self._value = None

            case int(num) | Integer(num):
                self._value = Integer(num)

            case float(num):
                from .fraction import Fraction

                self._value = Fraction.from_float(num)

            case UnknownNum(label, value, string, latex):
                raise ValueError(
                    "Cannot create an UnknownNum object whose value is "
                    "an instance of UnknownNum."
                )

            case other if isinstance(other, BasicClass):
                self._value = other.copy(try_deep_copy=False)

            case other:
                assert False

        self._label = label
        self._string = string
        self._latex = latex
        self._force_do_hit_count = 0

        if simplify and self._value is not None:
            self.simplify()

    @property
    def label(self) -> str:
        """
        Returns the label of the unknown number.
        """
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        """
        Sets the label of the unknown number.
        """
        if not isinstance(value, str):
            raise TypeError(invalid_type("label", value, expected="a string"))

        if not value:
            raise ValueError(
                invalid_value("label", value, expected="not an empty string")
            )

        self._label = value

    @property
    def o_label(self) -> str:
        """
        Returns the optimized label of the unknown number.
        """

        if len(self.wrapper) != 2:
            raise ValueError(
                invalid_value(
                    "wrapper",
                    self.wrapper,
                    expected="a tuple of two objects",
                )
            )

        if " " in self._label or self._label == "...":
            return f"{self.wrapper[0]}{self._label}{self.wrapper[1]}"
        return self._label

    @property
    def value(self) -> Optional[NewTypes]:
        """
        Returns the value of the unknown number.
        """
        return self._value

    @value.setter
    def value(self, value: Optional[SupportsTypes]) -> None:
        """
        Sets the value of the unknown number.
        """
        if not isinstance(value, Optional[SupportsTypes]):
            raise TypeError(
                invalid_type(
                    "value", value, expected="a number or a BasicClass"
                )
            )

        match value:
            case None:
                self._value = None
            case int(num) | Integer(num):
                self._value = Integer(num)
            case float(num):
                from .fraction import Fraction

                self._value = Fraction.from_float(num)
            case other if isinstance(other, BasicClass):
                self._value = other.copy(try_deep_copy=False)
            case other:
                assert False

    @property
    def string(self) -> str:
        """
        Returns the string representation of the unknown number.

        If the value of the unknown number is not None, the string
        representation will be the `to_string()` method of the value.
        Otherwise, the string representation will be the `string`
        property of the unknown number.
        """
        return self._string

    @string.setter
    def string(self, value: str) -> None:
        """
        Sets the string representation of the unknown number.
        """
        if not isinstance(value, str):
            raise TypeError(invalid_type("string", value, expected="a string"))

        if not value:
            raise ValueError(
                invalid_value("string", value, expected="not an empty string")
            )

        self._string = value

    @property
    def o_string(self) -> str:
        """
        Returns the optimized string representation
        of the unknown number.
        """

        if len(self.wrapper) != 2:
            raise ValueError(
                invalid_value(
                    "wrapper",
                    self.wrapper,
                    expected="a tuple of two objects",
                )
            )

        if " " in self._string or self._string == "...":
            return f"{self.wrapper[0]}{self._string}{self.wrapper[1]}"
        return self._string

    @property
    def latex(self) -> str:
        """
        Returns the latex representation of the unknown number.

        If the value of the unknown number is not None, the latex
        representation will be the `to_latex()` method of the value.
        Otherwise, the latex representation will be the `latex`
        property of the unknown number.
        """
        return self._latex

    @latex.setter
    def latex(self, value: str) -> None:
        """
        Sets the latex representation of the unknown number.
        """
        if not isinstance(value, str):
            raise TypeError(invalid_type("latex", value, expected="a string"))

        if not value:
            raise ValueError(
                invalid_value("latex", value, expected="not an empty string")
            )

        self._latex = value

    # ====================
    # public functions
    # ====================

    def get_factors(
        self, _circular_refs: Optional[set[NewTypes]] = None
    ) -> Generator[NewTypes, None, None]:
        """
        Returns the factors of the unknown number.

        If the value of the unknown number is not None, the factors
        will be the `get_factors()` method of the value.
        Otherwise, the factors will be an empty generator.
        """

        to_return, _circular_refs = self._circular_reference_set(
            _circular_refs=_circular_refs,
        )
        if to_return:
            yield Integer(1)
            return

        if self._value is not None:
            yield from self._value.get_factors(_circular_refs)
        else:
            yield Integer(1)

    def simplify(self, _circular_refs: Optional[set[NewTypes]] = None) -> None:
        """
        Simplifies the unknown number.

        If the value of the unknown number is not None, the `simplify()`
        method of the value will be called.
        """

        to_return, _circular_refs = self._circular_reference_set(
            _circular_refs=_circular_refs,
        )
        if to_return:
            return

        if self._value is not None:
            self._value.simplify(_circular_refs)

    def simplify_without_change(self) -> "UnknownNum":
        """
        Simplifies the unknown number without changing it.

        If the value of the unknown number is not None, the
        `simplify_without_change()` method of the value will be called.
        """

        if self._value is not None:
            return UnknownNum(
                label=self._label,
                value=self._value.simplify_without_change(),
                string=self._string,
                latex=self._latex,
                simplify=True,
            )
        return UnknownNum(
            label=self._label,
            value=None,
            string=self._string,
            latex=self._latex,
            simplify=True,
        )

    @classmethod
    def from_auto(
        cls,
        item: Union[SupportsTypes, None],
        *args,
        **kwargs,
    ) -> "UnknownNum":
        """
        Creates an UnknownNum object from an integer, float, string, or
        BasicClass object.
        Will create an UnknownNum object with the value of the input item.
        """
        return cls(*args, value=item, **kwargs)

    def copy(
        self,
        *,
        copy_unknown_num: bool = False,
        try_deep_copy: bool = False,
        force_deep_copy: bool = False,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> "UnknownNum":
        """
        Returns a copy of the unknown number.
        The parameter `force_deep_copy` is only effective when
        `try_deep_copy` is True.
        When doing a try_deep_copy, if the object's circular reference
        times is greater than the constant MAX_COPY_CR_DEPTH,
        it will return the object itself as the result.
        When doning a force_deep_copy, it will always try to use deepcopy
        to create a copy, until it causes ``RecursionError``,
        and then return the object itself.

        If the value of the unknown number is not None, the `copy()`
        method of the value will be called.

        Args:
            copy_unknown_num (bool, optional):
                Whether to copy the unknown number object.
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
            UnknownNum: A copy of the unknown number.
        """

        if not copy_unknown_num:
            return self

        if not try_deep_copy:
            return UnknownNum(
                label=self._label,
                value=self._value,
                string=self._string,
                latex=self._latex,
            )

        to_return, _circular_refs = self._circular_reference_dict(
            _circular_refs=_circular_refs,
            constant_get_key="MAX_COPY_CR_DEPTH",
        )
        if to_return:
            return self

        # try_deep_copy=True
        if self._value is None:
            return UnknownNum(
                label=self._label,
                value=None,
                string=self._string,
                latex=self._latex,
            )

        if force_deep_copy:
            try:
                # do forced deep copy by not passing _circular_refs
                value = self._value.copy(
                    copy_unknown_num=True,
                    try_deep_copy=True,
                    force_deep_copy=True,
                )
            except RecursionError:
                return self
        else:
            value = self._value.copy(
                copy_unknown_num=True,
                try_deep_copy=True,
                force_deep_copy=False,
                _circular_refs=_circular_refs,
            )

        return UnknownNum(
            label=self._label,
            value=value,
            string=self._string,
            latex=self._latex,
        )

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
        return {self}

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

        res = []
        for unknown_num in unknown_nums:
            if isinstance(unknown_num, str):
                if unknown_num == self._label:
                    res.append(Integer(1))
                else:
                    res.append(Integer(0))
            elif self.do_exactly_eq(unknown_num):
                res.append(Integer(1))
            else:
                res.append(Integer(0))

        return res

    def contain_unknown_num(
        self, _circular_refs: Optional[set[NewTypes]] = None
    ) -> bool:
        """
        Returns True if the object contains an unknown number,
        otherwise False.
        """
        return True

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
            _circular_refs=_circular_refs,
        )
        if to_return:
            return

        if not isinstance(values, dict):
            raise TypeError(invalid_type("values", values, expected=dict))

        for key, value in values.items():
            if not isinstance(key, (str, NewTypes)):
                raise TypeError(
                    invalid_type("key", key, expected=Union[str, NewTypes])
                )
            if not isinstance(value, Optional[CalculationSupportsTypes]):
                raise TypeError(
                    invalid_type(
                        "value",
                        value,
                        expected=Optional[CalculationSupportsTypes],
                    )
                )

            # Don't use `self._value = value`
            if isinstance(key, str):
                if key == self._label:
                    self.value = value
            else:  # isinstance(key, NewTypes)
                if self.do_exactly_eq(key):
                    self.value = value

    # ====================
    # represent
    # ====================

    # def __str__(self) -> str:
    #     return self.to_string()

    def to_string(
        self,
        *,
        unknown_num_show_mode: Optional[
            UnknownShowMode
        ] = UnknownShowMode.DEFAULT,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> str:
        """
        Represents the UnknownNum object in string format.

        If the value of the unknown number is not None,
        and the `unknown_num_show_mode` is `UnknownShowMode.VALUE` or
        `UnknownShowMode.DEFAULT`, the `to_string()` method of the value
        will be called.
        Otherwise, the string representation will be the `string`
        property of the unknown number (or the `unknown_num_show_mode`).

        Args:
            unknown_num_show_mode (Optional[UnknownShowMode], optional):
                The mode to show the unknown number.
                Defaults to UnknownShowMode.DEFAULT.
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            str: The string representation of the unknown number.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "str()",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_REPR_CR_DEPTH",
        )
        if to_return:
            return "..."

        if unknown_num_show_mode is None:
            unknown_num_show_mode = UnknownShowMode.DEFAULT

        if not isinstance(unknown_num_show_mode, UnknownShowMode):
            raise TypeError(
                invalid_type(
                    "unknown_num_show_mode",
                    unknown_num_show_mode,
                    expected=UnknownShowMode,
                )
            )

        match unknown_num_show_mode:
            case UnknownShowMode.LABEL:
                return self.o_label
            case UnknownShowMode.STRING:
                return self.o_string
            case UnknownShowMode.LATEX:
                return self._latex
            case UnknownShowMode.VALUE:
                if self._value is not None:
                    return self._value.to_string(
                        unknown_num_show_mode=unknown_num_show_mode,
                        _circular_refs=_circular_refs,
                    )
                return self.o_string

    # def __repr__(self) -> str:
    #     return self.do_repr()

    def do_repr(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        _force_do: bool = False,
    ) -> str:
        """
        A helper function to represent a UnknownNum object.
        """

        if not _force_do:
            assert self._force_do_hit_count == 0
            to_return, _circular_refs = self._circular_reference_dict(
                "repr()",
                _circular_refs=_circular_refs,
                constant_get_key="MAX_REPR_CR_DEPTH",
            )
            if to_return:
                return "..."

        elif self._force_do_helper():
            return "..."

        if self._value is None:
            return (
                f"UnknownNum(label='{self._label}', value=None, "
                f"string='{self._string}', latex='{self._latex}')"
            )

        value = self._value.do_repr(
            _circular_refs=_circular_refs, _force_do=_force_do
        )
        self._value._force_do_hit_count = 0

        return (
            f"UnknownNum(label='{self._label}', value={value}, "
            f"string='{self._string}', latex='{self._latex}')"
        )

    def to_latex(
        self,
        *,
        unknown_num_show_mode: Any = None,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Represents the UnknownNum object in LaTeX format.

        If the value of the unknown number is not None,
        and the `unknown_num_show_mode` is `UnknownShowMode.VALUE` or
        `UnknownShowMode.DEFAULT`, the `to_latex()` method of the value
        will be called.
        Otherwise, the latex representation will be the `latex`
        property of the unknown number (or the `unknown_num_show_mode`).

        Args:
            unknown_num_show_mode (Any, optional):
                The mode to show the unknown number.
                Defaults to None.
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            str: The latex representation of the unknown number.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            _circular_refs=_circular_refs,
            constant_get_key="MAX_REPR_CR_DEPTH",
        )
        if to_return:
            return "..."

        if unknown_num_show_mode is None:
            unknown_num_show_mode = UnknownShowMode.DEFAULT

        if not isinstance(unknown_num_show_mode, UnknownShowMode):
            raise TypeError(
                invalid_type(
                    "unknown_num_show_mode",
                    unknown_num_show_mode,
                    expected=UnknownShowMode,
                )
            )

        match unknown_num_show_mode:
            case UnknownShowMode.LABEL:
                return self.o_label
            case UnknownShowMode.STRING:
                return self.o_string
            case UnknownShowMode.LATEX:
                return self._latex
            case UnknownShowMode.VALUE:
                if self._value is not None:
                    return self._value.to_latex(
                        unknown_num_show_mode=unknown_num_show_mode,
                        _circular_refs=_circular_refs,
                        **kwargs,
                    )
                return self._latex

    # ====================
    # calculate
    # ====================

    # def __int__(self) -> int:
    #     return int(float(self))

    def do_float(
        self,
        *,
        unknown_num_default: Number = 1.0,
        unknown_num_use_default: bool = False,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        _force_do: bool = False,
        **kwargs: Any,
    ) -> float:
        """
        A helper function to convert the UnknownNum object to float.
        """

        if not _force_do:
            assert self._force_do_hit_count == 0
            to_return, _circular_refs = self._circular_reference_dict(
                "float()",
                _circular_refs=_circular_refs,
                constant_get_key="MAX_CALCULATION_CR_DEPTH",
            )
            if to_return:
                return 1.0

        elif self._force_do_helper():
            return 1.0

        if self._value is None:
            if unknown_num_use_default or _force_do:
                return float(unknown_num_default)

            raise ValueError(
                f"Unknown number {self._label} has no value, "
                "and `unknown_num_use_default` is False. "
                "So it cannot be converted to float."
            )

        ret = self._value.do_float(
            _circular_refs=_circular_refs, _force_do=_force_do, **kwargs
        )
        self._value._force_do_hit_count = 0

        return ret

    def do_abs(
        self, *, _circular_refs: Optional[dict[NewTypes, int]] = None
    ) -> "UnknownNum":
        """
        A helper function to calculate the absolute value
        of a UnknownNum object.
        """

        if self._value is None:
            return UnknownNum(
                label=self._label,
                value=None,
                string=self._string,
                latex=self._latex,
            )

        to_return, _circular_refs = self._circular_reference_dict(
            "abs()",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return UnknownNum(
                label=self._label,
                value=self._value,
                string=self._string,
                latex=self._latex,
            )

        return UnknownNum(
            label=self._label,
            value=self._value.do_abs(_circular_refs=_circular_refs),
            string=self._string,
            latex=self._latex,
        )

    def do_add(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to add a UnknownNum object and
        an anothor object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[UnknownNum, Multinomial]: The result of adding the
                two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "+",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return UnknownNum(
                label=self._label,
                value=self._value,
                string=self._string,
                latex=self._latex,
            )

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when adding a UnknownNum object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .multinomial import Multinomial

        return Multinomial((self, other), simplify=False).copy(
            try_deep_copy=True
        )

    def do_mul(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to multiply a UnknownNum object and
        an another object that is supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[UnknownNum, Power, Monomial]:
                The result of multiplying the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "*",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return UnknownNum(
                label=self._label,
                value=self._value,
                string=self._string,
                latex=self._latex,
            )

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when multiplying a UnknownNum object",
                    expected=CalculationSupportsTypes,
                )
            )

        # # to make `x * x` -> `x ** 2`
        # if self.do_exactly_eq(other):
        #     from .power import Power

        #     return Power(self, 2, simplify=False).copy(try_deep_copy=True)

        from .monomial import Monomial

        return Monomial((self, other), simplify=False).copy(try_deep_copy=True)

    def do_truediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to divide a UnknownNum object and
        an another object that is supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[UnknownNum, Division]:
                The result of dividing the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "/",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return UnknownNum(
                label=self._label,
                value=self._value,
                string=self._string,
                latex=self._latex,
            )

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when dividing a UnknownNum object",
                    expected=CalculationSupportsTypes,
                )
            )

        if other.value_eq(1) if isinstance(other, NewTypes) else other == 1:
            return self.copy(try_deep_copy=True)

        from .fraction import Fraction

        return Fraction(self, other, simplify=False).copy(try_deep_copy=True)

    def do_rtruediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to divide an another object that is
        supported by calculation and a UnknownNum object.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[UnknownNum, Division]:
                The result of dividing the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "/",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return UnknownNum(
                label=self._label,
                value=self._value,
                string=self._string,
                latex=self._latex,
            )

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when dividing a UnknownNum object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .fraction import Fraction

        return Fraction(other, self, simplify=False).copy(try_deep_copy=True)

    def __pow__(
        self,
        other: CalculationSupportsTypes,
    ) -> NewTypes:
        """
        Raises the UnknownNum object to the power of the other object.

        Args:
            other (CalculationSupportsTypes): The exponent to which
                the UnknownNum object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Power: The result of raising the UnknownNum object to
                the power of the other object.
        """

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when raising a UnknownNum object to the power",
                    expected=CalculationSupportsTypes,
                )
            )

        from .power import Power

        return Power(self, other, simplify=False).copy(try_deep_copy=True)

    def __rpow__(
        self,
        other: CalculationSupportsTypes,
    ) -> NewTypes:
        """
        Raises the other object to the power of the UnknownNum object.

        Args:
            other (CalculationSupportsTypes): The base to which
                the UnknownNum object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Power: The result of raising the other object to
                the power of the UnknownNum object.
        """

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when raising an object to the "
                    "power of a UnknownNum object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .power import Power

        return Power(other, self, simplify=False).copy(try_deep_copy=True)

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
        Checks if the UnknownNum object is exactly equal to the other object,
        which means they have the same label only.
        The value, string, and latex properties are not considered.

        Args:
            other (object): The object to compare with the UnknownNum object.
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            bool: True if the UnknownNum objects' labels are the same,
                False otherwise.
        """

        if not isinstance(other, UnknownNum):
            return False

        to_return, _circular_refs = self._circular_reference_dict(
            "==",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_COMPARISON_CR_DEPTH",
        )
        if to_return:
            return True

        return self._label == other._label

    def __hash__(self) -> int:
        return self._get_hash(
            f"{self.__class__.__name__}:{self._label}-{self._value}"
            f"-{self._string}-{self._latex}"
        )
