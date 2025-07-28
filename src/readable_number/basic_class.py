"""
Basic Class

The definition of basic class of the child classes.
"""

from abc import ABC, ABCMeta, abstractmethod
from hashlib import sha256
from itertools import tee
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Optional,
    Sequence,
    Union,
)

from . import constants
from ._error_helper import (
    assert_fail,
    invalid_type,
    invalid_value,
    not_implemented,
)
from .viewer import Viewer

# pylint: disable=import-outside-toplevel, protected-access
# pylint: disable=attribute-defined-outside-init


CalculationSupportsTypes = Union[int, float, "BasicClass"]
Number = Union[int, float]


class IsinstanceChecker(ABCMeta):
    """
    A metaclass to help check the instance of the child classes.
    """

    def __instancecheck__(cls, instance: Any) -> bool:
        from .integer import Integer

        if isinstance(instance, int) and issubclass(cls, (int, Integer)):
            return True

        if not issubclass(type(instance), BasicClass):
            return False

        if (
            issubclass(type(instance), Container)
            and not issubclass(cls, Container)
            # NOTE: All the child classes of `Container` should make sure that
            # the instanced objects have a `_items` attribute, which is a list
            # of the items in the container.
            and len(instance._items) == 1  # type: ignore
        ):
            return isinstance(instance._items[0], cls)  # type: ignore

        return super().__instancecheck__(instance)


class BasicClass(ABC, metaclass=IsinstanceChecker):
    """
    Basic class of the `Fraction` class, `Power` class and so on.
    """

    # ====================
    # initialization
    # ====================

    @abstractmethod
    def from_auto(self, item: Any, *args, **kwargs) -> "BasicClass":
        """
        Initializes the object from an object.
        """

    # ====================
    # public funcions
    # ====================

    @abstractmethod
    def get_factors(
        self, _circular_refs: Optional[set["BasicClass"]] = None
    ) -> Generator[Any, None, None]:
        """
        Returns a generator of ALL the factors of the object.
        Note:
            The result should not be empty.
            If negative, -1 is included in the generator.
            1 should be included in the generator.
            0 is not included in the generator.
        """

    @abstractmethod
    def simplify(
        self, _circular_refs: Optional[set["BasicClass"]] = None
    ) -> None:
        """
        Simplifies the object.
        """

    @abstractmethod
    def simplify_without_change(self) -> "BasicClass":
        """
        Simplifies the object without changing itself.
        """

    def get_value(
        self,
        *,
        unknown_num_default: Number = 1,
        unknown_num_use_default: bool = False,
    ) -> float:
        """
        Returns the value of the object.
        """
        return self.do_float(
            unknown_num_default=unknown_num_default,
            unknown_num_use_default=unknown_num_use_default,
        )

    @abstractmethod
    def copy(
        self,
        *,
        copy_unknown_num: bool = False,
        try_deep_copy: bool = False,
        force_deep_copy: bool = False,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Returns a copy of the object.
        """

    def __copy__(self) -> "BasicClass":
        """
        Returns a shallow copy of the object.
        """
        return self.copy(try_deep_copy=False, force_deep_copy=False)

    def __deepcopy__(self, memo: dict) -> "BasicClass":
        """
        Returns a deep copy of the object.
        """
        return self.copy(try_deep_copy=True, force_deep_copy=True)

    # ====================
    # Supports for UnknownNum
    # ====================

    @abstractmethod
    def get_unknowns(
        self,
        _circular_refs: Optional[set["BasicClass"]] = None,
    ) -> set["BasicClass"]:
        """
        Returns the UnknownNum objects in the container.
        """

    @abstractmethod
    def get_coefficient_of_unknowns(
        self,
        unknown_nums: Sequence[Union["BasicClass", str]],
        _do_simplify: bool = True,
        _circular_refs: Optional[set["BasicClass"]] = None,
    ) -> list["BasicClass"]:
        """
        Returns the coefficients of the UnknownNum objects in the container.
        If the object does not contain the UnknownNum object,
        the coefficient will be 0.
        NOTE: Now only support simple equation.
        """

    @abstractmethod
    def contain_unknown_num(
        self, _circular_refs: Optional[set["BasicClass"]] = None
    ) -> bool:
        """
        Returns True if the object contains an unknown number,
        otherwise False.
        """

    @abstractmethod
    def set_values(
        self,
        values: Optional[
            dict[Union["BasicClass", str], Optional[CalculationSupportsTypes]]
        ] = None,
        _circular_refs: Optional[set["BasicClass"]] = None,
    ) -> None:
        """
        Sets the values of the Unknown object.
        """

    # ====================
    # represent
    # ====================

    def __str__(self) -> str:
        """
        Returns a string representation of the object,
        for people to read and understand.
        """
        return self.to_string()

    @abstractmethod
    def to_string(
        self,
        *,
        unknown_num_show_mode: Any = None,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> str:
        """
        Returns a string representation of the object,
        which shows the form of the object in math.
        """

    def __repr__(self) -> str:
        """
        Returns a string representation of the object,
        which can be used to recreate the object.
        """
        return self.do_repr()

    @abstractmethod
    def do_repr(
        self,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
        _force_do: bool = False,
    ) -> str:
        """
        Returns a string representation of the object,
        which can be used to recreate the object.
        """

    def to_latex(
        self,
        *,
        unknown_num_show_mode: Any = None,
        fraction_manual_mode: Any = None,
        fraction_auto_mode: bool = False,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> str:
        """
        Returns a string representation of the object,
        which shows the form of the object in latex.
        """
        raise NotImplementedError(
            not_implemented(
                "to_latex",
                self,
            )
        )

    # ====================
    # calculate
    # ====================

    def __int__(self) -> int:
        """
        Converts the object to an integer.
        """
        return self.do_int()

    def do_int(
        self, *, _circular_refs: Optional[dict["BasicClass", int]] = None
    ) -> int:
        """
        Converts the object to an integer.
        """
        return int(self.do_float())

    def __float__(self) -> float:
        """
        Converts the object to a float.
        """
        return self.do_float()

    @abstractmethod
    def do_float(
        self,
        *,
        unknown_num_default: Number = 1.0,
        unknown_num_use_default: bool = False,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
        _force_do: bool = False,
    ) -> float:
        """
        Converts the object to a float.
        """

    def __abs__(self) -> "BasicClass":
        """
        Returns the absolute value of the object.
        """
        return self.do_abs()

    @abstractmethod
    def do_abs(
        self, *, _circular_refs: Optional[dict["BasicClass", int]] = None
    ) -> "BasicClass":
        """
        Returns the absolute value of the object.
        """

    def __add__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Adds two objects together.
        """
        return self.do_add(other)

    @abstractmethod
    def do_add(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Adds two objects together.
        """

    def __radd__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Adds two objects together,
        when the first object is not a BasicClass object.
        """
        return self.do_add(other)

    def do_radd(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Adds two objects together,
        when the first object is not a BasicClass object.
        """
        return self.do_add(other)

    def __neg__(self) -> "BasicClass":
        """
        Negates the object.
        """
        return -1 * self

    def do_neg(
        self, *, _circular_refs: Optional[dict["BasicClass", int]] = None
    ) -> "BasicClass":
        """
        Negates the object.
        """
        return -1 * self

    def __sub__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Subtracts two objects.
        """
        return self.do_add(-other)

    def do_sub(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Subtracts two objects.
        """
        return self.do_add(-other)

    def __rsub__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Subtracts two objects,
        when the first object is not a BasicClass object.
        """
        return (-self).do_add(other)

    def do_rsub(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Subtracts two objects,
        when the first object is not a BasicClass object.
        """
        return (-self).do_add(other)

    def __mul__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Multiplies two objects.
        """
        return self.do_mul(other)

    @abstractmethod
    def do_mul(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Multiplies two objects.
        """

    def __rmul__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Multiplies two objects,
        when the first object is not a BasicClass object.
        """
        return self.do_mul(other)

    def do_rmul(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Multiplies two objects,
        when the first object is not a BasicClass object.
        """
        return self.do_mul(other)

    def __truediv__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Divides two objects.
        """
        return self.do_truediv(other)

    @abstractmethod
    def do_truediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Divides two objects.
        """

    def __rtruediv__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Divides two objects,
        when the first object is not a BasicClass object.
        """
        return self.do_rtruediv(other)

    @abstractmethod
    def do_rtruediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Divides two objects,
        when the first object is not a BasicClass object.
        """

    def __floordiv__(self, other: CalculationSupportsTypes) -> int:
        """
        Divides two objects and returns the integer part of the result.
        """
        return int(self.do_truediv(other))

    def do_floordiv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> int:
        """
        Divides two objects and returns the integer part of the result.
        """
        return int(self.do_truediv(other))

    def __rfloordiv__(self, other: CalculationSupportsTypes) -> int:
        """
        Divides two objects and returns the integer part of the result,
        when the first object is not a BasicClass object.
        """
        return int(self.do_rtruediv(other))

    def do_rfloordiv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> int:
        """
        Divides two objects and returns the integer part of the result,
        when the first object is not a BasicClass object.
        """
        return int(self.do_rtruediv(other))

    def __mod__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Returns the remainder of the division of the object by another object.
        """
        return self.do_sub(self.do_floordiv(other) * other)

    def do_mod(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Returns the remainder of the division of the object by another object.
        """
        return self.do_sub(self.do_floordiv(other) * other)

    def __rmod__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Returns the remainder of the division of another object by the object,
        when the first object is not a BasicClass object.
        """
        return (self.do_rfloordiv(other) * self).do_rsub(other)

    def do_rmod(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> "BasicClass":
        """
        Returns the remainder of the division of another object by the object,
        when the first object is not a BasicClass object.
        """
        return (self.do_rfloordiv(other) * self).do_rsub(other)

    @abstractmethod
    def __pow__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Raises the object to the power of another object.
        """

    @abstractmethod
    def __rpow__(self, other: CalculationSupportsTypes) -> "BasicClass":
        """
        Raises another object to the power of the object,
        when the first object is not a BasicClass object.
        """

    # ====================
    # check
    # ====================

    def __eq__(self, other: object) -> bool:
        """
        Checks if two objects are equal.
        Please use `value_eq` instead.
        Because `__eq__` will be used when doing hash(obj),
        so we need do `do_float(_force_do=True)`,
        which will ignore the unknown number.
        NOTE: Will use `value_eq` and set `use_unknown_num_value` to True.
        """

        if not isinstance(other, (int, float, BasicClass)):
            return False

        if isinstance(other, BasicClass):
            other_float = other.do_float(
                unknown_num_use_default=True, _force_do=True
            )
            other._force_do_hit_count = 0
        else:
            other_float = float(other)

        self_float = self.do_float(
            unknown_num_use_default=True, _force_do=True
        )
        self._force_do_hit_count = 0

        return self_float == other_float

    def value_eq(
        self, other: object, *, use_unknown_num_value: bool = False
    ) -> bool:
        """
        Checks if two objects' values are equal.
        NOTE: if not isinstance(other, (int, float, BasicClass)):
                  return False
        """

        if not isinstance(other, (int, float, BasicClass)):
            return False

        if use_unknown_num_value:
            other_float = (
                other.do_float(unknown_num_use_default=True)
                if isinstance(other, BasicClass)
                else float(other)
            )
            return self.do_float(unknown_num_use_default=True) == other_float

        contains_unknown_num = self.contain_unknown_num() or (
            isinstance(other, BasicClass) and other.contain_unknown_num()
        )
        return not contains_unknown_num and float(self) == float(other)

    @abstractmethod
    def do_exactly_eq(
        self,
        other: object,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
    ) -> bool:
        """
        Checks if two objects are exactly equal,
        which means they have the same type
        and all the relevant attributes are the same.

        NOTE: A `UnknownNum` object is True if it's label is the same
            as the other object.
        """

    def __ne__(self, other: object) -> bool:
        """
        Checks if two objects are not equal.
        """
        return not self.__eq__(other)

    def __lt__(
        self,
        other: CalculationSupportsTypes,
    ) -> bool:
        """
        Checks if the Fraction object is less than the other object.

        Args:
            other (CalculationSupportsTypes)

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            bool: True if the Fraction object is less than the other object,
                False otherwise.
        """

        if not isinstance(other, Union[int, float, BasicClass]):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    expected=Union[int, float, BasicClass],
                )
            )

        return (self - other).do_float() < 0

    def __le__(self, other: CalculationSupportsTypes) -> bool:
        """
        Checks if the object is less than or equal to another object.
        """
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other: CalculationSupportsTypes) -> bool:
        """
        Checks if the object is greater than another object.
        """
        return not self.__le__(other)

    def __ge__(self, other: CalculationSupportsTypes) -> bool:
        """
        Checks if the object is greater than or equal to another object.
        """
        return not self.__lt__(other)

    def __bool__(self) -> bool:
        """
        Checks if the object is True or False.
        """
        return not self.value_eq(0)

    def __hash__(self) -> int:
        """
        Returns a hash value of the object.
        """
        value = self._get_hash()
        self._force_do_hit_count = 0
        return value

    def _get_hash(self, hash_str: str = "") -> int:
        """
        A helper function to get the hash value of the object.

        Args:
            hash_str (str, optional): The string to be hashed.

        Returns:
            int: The hash value of the object.
        """

        if hash_str == "":
            hash_str = (
                f"{self.__class__.__name__}:{self.do_repr(_force_do=True)}"
            )
            self._force_do_hit_count = 0

        hash_int = int(sha256(hash_str.encode()).hexdigest(), 16)

        if self._constants("HASH_USE_ALL", "bool"):
            return hash_int

        # We just want to take 16 digits, but make all the digits used
        # at the same time.
        hash_last_16 = hash_int % (10**16)
        hash_32_16 = int(hash_int // (10**16) / (10**16)) % (10**16)
        hash_res = (hash_last_16 ^ hash_32_16) % (10**16)
        return hash_res

    # ====================
    # other
    # ====================

    def _circular_reference_set(
        self,
        more_msg: str,
        *,
        _circular_refs: Optional[set["BasicClass"]] = None,
    ) -> tuple[bool, set["BasicClass"]]:
        """
        A helper function to check whether the current object
        is in the circular reference set.
        The `_circular_refs` parameter will be copied.

        Returns:
            tuple[bool, set["BasicClass"]]: A tuple of a bool value
                indicating whether the current object is in the
                circular reference set, and the updated circular
                reference set.
        """

        if _circular_refs is None:
            _circular_refs = set()
        if not isinstance(_circular_refs, set):
            raise TypeError(
                invalid_type(
                    "_circular_refs",
                    _circular_refs,
                    more_msg=more_msg,
                    expected=set,
                )
            )

        _circular_refs = _circular_refs.copy()
        if self in _circular_refs:
            return True, _circular_refs
        _circular_refs.add(self)
        return False, _circular_refs

    def _circular_reference_dict(
        self,
        more_msg: str,
        *,
        _circular_refs: Optional[dict["BasicClass", int]] = None,
        constant_get_key: str,
    ) -> tuple[bool, dict["BasicClass", int]]:
        """
        A helper function to check whether the current object
        is in the circular reference dict.
        The `_circular_refs` parameter will be copied.

        Returns:
            tuple[bool, dict["BasicClass", int]]: A tuple of a bool value
                indicating whether the current object is in the
                circular reference dict, and the updated circular
                reference dict.
        """

        if _circular_refs is None:
            _circular_refs = {}
        if not isinstance(_circular_refs, dict):
            raise TypeError(
                invalid_type(
                    "_circular_refs",
                    _circular_refs,
                    more_msg=more_msg,
                    expected=dict,
                )
            )

        _circular_refs = _circular_refs.copy()
        if _circular_refs.get(self, 0) > self._constants(
            constant_get_key, ">0"
        ):
            return True, _circular_refs
        _circular_refs[self] = _circular_refs.get(self, 0) + 1
        return False, _circular_refs

    @staticmethod
    def _get_int_factors(num: int) -> Generator["BasicClass", None, None]:
        """
        A helper function to get the factors of an integer.

        Args:
            num (int)

        Yields:
            Generator[Integer, None, None]: The factors of the integer.
        """

        assert isinstance(num, int), assert_fail(
            invalid_type, "num", num, expected=int
        )

        from .integer import Integer

        if num == 0:
            return
        if num < 0:
            yield Integer(-1)
            num = -num

        for i in range(1, num + 1):
            if num % i == 0:
                yield Integer(i)

    @classmethod
    def _constants(
        cls,
        key: str,
        valid_function: Optional[Union[Callable[[Any], bool], str]] = None,
    ) -> Any:
        """
        A helper function to get the constants of the class.

        Args:
            key (str): The key of the constant.
            valid_function
            (Optional[Union[Callable[[Any], bool], str]], optional):
                A function to check whether the value is valid,
                or a string representing the name of a function to check:
                 - "int": The value should be an integer.
                 - ">`num`": The value should be greater than `num`.
                 - "<`num`": The value should be less than `num`.
                 - "==`num`": The value should be equal to `num`.
                 - "bool": The value should be a bool.
                 - None: No check is performed.

        Returns:
            Any: The value of the constant.

        Raises:
            ValueError: When the value is not valid.
        """

        try:
            value = getattr(constants, key)
        except AttributeError as e:
            raise ValueError(f"The constant {key} is not defined.") from e

        if valid_function is None:
            return value

        if callable(valid_function):
            if not valid_function(value):
                raise ValueError(f"The value of {key} is not valid.")
            return value

        if not isinstance(valid_function, str):
            raise TypeError(
                invalid_type(
                    "valid_function",
                    valid_function,
                    expected=Union[Callable[[Any], bool], str],
                )
            )

        if valid_function.startswith(">"):
            num = int(valid_function[1:])
            if value <= num:
                raise ValueError(f"The value of {key} is not valid.")
            return value
        elif valid_function.startswith("<"):
            num = int(valid_function[1:])
            if value >= num:
                raise ValueError(f"The value of {key} is not valid.")
            return value
        elif valid_function.startswith("=="):
            num = int(valid_function[2:-1])
            if value != num:
                raise ValueError(f"The value of {key} is not valid.")
            return value
        elif valid_function == "bool":
            try:
                return bool(value)
            except ValueError as e:
                raise ValueError(f"The value of {key} is not valid.") from e
        elif valid_function == "int":
            try:
                return int(value)
            except ValueError as e:
                raise ValueError(f"The value of {key} is not valid.") from e
        else:
            raise ValueError(
                f"The valid_function {valid_function} is not valid."
            )

    @staticmethod
    def get_intersection_by_value(
        items1: Iterable["BasicClass"], items2: Iterable["BasicClass"]
    ) -> Generator["BasicClass", None, None]:
        """
        Returns the intersection of two sets,
        but use `==`(float) when comparing.
        Will yields the item in items1.
        """

        if not isinstance(items1, Iterable):
            raise TypeError(
                invalid_type(
                    "items1",
                    items1,
                    expected=Iterable,
                )
            )
        if not isinstance(items2, Iterable):
            raise TypeError(
                invalid_type(
                    "items2",
                    items2,
                    expected=Iterable,
                )
            )

        for item1 in items1:
            items2_it, items2 = tee(items2)
            for item2 in items2_it:
                if not isinstance(item1, BasicClass):
                    raise ValueError(
                        invalid_value(
                            "item1",
                            item1,
                            expected="an instance of BasicClass",
                        )
                    )
                if not isinstance(item2, BasicClass):
                    raise ValueError(
                        invalid_value(
                            "item2",
                            item2,
                            expected="an instance of BasicClass",
                        )
                    )
                if float(item1) == float(item2):
                    yield item1

    def _optimize(self) -> "BasicClass":
        """
        Optimizes self, and return the result.
        """

        from .fraction import Fraction
        from .integer import Integer
        from .power import Power
        from .unknown_num import UnknownNum

        match self:
            case Fraction(numerator, denominator):
                assert not denominator.value_eq(0)
                if numerator.value_eq(0):
                    return Integer(0)
                if numerator.value_eq(denominator):
                    return Integer(1)
                if numerator.value_eq(-denominator):
                    return Integer(-1)
                if denominator.value_eq(1):
                    return numerator
                if denominator.value_eq(-1):
                    return -numerator
                return self

            case Power(base, exponent):
                if base.value_eq(0):
                    return Integer(0)
                if exponent.value_eq(0):
                    return Integer(1)
                if exponent.value_eq(1):
                    return base
                return self

            case UnknownNum(label, value, string, latex):
                if value is not None and not isinstance(value, UnknownNum):
                    return UnknownNum(
                        label=label,
                        value=value._optimize(),
                        string=string,
                        latex=latex,
                    )
                return self

            case other if isinstance(other, Container):
                # pylint: disable=no-member
                # NOTE: All the child classes of `Container` should make
                # sure that the instanced objects have a `_items`
                # attribute, which is a list of the items in the container.
                if len(self._items) == 0:  # type: ignore
                    return self._items[0]  # type: ignore
                return self
                # pylint: enable=no-member

            case other:
                return self


class Container(BasicClass, ABC, metaclass=IsinstanceChecker):
    """
    A base class for containers.

    NOTE: All the child classes of `Container` should make sure that
        the instanced objects have a `_items` attribute, which is a list
        of the items in the container.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the container.
        """

    @abstractmethod
    def __getitem__(self, key: int) -> Viewer:
        """
        Returns the item at the given index.
        """

    @abstractmethod
    def __setitem__(self, key: Union[int, slice], value: Any) -> None:
        """
        Sets the item at the given index to the given value.
        """

    @abstractmethod
    def __delitem__(self, key: Union[int, slice]) -> None:
        """
        Deletes the item at the given index.
        """

    @abstractmethod
    def __iter__(self) -> Generator[Viewer, None, None]:
        """
        Returns a generator of the items in the container.
        """

    @abstractmethod
    def __contains__(self, item: Any) -> bool:
        """
        Checks if the container contains the given item.
        """

    @abstractmethod
    def append(self, item: Any) -> None:
        """
        Appends the given item to the end of the container.
        """

    @abstractmethod
    def extend(self, items: list) -> None:
        """
        Extends the container with the given items.
        """

    @abstractmethod
    def remove(self, item: Any) -> None:
        """
        Removes the first occurrence of the given item from the container.
        """

    @abstractmethod
    def clear(self) -> None:
        """
        Removes all items from the container.
        """

    @abstractmethod
    def count(self, item: Any) -> int:
        """
        Returns the number of occurrences of the given item in the container.
        """
