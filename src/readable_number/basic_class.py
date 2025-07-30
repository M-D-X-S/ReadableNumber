"""
Basic Class

The definition of basic class of the child classes.
"""

from abc import ABC, ABCMeta, abstractmethod
from enum import StrEnum, auto
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
        instead: str = "",
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

        more_msg = (
            "do not pass `_circular_refs` in wrong type "
            "when calling this method"
            + (f", use `{instead}` instead" if instead else "")
        )

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
        instead: str = "",
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

        more_msg = (
            "do not pass `_circular_refs` in wrong type "
            "when calling this method"
            + (f", use `{instead}` instead" if instead else "")
        )

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
        if valid_function.startswith("<"):
            num = int(valid_function[1:])
            if value >= num:
                raise ValueError(f"The value of {key} is not valid.")
            return value
        if valid_function.startswith("=="):
            num = int(valid_function[2:-1])
            if value != num:
                raise ValueError(f"The value of {key} is not valid.")
            return value
        if valid_function == "bool":
            try:
                return bool(value)
            except ValueError as e:
                raise ValueError(f"The value of {key} is not valid.") from e
        if valid_function == "int":
            try:
                return int(value)
            except ValueError as e:
                raise ValueError(f"The value of {key} is not valid.") from e

        raise ValueError(f"The valid_function {valid_function} is not valid.")

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

    def _force_do_helper(self) -> bool:
        """
        A helper function to force do.
        Will set the `_force_do_hit_count`.

        Returns:
            bool: Whether to return.
        """

        if self._force_do_hit_count > self._constants(
            "MAX_FORCE_DO_HIT_COUNT", "<13"
        ):
            self._force_do_hit_count = 0
            return True

        self._force_do_hit_count += 1
        return False


NewTypes = BasicClass
ItemsSupportsTypes = Iterable[CalculationSupportsTypes]
PropertyTypes = list[NewTypes]


class OrderMode(StrEnum):
    """
    The order mode of the monomial.
    """

    # NOTE: The order of the following enums is the default order.

    INTEGER = auto()
    MULTINOMIAL = auto()
    POWER = auto()
    FRACTION = auto()
    MONOMIAL = auto()
    UNKNOWN_NUM = auto()


class Container(BasicClass, ABC, metaclass=IsinstanceChecker):
    """
    A base class for containers.

    NOTE: All the child classes of `Container` should make sure that
        the instanced objects have a `_items` attribute, which is a list
        of the items in the container.
    """

    # Whether to simplify the Multinomial after calculation.
    simplify_after_calculation = False

    # Whether the separator is between the space.
    # If True, the separator is " + " or " * ",
    # if False, the separator is "+" or "*".
    use_space_separator = False

    # Whether to optimize the recursive representation of the multinomial.
    # For example, if a power is represented as `(...)+(...)+(...)`,
    # it will be optimized to `...` if this attribute is True.
    # Also, if `0` is in the string representation,
    # it will be skipped if this attribute is True.
    # e.g. `0+0+0` -> `0`
    optimize_recursive_repr = False

    # ====================
    # initialization
    # ====================

    def __init__(
        self,
        items: ItemsSupportsTypes,
        *,
        simplify: bool = True,
        order_mode: Union[list[OrderMode], tuple[OrderMode, ...]] = tuple(
            OrderMode
        ),
    ) -> None:
        """
        The initialization of the Multinomial class.

        The argument `order_mode` is a list or tuple of `OrderMode`.
        It specifies the order of the items in the string
        representation of the Multinomial object.
        For example, by default, the order is
        (Integer, Multinomial, Power, Fraction, Monomial, UnknownNum)
        which means the items will be sorted in the order
        of `Integer`, `Multinomial`, `Power`, `Fraction`,
        `Monomial` and `UnknownNum`.

        Args:
            items (ItemsSupportsTypes): The items of the multinomial.
            simplify (bool, optional):
                Whether to simplify the multinomial after initialization.
                Defaults to True.
            order_mode
            (Union[list[OrderMode], tuple[OrderMode, ...]], optional):
                The order mode of the multinomial.

        Raises:
            FROM `_init_args_handler`:
                TypeError: When the items is not a list or tuple.
                TypeError: When the item in the items
                    is not a CalculationSupportsTypes.
        """

        if items is None:
            raise TypeError(invalid_value("items", items, expected="not None"))

        items_, order_mode = self._init_args_handler(items, order_mode)

        self._items = [item.copy(try_deep_copy=True) for item in items_]
        self._order_mode = order_mode

        self._force_do_hit_count = 0

        self._order_items()

        if simplify:
            self.simplify()

    @property
    def items(self) -> Viewer:
        """
        Returns a viewer of the items of the multinomial.
        """
        return Viewer(self._items)

    @property
    def order_mode(self) -> tuple[OrderMode, ...]:
        """
        Returns the order mode of the multinomial.
        """
        return self._order_mode

    @order_mode.setter
    def order_mode(
        self, value: Union[list[OrderMode], tuple[OrderMode, ...]]
    ) -> None:
        """
        Sets the order mode of the multinomial.
        """
        _, self._order_mode = self._init_args_handler(None, value)

    def _order_items(self) -> None:
        """
        Orders the items of the multinomial.
        """

        from .fraction import Fraction
        from .integer import Integer
        from .monomial import Monomial
        from .multinomial import Multinomial
        from .power import Power
        from .unknown_num import UnknownNum

        mapping = {
            Fraction: OrderMode.FRACTION,
            Integer: OrderMode.INTEGER,
            Monomial: OrderMode.MONOMIAL,
            Multinomial: OrderMode.MULTINOMIAL,
            Power: OrderMode.POWER,
            UnknownNum: OrderMode.UNKNOWN_NUM,
        }

        def get_key(item: NewTypes) -> tuple[int, float]:
            index = self._order_mode.index(mapping[type(item)])
            return (index, item.do_float(unknown_num_use_default=True))

        self._items.sort(key=get_key)

    def _init_args_handler(
        self,
        items: Optional[ItemsSupportsTypes],
        order_mode: Union[list[OrderMode], tuple[OrderMode, ...]],
    ) -> tuple[PropertyTypes, tuple[OrderMode, ...]]:
        """
        A helper function to handle the arguments.

        Args:
            items (Optional[ItemsSupportsTypes])
            order_mode (Union[list[OrderMode], tuple[OrderMode, ...]])

        Raises:
            TypeError: When the items is not a list or tuple.
            TypeError: When the item in the items
                is not a CalculationSupportsTypes.

        Returns:
            tuple[PropertyTypes, tuple[OrderMode, ...]]:
                The items of the multinomial.
        """

        if items is None:
            items = []
        if not isinstance(items, (list, tuple)):
            raise TypeError(
                invalid_type("items", items, expected=(list, tuple))
            )

        from .integer import Integer

        res_items = []
        for item in items:
            if not isinstance(item, Union[int, float, BasicClass]):
                raise TypeError(
                    invalid_type(
                        "items", item, expected=Union[int, float, BasicClass]
                    )
                )
            if isinstance(item, float):
                from .fraction import Fraction

                item = Fraction.from_float(item)
            if isinstance(item, int):
                item = Integer(item)
            res_items.append(item.copy(try_deep_copy=True))

        if len(order_mode) > len(
            OrderMode
        ):  # or len(set(order_mode)) != len(order_mode)
            raise ValueError(
                invalid_value(
                    "order_mode", order_mode, more_msg="contains duplicates"
                )
            )

        res_order = []

        for order in order_mode:
            if not isinstance(order, OrderMode):
                raise TypeError(
                    invalid_type("order_mode", order, expected=OrderMode)
                )
            if order in res_order:
                raise ValueError(
                    invalid_value(
                        "order_mode",
                        order_mode,
                        more_msg="contains duplicates",
                    )
                )
            res_order.append(order)

        for member in OrderMode:
            if member not in res_order:
                res_order.append(member)

        return res_items, tuple(res_order)

    # ====================
    # public methods
    # ====================

    def __len__(self) -> int:
        """
        Returns the length of items in the multinomial.
        """
        return len(self._items)

    def __getitem__(self, index: Union[int, slice]) -> Viewer:
        """
        Returns a viewer of the items of the multinomial.
        """
        return Viewer(self._items[index])

    def __setitem__(
        self,
        index: Union[int, slice],
        value: Union[CalculationSupportsTypes, ItemsSupportsTypes],
    ) -> None:
        """
        Sets the item at the given index to the given value.
        """

        from .integer import Integer

        if isinstance(index, int):
            if not isinstance(value, Union[int, float, BasicClass]):
                raise TypeError(
                    invalid_type(
                        "value", value, expected=Union[int, float, BasicClass]
                    )
                )
            if isinstance(value, float):
                from .fraction import Fraction

                value = Fraction.from_float(value)
            if isinstance(value, int):
                value = Integer(value)
            self._items[index] = value.copy(try_deep_copy=True)

        elif isinstance(index, slice):
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    invalid_type("value", value, expected=(list, tuple))
                )

            res = []
            for item in value:
                if not isinstance(item, Union[int, float, BasicClass]):
                    raise TypeError(
                        invalid_type(
                            "value",
                            item,
                            expected=Union[int, float, BasicClass],
                        )
                    )
                if isinstance(item, float):
                    from .fraction import Fraction

                    item = Fraction.from_float(item)
                if isinstance(item, int):
                    item = Integer(item)
                res.append(item.copy(try_deep_copy=True))

            self._items[index] = res

        else:
            raise TypeError(
                invalid_type("index", index, expected=(int, slice))
            )

    def __delitem__(self, index: Union[int, slice]) -> None:
        """
        Deletes the items of the multinomial.
        """
        del self._items[index]

    def __iter__(self) -> Generator[Viewer, None, None]:
        """
        Returns a generator of the items of the multinomial.
        """
        for item in self._items:
            yield Viewer(item.copy(try_deep_copy=True))

    def __contains__(self, item: Any) -> bool:
        """
        Returns whether the item is in the multinomial.
        """
        return item in self._items

    def append(self, item: CalculationSupportsTypes) -> None:
        """
        Appends the item to the end of the multinomial.
        """

        if not isinstance(item, Union[int, float, BasicClass]):
            raise TypeError(
                invalid_type(
                    "item", item, expected=Union[int, float, BasicClass]
                )
            )

        from .integer import Integer

        if isinstance(item, float):
            from .fraction import Fraction

            item = Fraction.from_float(item)
        if isinstance(item, int):
            item = Integer(item)
        self._items.append(item.copy(try_deep_copy=True))

    def extend(self, items: ItemsSupportsTypes) -> None:
        """
        Extends the multinomial by appending the items from the iterable.
        """

        if not isinstance(items, (list, tuple)):
            raise TypeError(
                invalid_type("items", items, expected=(list, tuple))
            )

        from .integer import Integer

        for item in items:
            if not isinstance(item, Union[int, float, BasicClass]):
                raise TypeError(
                    invalid_type(
                        "items", item, expected=Union[int, float, BasicClass]
                    )
                )
            if isinstance(item, float):
                from .fraction import Fraction

                item = Fraction.from_float(item)
            if isinstance(item, int):
                item = Integer(item)
            self._items.append(item.copy(try_deep_copy=True))

    def remove(self, item: NewTypes) -> None:
        """
        Removes the first occurrence of the item in the multinomial.
        """
        self._items.remove(item)

    def clear(self) -> None:
        """
        Removes all items from the multinomial.
        """
        self._items.clear()

    def count(self, item: NewTypes) -> int:
        """
        Returns the number of occurrences of the item in the multinomial.
        """
        return self._items.count(item)

    def _simplify(
        self,
        _circular_refs: Optional[set[NewTypes]] = None,
        *,
        optimize_mapping: Optional[
            dict[type[NewTypes], Callable[[NewTypes], list[NewTypes]]]
        ] = None,
    ) -> None:
        """
        A helper function to simplify the Container object.

        `optimize_mapping` is a dictionary that maps the type of the object
        to a function that returns the simplified object.
        If it is None, it will use the default mapping:
        {type(self): (lambda _: self._items)}

        Will do `res.extend(optimize_func(item))`
        """

        to_return, _circular_refs = self._circular_reference_set(
            _circular_refs=_circular_refs,
        )
        if to_return:
            return

        if optimize_mapping is None:
            optimize_mapping = {type(self): (lambda _: self._items)}
        if not isinstance(optimize_mapping, dict):
            raise TypeError(
                invalid_type(
                    "optimize_mapping",
                    optimize_mapping,
                    expected=dict,
                )
            )

        # This method is needless to copy the items,
        # because the items are already copied in the constructor.
        # So we can directly use the original items.

        res1: list[NewTypes] = []
        for index, item1 in enumerate(self._items):
            for item2 in self._items[index + 1 :]:
                if isinstance(item1, NewTypes):
                    item1.simplify(_circular_refs)
                if isinstance(item2, NewTypes):
                    item2.simplify(_circular_refs)

                add_res = item1 + item2
                if not isinstance(add_res, type(self)):
                    res1.append(add_res)
                else:
                    res1.extend((item1, item2))

        # optimization
        res2 = []

        for item in res1:
            if type(item) in optimize_mapping:
                res2.extend(optimize_mapping[type(item)](item))
            else:
                res2.append(item)

        self._items = res2
        self._order_items()

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
            _circular_refs=_circular_refs,
        )
        if to_return:
            return set()

        res = set()

        for item in self._items:
            res.update(item.get_unknowns(_circular_refs=_circular_refs))

        return res

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

        from .integer import Integer

        to_return, _circular_refs = self._circular_reference_set(
            _circular_refs=_circular_refs,
        )
        if to_return or not unknown_nums:
            return [Integer(0)] * len(unknown_nums)

        from .unknown_num import UnknownNum

        self._order_items()
        res = []

        for unknown_num in unknown_nums:
            if not isinstance(unknown_num, (UnknownNum, str)):
                raise TypeError(
                    invalid_type(
                        "unknown_num",
                        unknown_num,
                        expected=(UnknownNum, str),
                    )
                )

            buff = []
            for item in self._items:
                item_coef = item.get_coefficient_of_unknowns(
                    [unknown_num], False, _circular_refs=_circular_refs
                )[0]
                if isinstance(item_coef, Integer) and item_coef.value_eq(0):
                    continue

                buff.append(item_coef)

            if buff:
                res.append(type(self)(buff, simplify=_do_simplify))
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

        to_return, _circular_refs = self._circular_reference_set(
            _circular_refs=_circular_refs,
        )
        if to_return:
            return False

        return any(
            item.contain_unknown_num(_circular_refs) for item in self._items
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
            _circular_refs=_circular_refs,
        )
        if to_return:
            return

        for item in self._items:
            item.set_values(values, _circular_refs=_circular_refs)

    def copy(
        self,
        *,
        copy_unknown_num: bool = False,
        try_deep_copy: bool = False,
        force_deep_copy: bool = False,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> "Container":
        """
        Creates a copy of the Container object.
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
            Container: The copy of the Container object.
        """

        if not try_deep_copy:
            return type(self)(tuple(self._items), simplify=False)

        to_return, _circular_refs = self._circular_reference_dict(
            _circular_refs=_circular_refs,
            constant_get_key="MAX_COPY_CR_DEPTH",
        )
        if to_return:
            return self

        # try_deep_copy=True
        from .unknown_num import UnknownNum

        items = []
        for item in self._items:
            if not isinstance(item, UnknownNum) or copy_unknown_num:
                if force_deep_copy:
                    try:
                        # do forced deep copy by not passing _circular_refs
                        item = item.copy(
                            try_deep_copy=True,
                            force_deep_copy=True,
                        )
                    except RecursionError:
                        return self
                else:
                    item = item.copy(
                        try_deep_copy=True,
                        force_deep_copy=False,
                        _circular_refs=_circular_refs,
                    )
            items.append(item)

        return type(self)(items, simplify=False)

    # ====================
    # calculate
    # ====================

    def _do_abs_helper(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> Optional["Container"]:
        """
        A helper function to calculate the absolute value of
        a Container object.

        NOTE: The res will not be copied.

        If `to_return` is True, it will return None,
        then you should return the default value.

        Returns:
            Optional[Container]: The absolute value of the Container object.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "abs()",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return None

        items = [
            item.do_abs(_circular_refs=_circular_refs) for item in self._items
        ]

        return type(self)(items)

    def _do_truediv_helper(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> Optional[NewTypes]:
        """
        A helper function to divide a Container object and
        an anothor object that be supported by calculation.

        NOTE: The res will not be copied.

        If `to_return` is True, it will return None,
        then you should return the default value.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Optional[Fraction]: The result of the division.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "/",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return None

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg=f"when dividing a {type(self).__name__} object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .fraction import Fraction

        return Fraction(self, other, simplify=self.simplify_after_calculation)

    def _do_rtruediv_helper(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> Optional[NewTypes]:
        """
        A helper function to divide an anothor object and
        a Container object that be supported by calculation.

        NOTE: This function depends on the `do_truediv_helper` function.

        If `to_return` is True, it will return None,
        then you should return the default value.

        Returns:
            Optimal[Fraction]: The result of the division.
        """

        from .fraction import Fraction

        res = self._do_truediv_helper(other, _circular_refs=_circular_refs)
        if isinstance(res, Fraction):
            res.self_reciprocal()
            return res

        assert res is None
        return res

    def __pow__(self, other: CalculationSupportsTypes) -> NewTypes:
        """
        Raises a Container object to the power of the other object.

        Args:
            other (CalculationSupportsTypes): The exponent to which
                the Container object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Power: The result of raising the Container object to
                the power of the other object.
        """

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg=f"when raising a {type(self).__name__} object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .power import Power

        return Power(
            self, other, simplify=self.simplify_after_calculation
        ).copy(try_deep_copy=True)

    def __rpow__(self, other: CalculationSupportsTypes) -> NewTypes:
        """
        Raises an object to the power of a Container object.

        Args:
            other (CalculationSupportsTypes): The base to which
                the Container object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Power: The result of raising the other object to
                the power of the Container object.
        """

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg=f"when raising an {type(self).__name__} object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .power import Power

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
        Checks if the Monomial objects are exactly equal to the other object,
        which means they have the same items.

        Args:
            other (object): The object to be compared with.
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            bool: True if the two objects are exactly equal, False otherwise.
        """

        if not isinstance(other, type(self)):
            return False

        to_return, _circular_refs = self._circular_reference_dict(
            _circular_refs=_circular_refs,
            constant_get_key="MAX_COMPARISON_CR_DEPTH",
        )
        if to_return:
            return True

        return self._items == other._items

    def __hash__(self) -> int:
        self._order_items()
        return self._get_hash()
