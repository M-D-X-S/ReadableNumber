"""
Multinomial

The definition of the `Multinomial` class.
"""

from enum import StrEnum, auto
from typing import Any, Generator, Iterable, Optional, Sequence, Union

from ._error_helper import (
    invalid_type,
    invalid_value,
    assert_fail,
)
from ._types import CalculationSupportsTypes, NewTypes
from .basic_class import Container
from .integer import Integer
from .viewer import Viewer


# pylint: disable=import-outside-toplevel, protected-access


ItemsSupportsTypes = Iterable[CalculationSupportsTypes]
PropertyTypes = list[NewTypes]


class OrderMode(StrEnum):
    """
    The order mode of the monomial.
    """

    # NOTE: The order of the following enums is the default order.

    INTEGER = auto()
    MULTIPLICATION = auto()
    POWER = auto()
    FRACTION = auto()
    MONOMIAL = auto()
    UNKNOWN_NUM = auto()


class Multinomial(Container):
    """
    A class to represent a multinomial.
    """

    # Whether to simplify the Multinomial after calculation.
    simplify_after_calculation = False

    # Whether the separator is between the space.
    # If True, the separator is " + ",
    # if False, the separator is "+".
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
        (Integer, Multiplication, Power, Fraction, Monomial, UnknownNum)
        which means the items will be sorted in the order
        of `Integer`, `Multiplication`, `Power`, `Fraction`,
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
        from .monomial import Monomial
        from .power import Power
        from .unknown_num import UnknownNum

        mapping = {
            Fraction: OrderMode.FRACTION,
            Integer: OrderMode.INTEGER,
            Monomial: OrderMode.MONOMIAL,
            Multinomial: OrderMode.MULTIPLICATION,
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

        res_items = []
        for item in items:
            if not isinstance(item, CalculationSupportsTypes):
                raise TypeError(
                    invalid_type(
                        "items", item, expected=CalculationSupportsTypes
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

        if isinstance(index, int):
            if not isinstance(value, CalculationSupportsTypes):
                raise TypeError(
                    invalid_type(
                        "value", value, expected=CalculationSupportsTypes
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
                if not isinstance(item, CalculationSupportsTypes):
                    raise TypeError(
                        invalid_type(
                            "value", item, expected=CalculationSupportsTypes
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

        if not isinstance(item, CalculationSupportsTypes):
            raise TypeError(
                invalid_type("item", item, expected=CalculationSupportsTypes)
            )
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

        for item in items:
            if not isinstance(item, CalculationSupportsTypes):
                raise TypeError(
                    invalid_type(
                        "items", item, expected=CalculationSupportsTypes
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

    def get_factors(
        self, _circular_refs: Optional[set[NewTypes]] = None
    ) -> Generator[NewTypes, None, None]:
        """
        Gets the factors of the multinomial.

        Yields:
            Generator[NewTypes, None, None]:
                A generator of the factors.
        """

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if to_return:
            yield Integer(1)
            return

        res = set()
        for item in self._items:
            res.intersection_update(item.get_factors(_circular_refs))
            if not res:
                yield Integer(1)
                return

        if not res:
            yield Integer(1)
        else:
            yield from res

    def simplify(self, _circular_refs: Optional[set[NewTypes]] = None) -> None:
        """
        Simplifies the multinomial.
        """

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if to_return:
            return

        # This method is needless to copy the items,
        # because the items are already copied in the constructor.
        # So we can directly use the original items.

        res1 = []
        for index, item1 in enumerate(self._items):
            for item2 in self._items[index + 1 :]:
                if isinstance(item1, NewTypes):
                    item1.simplify(_circular_refs)
                if isinstance(item2, NewTypes):
                    item2.simplify(_circular_refs)

                add_res = item1 + item2
                if not isinstance(add_res, Multinomial):
                    res1.append(add_res)
                else:
                    res1.extend((item1, item2))

        # optimization
        res2 = []

        # pylint: disable=unused-variable

        for item in res1:
            match item:
                case Multinomial(items=items) as obj:
                    res2.extend(obj._items)
                case other:
                    res2.append(item)

        # pylint: enable=unused-variable

        self._items = res2
        self._order_items()

    def simplify_without_change(self) -> "Multinomial":
        """
        Returns a new simplified multinomial without changing the original one.
        """

        return Multinomial(
            tuple(self._items), simplify=True, order_mode=self._order_mode
        )

    @classmethod
    def from_auto(
        cls, item: Union[int, "Multinomial"], *args, **kwargs
    ) -> "Multinomial":
        """
        Creates a multinomial from an integer or another Multinomial
        object.
        Will set `simplify` to False by default.
        """

        # pylint: disable=unused-variable

        match item:
            case int(x) | Integer(x):
                return cls.from_int(item, *args, **kwargs)
            case Multinomial(items=items):
                return item.copy(*args, **kwargs)

            case other:
                raise TypeError(
                    invalid_type(
                        "item",
                        item,
                        more_msg="when creating a multinomial from an integer",
                        expected=int,
                    )
                )

        # pylint: enable=unused-variable

    @classmethod
    def from_int(cls, num: int, *args, **kwargs) -> "Multinomial":
        """
        Creates a multinomial from an integer.
        Will set `simplify` to False by default.
        """

        if not isinstance(num, int):
            raise TypeError(
                invalid_type(
                    "num",
                    num,
                    more_msg="when creating a multinomial from an integer",
                    expected=int,
                )
            )

        kwargs["simplify"] = kwargs.get("simplify", False)
        return cls([num], *args, **kwargs)

    # def get_value(self) -> float:
    #     return float(self)

    def copy(
        self,
        *,
        copy_unknown_num: bool = False,
        try_deep_copy: bool = False,
        force_deep_copy: bool = False,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> "Multinomial":
        """
        Creates a copy of the Multinomial object.
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
            Multinomial: The copy of the Multinomial object.
        """

        if not try_deep_copy:
            return Multinomial(tuple(self._items), simplify=False)

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
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

        return Multinomial(items, simplify=False)

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

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
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
                res.append(Multinomial(buff, simplify=_do_simplify))
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
        return any(
            item.contain_unknown_num(_circular_refs=_circular_refs)
            for item in self._items
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

        for item in self._items:
            item.set_values(values, _circular_refs=_circular_refs)

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
        Represents the multinomial in string format.

        NOTE: If the object's items is empty, it will return
        an empty string.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `str()` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_REPR_CR_DEPTH",
        )
        if to_return:
            return "..."

        self._order_items()

        result = []
        has_epsilon = False
        has_0 = False

        for item in self._items:
            item_str = item.to_string(
                _circular_refs=_circular_refs,
                **kwargs,
            )
            if item_str == "":
                continue

            if self.optimize_recursive_repr:
                if not has_epsilon and item_str == "...":
                    has_epsilon = True
                else:
                    continue
                if item_str == "0":
                    if has_0:
                        continue
                    else:
                        has_0 = True

            if item_str == "...":
                item_str = "(...)"

            result.append(item_str)

        if not result:
            return ""

        if len(result) == 1 and has_epsilon:  # result is ["(...)"]
            return "..."

        if self.use_space_separator:
            sep = " + "
        else:
            sep = "+"
        return sep.join(result)

    # def __repr__(self) -> str:
    #     return self.do_repr()

    def do_repr(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        _force_do: bool = False,
    ) -> str:
        """
        A helper function to represent a Multinomial object.
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

        self._order_items()

        result = ["Multinomial(", "["]
        has_epsilon = False

        for i, item in enumerate(self._items):
            item_repr = item.do_repr(
                _circular_refs=_circular_refs, _force_do=_force_do
            )
            item._force_do_hit_count = 0

            if self.optimize_recursive_repr:
                if not has_epsilon and item_repr == "...":
                    has_epsilon = True
                else:
                    continue

            result.append(item_repr)

            if i < len(self._items) - 1:
                result.append(", ")

        result.append("])")

        return "".join(result)

    def to_latex(
        self,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Represents the multinomial in LaTeX format.
        manual_mode and auto_mode are not used in Multinomial class.

        NOTE: If the object's items is empty, it will
        return an empty string.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `latex()` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_REPR_CR_DEPTH",
        )
        if to_return:
            return "..."

        self._order_items()

        result = []
        has_epsilon = False
        has_0 = False

        for item in self._items:
            item_latex = item.to_latex(
                _circular_refs=_circular_refs,
                **kwargs,
            )
            if item_latex == "":
                continue

            if self.optimize_recursive_repr:
                if not has_epsilon and item_latex == "...":
                    has_epsilon = True
                else:
                    continue
                if item_latex == "0":
                    if has_0:
                        continue
                    else:
                        has_0 = True

            if item_latex == "...":
                item_latex = r"\left( ... \right)"

            result.append(item_latex)

        if not result:
            return ""

        if (
            len(result) == 1 and has_epsilon
        ):  # result is ["\left( ... \right)"]
            return "..."

        return r" + ".join(result)

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
        A helper function to convert a Multinomial object to float.
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
                return 0.0
        elif self._force_do_hit_count > self._constants(
            "MAX_FORCE_DO_HIT_COUNT", "<13"
        ):
            self._force_do_hit_count = 0
            return 0.0
        else:
            self._force_do_hit_count += 1

        res = 0.0
        for item in self._items:
            res += item.do_float(
                _circular_refs=_circular_refs, _force_do=_force_do, **kwargs
            )
            item._force_do_hit_count = 0

        return res

    def do_abs(
        self, *, _circular_refs: Optional[dict[NewTypes, int]] = None
    ) -> "Multinomial":
        """
        A helper function to calculate the absolute value of a
        Multinomial object.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `abs()` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Multinomial([0])

        items = [
            item.do_abs(_circular_refs=_circular_refs) for item in self._items
        ]

        return Multinomial(items)

    def do_add(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> "Multinomial":
        """
        A helper function to add a Multinomial object and
        an another object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: If the other object is not supported by calculation.

        Returns:
            Multinomial: The result of adding the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `+` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Multinomial([0])

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when adding a multinomial object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .fraction import Fraction

        res = self.copy(try_deep_copy=True)

        # pylint: disable=unused-variable

        match other:
            case Multinomial(items=items):
                res._items.extend(other._items)
            case int(x) | Integer(x):
                res._items.append(Integer(x))
            case float(x):
                res._items.append(Fraction.from_float(x))
            case other if isinstance(other, NewTypes):
                res._items.append(other)
            case _:
                assert False, assert_fail(
                    invalid_type,
                    "other",
                    other,
                    expected=CalculationSupportsTypes,
                )

        # pylint: enable=unused-variable

        res._order_items()

        if self.simplify_after_calculation:
            res.simplify()

        return res.copy(try_deep_copy=True)

    def do_mul(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to multiply a Multinomial object and
        an anothor object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Multinomial, Monomial]: The result of
                multiplying the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `*` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Multinomial([0])

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when multiplying a multinomial object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .monomial import Monomial

        return Monomial(
            [self, other], simplify=self.simplify_after_calculation
        ).copy(try_deep_copy=True)

    def do_truediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to divide a Multinomial object and
        an anothor object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Multinomial, Fraction]: The result of
                dividing the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `/` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Multinomial([0])

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when dividing a multinomial object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .fraction import Fraction

        return Fraction(
            self, other, simplify=self.simplify_after_calculation
        ).copy(try_deep_copy=True)

    def do_rtruediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to divide an anothor object and
        a Multinomial object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Multinomial, Fraction]: The result of
                dividing the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `/` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Multinomial([0])

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when dividing a multinomial object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .fraction import Fraction

        return Fraction(
            other, self, simplify=self.simplify_after_calculation
        ).copy(try_deep_copy=True)

    def __pow__(self, other: CalculationSupportsTypes) -> NewTypes:
        """
        Raises a Multinomial object to the power of the other object.

        Args:
            other (CalculationSupportsTypes): The exponent to which
                the Multinomial object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Multinomial, Fraction]: The result of
                raising the Multinomial object to the power of
                the other object.
        """

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when raising a multinomial object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .power import Power

        return Power(
            self, other, simplify=self.simplify_after_calculation
        ).copy(try_deep_copy=True)

    def __rpow__(self, other: CalculationSupportsTypes) -> NewTypes:
        """
        Raises an other object to the power of a Multinomial object.

        Args:
            other (CalculationSupportsTypes): The base to which
                the Multinomial object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Multinomial, Fraction]: The result of
                raising the other object to the power of
                the Multinomial object.
        """

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when raising an object",
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
        Checks if the Multinomial object is exactly equal to the other object.,
        which means they have the same items.

        Args:
            other (object): The object to compare with.
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Returns:
            bool: True if the two objects are exactly equal, False otherwise.
        """

        if not isinstance(other, Multinomial):
            return False

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_COMPARISON_CR_DEPTH",
        )
        if to_return:
            return True

        return self._items == other._items

    def __hash__(self) -> int:
        self._order_items()
        return self._get_hash()
