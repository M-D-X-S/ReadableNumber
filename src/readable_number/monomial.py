"""
Monomial

The definition of the `Monomial` class.
"""

from enum import StrEnum, auto
from itertools import tee
from typing import Any, Generator, Iterable, Optional, Union, Sequence

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


class Monomial(Container):
    """
    A class to represent a monomial.
    """

    # Whether to simplify the Monomial after calculation.
    simplify_after_calculation = False

    # Whether the separator is between the space.
    # If True, the separator is " * ",
    # If False, the separator is "*".
    use_space_separator = False

    # Whether to optimize the recursive representation of the monomial.
    # For example, if a power is represented as `(...)*(...)*(...)`,
    # it will be optimized to `...` if this attribute is True.
    # Also, if `1` is in the string representation of the monomial,
    # it will be skipped if this attribute is True.
    # e.g. `1*1*1*1` -> `1`
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
        The initialization of the Monomial class.

        The argument `order_mode` is a list or tuple of `OrderMode`.
        It specifies the order of the items in the string
        representation of the Monomial object.
        For example, by default, the order is
        (Integer, Multiplication, Power, Fraction, Monomial, UnknownNum)
        which means the items will be sorted in the order
        of `Integer`, `Multiplication`, `Power`, `Fraction`,
        `Monomial` and `UnknownNum`.

        Args:
            items (ItemsSupportsTypes): The items of the monomial.
            simplify (bool, optional):
                Whether to simplify the monomial after initialization.
                Defaults to True.
            order_mode
            (Union[list[OrderMode], tuple[OrderMode, ...]], optional):
                The order mode of the monomial.

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
        Returns a viewer of the items of the monomial.
        """
        return Viewer(self._items)

    @property
    def order_mode(self) -> tuple[OrderMode, ...]:
        """
        Returns the order mode of the monomial.
        """
        return self._order_mode

    @order_mode.setter
    def order_mode(
        self, value: Union[list[OrderMode], tuple[OrderMode, ...]]
    ) -> None:
        """
        Sets the order mode of the monomial.
        """
        _, self._order_mode = self._init_args_handler(None, value)

    def _order_items(self) -> None:
        """
        Orders the items of the monomial.
        """

        from .fraction import Fraction
        from .multinomial import Multinomial
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
        A helper method to handle the arguments.

        Args:
            items (Optional[ItemsSupportsTypes])
            order_mode (Union[list[OrderMode], tuple[OrderMode, ...]])

        Raises:
            TypeError: When the items is not a list or tuple.
            TypeError: When the item in the items
                is not a CalculationSupportsTypes.

        Returns:
            tuple[PropertyTypes, tuple[OrderMode, ...]]:
                The items of the monomial.
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
        Returns the length of the items of the monomial.
        """
        return len(self._items)

    def __getitem__(self, index: Union[int, slice]) -> Viewer:
        """
        Returns a viewer of the items of the monomial.
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
        Deletes the items of the monomial.
        """
        del self._items[index]

    def __iter__(self) -> Generator[Viewer, None, None]:
        """
        Returns a generator of the items of the monomial.
        """
        for item in self._items:
            yield Viewer(item.copy(try_deep_copy=True))

    def __contains__(self, item: Any) -> bool:
        """
        Returns whether the item is in the monomial.
        """
        return item in self._items

    def append(self, item: CalculationSupportsTypes) -> None:
        """
        Appends an item to the end of the monomial.
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
        Extends the monomial with the items.
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
        Removes the first occurrence of the item in the monomial.
        """
        self._items.remove(item)

    def clear(self) -> None:
        """
        Removes all items from the monomial.
        """
        self._items.clear()

    def count(self, item: NewTypes) -> int:
        """
        Returns the number of occurrences of the item in the monomial.
        """
        return self._items.count(item)

    def get_factors(
        self, _circular_refs: Optional[set[NewTypes]] = None
    ) -> Generator[NewTypes, None, None]:
        """
        Gets the factors of the monomial.

        Yields:
            Generator[NewTypes, None, None]:
                A generator of the factors.
        """

        to_return, _circular_refs = self._circular_reference_set(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
            _circular_refs=_circular_refs,
        )
        if to_return or not self._items:
            yield Integer(1)
            return

        def get_all_factors(
            factors_iter: Generator[
                Generator[NewTypes, None, None], None, None
            ],
        ) -> Generator[NewTypes, None, None]:
            """
            A helper method to get all factors of the monomial.

            In one iteration, it will get factors of the first item
            of the `factors_iter`.
            """

            try:
                factors = next(factors_iter)
            except StopIteration:
                yield Integer(1)
                return

            for factor1 in factors:
                for factor2 in get_all_factors(factors_iter):
                    yield factor1 * factor2

        former_factors = set()
        factors_iter = (
            item.get_factors(_circular_refs) for item in self._items
        )

        for factor in get_all_factors(factors_iter):
            if factor in former_factors:
                continue
            former_factors.add(factor)
            yield factor

    def simplify(self, _circular_refs: Optional[set[NewTypes]] = None) -> None:
        """
        Simplifies the monomial.
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

                mul_res = item1 * item2
                if not isinstance(mul_res, Monomial):
                    res1.append(mul_res)
                else:
                    res1.extend((item1, item2))

        # optimization
        res2 = []

        from .power import Power

        # pylint: disable=unused-variable

        for item in res1:
            match item:
                case Monomial(items=items) as obj:
                    res2.extend(obj._items)
                case Power(b, e):
                    res2.extend(self._optimize_power(item))
                case other:
                    res2.append(item)

        # pylint: enable=unused-variable

        self._items = res2
        self._order_items()

    def _optimize_power(self, item: NewTypes) -> list[NewTypes]:

        buff_res = []

        from .power import Power

        if not isinstance(item, Power):
            raise TypeError(
                invalid_type(
                    "item",
                    item,
                    expected=Power,
                    more_msg="when optimizing power",
                )
            )

        def get_middle(
            item: Generator[NewTypes, None, None],
        ) -> Optional[NewTypes]:
            g1, g2 = tee(item)
            val = None
            for index, _ in enumerate(g1):
                if index % 2 == 0:
                    val = next(g2)
            return val

        # After a optimization, the result will be stored in
        # buff_res, so we need to check the items in buff_res
        # to see if they need further optimization,
        # and that's why the optimization is mutually exclusive.

        from .fraction import Fraction

        # 15**3 -> 3**3 * 5**3
        max_base: int = self._constants("EXPECTED_MAX_BASE", ">2")
        max_exponent: int = self._constants("EXPECTED_MAX_EXPONENT", ">2")

        if isinstance(item.base, Fraction) and not self._constants(
            "FORCE_FRACTION_AS_BASE", "bool"
        ):
            buff_res.append(
                Fraction(
                    Monomial((Power(item.base.numerator, item.exponent),)),
                    Monomial((Power(item.base.denominator, item.exponent),)),
                )
            )

        elif (
            item.base > max_base
            and (middle := get_middle(item.base.get_factors())) is not None
        ):
            base1: NewTypes = item.base / middle
            base2: NewTypes = middle
            handling = [
                Power(base1, item.exponent),
                Power(base2, item.exponent),
            ]

            while handling:
                handling_item = handling.pop()
                if (
                    handling_item.base > max_base
                    and (
                        middle := get_middle(handling_item.base.get_factors())
                    )
                    is not None
                ):
                    base1, base2 = (
                        handling_item.base / middle,
                        middle,
                    )
                    handling.extend(
                        (
                            Power(base1, handling_item.exponent),
                            Power(base2, handling_item.exponent),
                        )
                    )
                else:
                    buff_res.append(handling_item)

        # 3**18 -> 3**9 * 3**9
        elif item.exponent > max_exponent:
            exponent1: int = item.exponent // 2
            exponent2: NewTypes = item.exponent - exponent1
            handling = [
                Power(item.base, exponent1),
                Power(item.base, exponent2),
            ]

            while handling:
                handling_item = handling.pop()
                if handling_item.exponent > max_exponent:
                    exponent1, exponent2 = (
                        handling_item.exponent // 2,
                        handling_item.exponent - exponent1,
                    )
                    handling.extend(
                        (
                            Power(handling_item.base, exponent1),
                            Power(handling_item.base, exponent2),
                        )
                    )
                else:
                    buff_res.append(handling_item)

        res = []

        # pylint: disable=unused-variable

        for item in buff_res:
            match item:
                case Power(b, e):
                    res.extend(self._optimize_power(item))
                case other:
                    res.append(item)

        # pylint: enable=unused-variable

        return res

    def simplify_without_change(self) -> "Monomial":
        """
        Returns a new simplified monomial without changing the original one.
        """
        return Monomial(
            tuple(self._items), simplify=True, order_mode=self._order_mode
        )

    @classmethod
    def from_auto(
        cls, item: Union[int, "Monomial"], *args, **kwargs
    ) -> "Monomial":
        """
        Creates a monomial from an integer or another Monomial object.
        Will set `simplify` to False by default.
        """

        # pylint: disable=unused-variable

        match item:
            case int(x) | Integer(x):
                return cls.from_int(item, *args, **kwargs)
            case Monomial(items=items):
                return item.copy(*args, **kwargs)

            case other:
                raise TypeError(
                    invalid_type(
                        "item",
                        item,
                        more_msg="when creating a monomial from an integer",
                        expected=int,
                    )
                )

        # pylint: enable=unused-variable

    @classmethod
    def from_int(cls, num: int, *args, **kwargs) -> "Monomial":
        """
        Creates a monomial from an integer.
        Will set `simplify` to False by default.
        """

        if not isinstance(num, int):
            raise TypeError(
                invalid_type(
                    "num",
                    num,
                    more_msg="when creating a monomial from an integer",
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
    ) -> "Monomial":
        """
        Creates a copy of the Monomial object.
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
            Monomial: The copy of the Monomial object.
        """

        if not try_deep_copy:
            return Monomial(tuple(self._items), simplify=False)

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

        return Monomial(items, simplify=False)

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
                res.append(Monomial(buff, simplify=_do_simplify))
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
            "do not pass `_circular_refs` in wrong type "
            "when calling this method",
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
        Represents the monomial in string format.

        NOTE: If the object's items is empty, it will
        return an empty string.
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

        from .power import Power
        from .unknown_num import UnknownNum

        result = []
        has_epsilon = False
        has_1 = False

        # pylint: disable=unused-variable

        for item in self._items:
            match item:
                case int(x) | Integer(x):
                    r_str = str(x)

                case Power(b, e):
                    r_str = item.to_string(
                        _circular_refs=_circular_refs,
                        **kwargs,
                    )

                    if self.optimize_recursive_repr and r_str == "...":
                        if not has_epsilon:
                            has_epsilon = True
                            r_str = "(...)"
                        else:
                            continue
                    # else:
                    #     r_str = r_str

                case UnknownNum(label, value, string, latex):
                    r_str = item.to_string(
                        _circular_refs=_circular_refs,
                        **kwargs,
                    )

                    if self.optimize_recursive_repr and r_str == "...":
                        if not has_epsilon:
                            has_epsilon = True
                            r_str = "(...)"
                        else:
                            continue
                    # else:
                    #     r_str = r_str

                case other if isinstance(other, Container):
                    r_str = item.to_string(
                        _circular_refs=_circular_refs,
                        **kwargs,
                    )

                    if r_str == "":
                        continue

                    if self.optimize_recursive_repr and r_str == "...":
                        if not has_epsilon:
                            has_epsilon = True
                            r_str = "(...)"
                        else:
                            continue
                    else:
                        r_str = f"({r_str})"

                case other if isinstance(other, NewTypes):
                    r_str = item.to_string(
                        _circular_refs=_circular_refs,
                        **kwargs,
                    )

                    if self.optimize_recursive_repr and r_str == "...":
                        if not has_epsilon:
                            has_epsilon = True
                            r_str = "(...)"
                        else:
                            continue
                    else:
                        r_str = f"({r_str})"

                case other:
                    assert False, assert_fail(
                        invalid_type, "item", item, expected=NewTypes
                    )

            if self.optimize_recursive_repr:
                # if r_str == "0":
                #     return "0"
                if r_str == "1":
                    if has_1:
                        continue
                    has_1 = True

            result.append(r_str)

        # pylint: enable=unused-variable

        if not result:
            return ""

        if len(result) == 1 and has_epsilon:  # result is ["(...)"]
            return "..."

        if self.use_space_separator:
            sep = " * "
        else:
            sep = "*"
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
        A helper function to represent a Monomial object.
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
            "MAX_FORCE_DO_HIT_COUNT"
        ):
            self._force_do_hit_count = 0
            return "..."
        else:
            self._force_do_hit_count += 1

        self._order_items()

        result = ["Monomial(", "["]
        has_epsilon = False

        for i, item in enumerate(self._items):
            # pylint: disable=unused-variable

            try:
                match item:
                    case Integer(value=value):
                        result.append(item.do_repr())
                        item._force_do_hit_count = 0

                    case other if isinstance(other, NewTypes):
                        r_str = item.do_repr(
                            _circular_refs=_circular_refs,
                            _force_do=_force_do,
                        )
                        item._force_do_hit_count = 0

                        if self.optimize_recursive_repr and r_str == "...":
                            if not has_epsilon:
                                has_epsilon = True
                                result.append("...")
                            else:
                                continue
                        else:
                            result.append(r_str)

                    case other:
                        assert False, assert_fail(
                            invalid_type, "item", item, expected=NewTypes
                        )

            # pylint: enable=unused-variable

            except RecursionError:
                result.append("...")

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
        Represents the monomial in LaTeX format.
        manual_mode and auto_mode are not used in Monomial class.

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

        from .power import Power
        from .unknown_num import UnknownNum

        result = []
        has_epsilon = False
        has_1 = False

        # pylint: disable=unused-variable

        for item in self._items:
            match item:
                case int(x) | Integer(x):
                    r_str = str(x)

                case Power(b, e):
                    r_str = item.to_latex(
                        _circular_refs=_circular_refs,
                        **kwargs,
                    )

                    if self.optimize_recursive_repr and r_str == "...":
                        if not has_epsilon:
                            has_epsilon = True
                            r_str = r"\left( ... \right)"
                        else:
                            continue
                    # else:
                    #     r_str = r_str

                case UnknownNum(label, value, string, latex):
                    r_str = item.to_latex(
                        _circular_refs=_circular_refs,
                        **kwargs,
                    )

                    if self.optimize_recursive_repr and r_str == "...":
                        if not has_epsilon:
                            has_epsilon = True
                            r_str = r"\left( ... \right)"
                        else:
                            continue
                    # else:
                    #     r_str = r_str

                case other if isinstance(other, Container):
                    r_str = item.to_latex(
                        _circular_refs=_circular_refs,
                        **kwargs,
                    )

                    if r_str == "":
                        continue

                    if self.optimize_recursive_repr and r_str == "...":
                        if not has_epsilon:
                            has_epsilon = True
                            r_str = r"\left( ... \right)"
                        else:
                            continue
                    else:
                        r_str = rf"\left( {r_str} \right)"

                case other if isinstance(other, NewTypes):
                    r_str = item.to_latex(
                        _circular_refs=_circular_refs,
                        **kwargs,
                    )

                    if self.optimize_recursive_repr and r_str == "...":
                        if not has_epsilon:
                            has_epsilon = True
                            result.append(r"\left( ... \right)")
                        else:
                            continue
                    else:
                        r_str = rf"\left( {r_str} \right)"

                case other:
                    assert False, assert_fail(
                        invalid_type, "item", item, expected=NewTypes
                    )

            # if r_str == "0":
            #     return "0"
            if r_str == "1":
                if has_1:
                    continue
                else:
                    has_1 = True

            result.append(r_str)

        # pylint: enable=unused-variable

        if not result:
            return ""

        if (
            len(result) == 1 and has_epsilon
        ):  # result is ["\left( ... \right)"]
            return "..."

        return r" \cdot ".join(result)

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
        A helper function to convert a Monomial object to float.
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

        res = 1.0
        for item in self._items:
            res *= item.do_float(
                _circular_refs=_circular_refs,
                _force_do=_force_do,
                **kwargs,
            )
            item._force_do_hit_count = 0

        return res

    def do_abs(
        self, *, _circular_refs: Optional[dict[NewTypes, int]] = None
    ) -> "Monomial":
        """
        A helper function to calculate the absolute value of a Monomial object.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `abs()` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Monomial([1])

        items = [
            item.do_abs(_circular_refs=_circular_refs) for item in self._items
        ]

        return Monomial(items).copy(try_deep_copy=True)  # type: ignore

    def do_add(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to add a Monomial object and
        an anothor object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Monomial, Multinomial]: The result of
                adding the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `+` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Monomial([1])

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when adding a Monomial object",
                    expected=CalculationSupportsTypes,
                )
            )

        if isinstance(other, Monomial) and self._items == other._items:
            return Monomial(
                [Integer(2), *self._items.copy()],
                simplify=self.simplify_after_calculation,
                order_mode=self._order_mode,
            )

        from .multinomial import Multinomial

        if isinstance(other, Multinomial):
            return other.do_add(self, _circular_refs=_circular_refs)

        return Multinomial(
            [self, other], simplify=self.simplify_after_calculation
        )

    def do_mul(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> "Monomial":
        """
        A helper function to multiply a Monomial object and
        an anothor object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Monomial: The result of multiplying the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `*` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Monomial([1])

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when multiplying a Monomial object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .fraction import Fraction

        res = self.copy(try_deep_copy=True)

        # pylint: disable=unused-variable

        match other:
            case Monomial(items=items):
                res._items.extend(other._items)
            case int(x) | Integer(x):
                res._items.append(Integer(x))
            case float(x):
                res._items.append(Fraction.from_float(x))
            case other if isinstance(other, NewTypes):
                res._items.append(other)
            case other:
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

    def do_truediv(
        self,
        other: CalculationSupportsTypes,
        *,
        _circular_refs: Optional[dict[NewTypes, int]] = None,
    ) -> NewTypes:
        """
        A helper function to divide a Monomial object and
        an anothor object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Monomial, Fraction]: The result of
                dividing the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `/` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Monomial([1])

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when dividing a Monomial object",
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
        a Monomial object that be supported by calculation.

        Args:
            other (CalculationSupportsTypes)
            _circular_refs (Optional[dict[NewTypes, int]], optional):
                A dictionary to keep track of circular references.
                Defaults to None.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Union[Monomial, Fraction]: The result of
                dividing the two objects.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "do not pass `_circular_refs` in wrong type "
            "when calling this method, use `/` instead",
            _circular_refs=_circular_refs,
            constant_get_key="MAX_CALCULATION_CR_DEPTH",
        )
        if to_return:
            return Monomial([1])

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when dividing a Monomial object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .fraction import Fraction

        return Fraction(
            other, self, simplify=self.simplify_after_calculation
        ).copy(try_deep_copy=True)

    def __pow__(self, other: CalculationSupportsTypes) -> NewTypes:
        """
        Raises a Monomial object to the power of the other object.

        Args:
            other (CalculationSupportsTypes): The exponent to which
                the Monomial object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Power: The result of raising the Monomial object to
                the power of the other object.
        """

        if not isinstance(other, CalculationSupportsTypes):
            raise TypeError(
                invalid_type(
                    "other",
                    other,
                    more_msg="when raising a Monomial object",
                    expected=CalculationSupportsTypes,
                )
            )

        from .power import Power

        return Power(
            self, other, simplify=self.simplify_after_calculation
        ).copy(try_deep_copy=True)

    def __rpow__(self, other: CalculationSupportsTypes) -> NewTypes:
        """
        Raises an object to the power of a Monomial object.

        Args:
            other (CalculationSupportsTypes): The base to which
                the Monomial object is raised.

        Raises:
            TypeError: When the other is not supported by calculation.

        Returns:
            Power: The result of raising the other object to
                the power of the Monomial object.
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

        if not isinstance(other, Monomial):
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
