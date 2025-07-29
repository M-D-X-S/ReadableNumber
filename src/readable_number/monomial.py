"""
Monomial

The definition of the `Monomial` class.
"""

from itertools import tee
from typing import Any, Generator, Optional, Union

from ._error_helper import invalid_type, assert_fail
from ._types import CalculationSupportsTypes, NewTypes
from .basic_class import Container
from .integer import Integer


# pylint: disable=import-outside-toplevel, protected-access


class Monomial(Container):
    """
    A class to represent a monomial.
    """

    # ====================
    # public methods
    # ====================

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
        from .power import Power

        self._simplify(
            _circular_refs=_circular_refs,
            optimize_mapping={
                Monomial: (lambda _: self._items),
                Power: self._optimize_power,
            },
        )

    def _optimize_power(self, item: NewTypes) -> list[NewTypes]:
        """
        A helper method to optimize the power in the monomial.

        Args:
            item (Power): The item to optimize.

        Returns:
            list[NewTypes]: The optimized power.
        """

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

        for buff_item in buff_res:
            match buff_item:
                case Power(b, e):
                    res.extend(self._optimize_power(buff_item))
                case other:
                    res.append(buff_item)

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

    # ====================
    # represent
    # ====================

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
            "str()",
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
                "repr()",
                _circular_refs=_circular_refs,
                constant_get_key="MAX_REPR_CR_DEPTH",
            )
            if to_return:
                return "..."

        elif self._force_do_helper():
            return "..."

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
                "float()",
                _circular_refs=_circular_refs,
                constant_get_key="MAX_CALCULATION_CR_DEPTH",
            )
            if to_return:
                return 1.0

        elif self._force_do_helper():
            return 1.0

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

        res = self._do_abs_helper(_circular_refs=_circular_refs)
        if res is None:
            return Monomial([1])
        return res.copy(try_deep_copy=True)

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
            "+",
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
            "*",
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

        res = self._do_truediv_helper(other, _circular_refs=_circular_refs)
        if res is None:
            return Monomial([1])
        return res.copy(try_deep_copy=True)

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

        res = self._do_rtruediv_helper(other, _circular_refs=_circular_refs)
        if res is None:
            return Monomial([1])
        return res.copy(try_deep_copy=True)
