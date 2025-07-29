"""
Multinomial

The definition of the `Multinomial` class.
"""

from typing import Any, Generator, Optional, Union

from ._error_helper import invalid_type, assert_fail
from ._types import CalculationSupportsTypes, NewTypes
from .basic_class import Container
from .integer import Integer


# pylint: disable=import-outside-toplevel, protected-access


class Multinomial(Container):
    """
    A class to represent a multinomial.
    """

    # ====================
    # public methods
    # ====================

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
        self._simplify(_circular_refs=_circular_refs)

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
        Represents the multinomial in string format.

        NOTE: If the object's items is empty, it will return
        an empty string.
        """

        to_return, _circular_refs = self._circular_reference_dict(
            "str()",
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
                "repr()",
                _circular_refs=_circular_refs,
                constant_get_key="MAX_REPR_CR_DEPTH",
            )
            if to_return:
                return "..."

        elif self._force_do_helper():
            return "..."

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
                "float()",
                _circular_refs=_circular_refs,
                constant_get_key="MAX_CALCULATION_CR_DEPTH",
            )
            if to_return:
                return 0.0

        elif self._force_do_helper():
            return 0.0

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

        res = self._do_abs_helper(_circular_refs=_circular_refs)
        if res is None:
            return Multinomial([0])
        return res.copy(try_deep_copy=True)

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
            "+",
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
            "*",
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

        res = self._do_truediv_helper(other, _circular_refs=_circular_refs)
        if res is None:
            return Multinomial([0])
        return res.copy(try_deep_copy=True)

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

        res = self._do_rtruediv_helper(other, _circular_refs=_circular_refs)
        if res is None:
            return Multinomial([0])
        return res.copy(try_deep_copy=True)
