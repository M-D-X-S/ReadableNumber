"""
A file to test the fraction module.
"""

import unittest
from typing import Generator
from unittest.mock import patch

from src.readable_number import constants, Fraction, LatexMode, Integer, x

# pylint: disable=missing-function-docstring, missing-class-docstring


class TestFraction(unittest.TestCase):
    def test_init_shortcut(self):
        # both numerator and denominator are integers
        f12 = Fraction(1, 2)
        f34 = Fraction(3, 4)

        self.assertIsInstance(f12, Fraction)
        self.assertIsInstance(f34, Fraction)

        self.assertEqual(f12.numerator, 1)
        self.assertEqual(f12.denominator, 2)
        self.assertEqual(f34.numerator, 3)
        self.assertEqual(f34.denominator, 4)

    def test_init_with_wrong_type(self):
        with self.assertRaises(TypeError):
            Fraction(1, None)  # type: ignore
        with self.assertRaises(TypeError):
            Fraction(None, 2)  # type: ignore

    def test_init_with_string(self):
        f = Fraction("1", "2")

        self.assertIsInstance(f, Fraction)
        self.assertEqual(f.numerator, 1)
        self.assertEqual(f.denominator, 2)

    def test_init_with_string_wrong_value(self):
        with self.assertRaises(ValueError):
            Fraction(1, "a")
        with self.assertRaises(ValueError):
            Fraction("a", 1)

    def test_init_with_float(self):
        f = Fraction(1.5, 2.5, simplify=False)

        self.assertIsInstance(f, Fraction)
        self.assertEqual(f.numerator, 15)
        self.assertEqual(f.denominator, 25)

    def test_init_with_wrong_value(self):
        with self.assertRaises(ValueError):
            Fraction(float("inf"), 1)
        with self.assertRaises(ValueError):
            Fraction(float("nan"), 1)
        with self.assertRaises(ValueError):
            Fraction(1, float("-inf"))
        with self.assertRaises(ValueError):
            Fraction(1, float("nan"))

    def test_init_with_zero_denominator(self):
        with self.assertRaises(ValueError):
            Fraction(1, 0)

    def test_set_numerator(self):
        f = Fraction(1, 2)
        f.numerator = 3

        self.assertEqual(f.numerator, 3)
        self.assertEqual(f.denominator, 2)

    def test_set_denominator(self):
        f = Fraction(1, 2)
        f.denominator = 3

        self.assertEqual(f.numerator, 1)
        self.assertEqual(f.denominator, 3)

    def test_set_zero_denominator(self):
        f = Fraction(1, 2)

        with self.assertRaises(ValueError):
            f.denominator = 0

    def test_set_wrong_type(self):
        f = Fraction(1, 2)

        with self.assertRaises(TypeError):
            f.numerator = None  # type: ignore
        with self.assertRaises(TypeError):
            f.denominator = None  # type: ignore

    def test_set_numerator_without_copy(self):
        f1 = Fraction(1, 2)
        f2 = Fraction(2, 3)

        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f1.numerator = f2

        self.assertIs(f1.numerator, f2)

    def test_set_denominator_without_copy(self):
        f1 = Fraction(1, 2)
        f2 = Fraction(2, 3)

        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f1.denominator = f2

        self.assertIs(f1.denominator, f2)

    def test_get_factors_basic(self):
        factors = Fraction(1, 2 * 3 * 5).get_factors()

        self.assertIsInstance(factors, Generator)
        self.assertListEqual(
            list(factors),
            [
                Fraction(1, 1),
                Fraction(1, 2),
                Fraction(1, 3),
                Fraction(1, 5),
                Fraction(1, 2 * 3),
                Fraction(1, 2 * 5),
                Fraction(1, 3 * 5),
                Fraction(1, 2 * 3 * 5),
            ],
        )

    def test_get_factors_with_circular_reference(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.denominator = f
        factors = f.get_factors()

        self.assertIsInstance(factors, Generator)
        self.assertListEqual(list(factors), [Fraction(1, 1)])

    def test_get_factors_has_1(self):
        f = Fraction(1, 2)

        with patch.object(
            f.denominator,
            "get_factors",
            return_value=(1 for _ in range(10)),
        ):
            factors = f.get_factors()

            self.assertIsInstance(factors, Generator)
            self.assertListEqual(list(factors), [Fraction(1, 1)])

    def test_simplify_basic(self):
        f = Fraction(15, 25, simplify=False)

        self.assertEqual(f.numerator, 15)
        self.assertEqual(f.denominator, 25)

        f.simplify()

        self.assertEqual(f.numerator, 3)
        self.assertEqual(f.denominator, 5)

    def test_simplify_with_zero_numerator(self):
        f = Fraction(0, 12)

        self.assertEqual(f.numerator, 0)
        self.assertEqual(f.denominator, 1)

    def test_simplify_with_circualr_reference(self):
        f = Fraction(1, 2)

        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = f
            f.simplify()

        self.assertEqual(f.numerator, f)
        self.assertEqual(f.denominator, 2)

    def test_simplify_with_negative_denominator(self):
        f = Fraction(1, -2, simplify=False)

        self.assertEqual(f.numerator, 1)
        self.assertEqual(f.denominator, -2)

        f.simplify()

        self.assertEqual(f.numerator, -1)
        self.assertEqual(f.denominator, 2)

    def test_simplify_without_change(self):
        f = Fraction(15, 25, simplify=False)
        f_simplified = f.simplify_without_change()

        self.assertEqual(f.numerator, 15)
        self.assertEqual(f.denominator, 25)
        self.assertEqual(f_simplified.numerator, 3)
        self.assertEqual(f_simplified.denominator, 5)

    def test_from_int_basic(self):
        f = Fraction.from_int(2)

        self.assertEqual(f.numerator, 2)
        self.assertEqual(f.denominator, 1)

    def test_from_int_wrong_type(self):
        with self.assertRaises(TypeError):
            Fraction.from_int(2.5)  # type: ignore

    def test_from_float_basic(self):
        f = Fraction.from_float(2.5)

        self.assertEqual(f.numerator, 5)
        self.assertEqual(f.denominator, 2)

    def test_from_float_wrong_type(self):
        with self.assertRaises(TypeError):
            Fraction.from_float("1")  # type: ignore

    def test_from_float_wrong_value(self):
        with self.assertRaises(ValueError):
            Fraction.from_float(float("inf"))
        with self.assertRaises(ValueError):
            Fraction.from_float(float("nan"))

    def test_from_str_basic(self):
        f = Fraction.from_str("3/4")

        self.assertEqual(f.numerator, 3)
        self.assertEqual(f.denominator, 4)

    def test_from_str_wrong_type(self):
        with self.assertRaises(TypeError):
            Fraction.from_str(1)  # type: ignore

    def test_from_str_wrong_format(self):
        with self.assertRaises(ValueError):
            Fraction.from_str("3:4")
        with self.assertRaises(ValueError):
            Fraction.from_str("3/4/5")

    def test_from_auto_basic(self):
        f = Fraction.from_auto(3)

        self.assertEqual(f.numerator, 3)
        self.assertEqual(f.denominator, 1)

        f = Fraction.from_auto(3.5)

        self.assertEqual(f.numerator, 7)
        self.assertEqual(f.denominator, 2)

        f = Fraction.from_auto("3/4")

        self.assertEqual(f.numerator, 3)
        self.assertEqual(f.denominator, 4)

        f = Fraction.from_auto(Fraction(1, 2))

        self.assertEqual(f.numerator, 1)
        self.assertEqual(f.denominator, 2)

    def test_from_auto_wrong_type(self):
        with self.assertRaises(TypeError):
            Fraction.from_auto(None)  # type: ignore

    def test_shallow_copy(self):
        f = Fraction(1, 2)
        f_copy = f.copy()

        self.assertIsInstance(f_copy, Fraction)
        self.assertIsNot(f, f_copy)
        self.assertEqual(f_copy.numerator, 1)
        self.assertEqual(f_copy.denominator, 2)

    def test_deep_copy(self):
        f_n1 = Fraction(1, 2)
        f_d1 = Fraction(2, 3)
        f_n2 = Fraction(f_n1, f_d1, simplify=False)
        f_d2 = Fraction(f_d1, f_n1, simplify=False)
        f = Fraction(f_n2, f_d2, simplify=False)
        f_copy = f.copy()
        f_deep_copy = f.copy(try_deep_copy=True)

        self.assertIsInstance(f_copy, Fraction)
        self.assertIs(f_copy.numerator, f.numerator)
        self.assertIs(f_copy.denominator, f.denominator)

        self.assertIsInstance(f_deep_copy, Fraction)
        self.assertIsNot(f_deep_copy.numerator, f.numerator)
        self.assertIsNot(f_deep_copy.denominator, f.denominator)

    def test_force_deep_copy_numerator(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = f
        f_copy = f.copy(try_deep_copy=True, force_deep_copy=True)

        self.assertIsInstance(f_copy, Fraction)

        # with self.assertRaises(RecursionError):
        #     f_copy.to_string()

    def test_force_deep_copy_denominator(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.denominator = f

        # This is used to avoid checking the value of f.numerator
        # is equal to 0 or not.
        # If we don't do this, the test will cost a lot of time.
        with patch.object(Fraction, "value_eq", return_value=False):
            f_copy = f.copy(try_deep_copy=True, force_deep_copy=True)

        self.assertIsInstance(f_copy, Fraction)

        # with self.assertRaises(RecursionError):
        #     f_copy.to_string()

    def test_get_unknowns_without_unknown_num(self):
        f = Fraction(1, 2)
        f_unknowns = f.get_unknowns()

        self.assertIsInstance(f_unknowns, set)
        self.assertSetEqual(f_unknowns, set())

    def test_get_unknowns_with_unknown_num(self):
        f = Fraction(x, 2)
        f_unknowns = f.get_unknowns()

        self.assertIsInstance(f_unknowns, set)
        self.assertSetEqual(f_unknowns, {x})

    def test_get_unknowns_with_circular_reference(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = x
            f.denominator = f
        f_unknowns = f.get_unknowns()

        self.assertIsInstance(f_unknowns, set)
        self.assertSetEqual(f_unknowns, {x})

    def test_get_coefficient_of_unknowns_without_unknown_num(self):
        f = Fraction(1, 2)
        f_coefficients = f.get_coefficient_of_unknowns([x])

        self.assertIsInstance(f_coefficients, list)
        self.assertListEqual(f_coefficients, [Integer(0)])

    def test_get_coefficient_of_unknowns_with_unknown_num(self):
        f = Fraction(x, 2)
        f_coefficients = f.get_coefficient_of_unknowns([x])

        self.assertIsInstance(f_coefficients, list)
        self.assertListEqual(f_coefficients, [Fraction(1, 2)])

    def test_get_coefficient_of_unknowns_with_empty_unknown_nums(self):
        f = Fraction(x, 2)
        f_coefficients = f.get_coefficient_of_unknowns([])

        self.assertIsInstance(f_coefficients, list)
        self.assertListEqual(f_coefficients, [])

    def test_get_coefficient_of_unknowns_with_circular_reference(self):
        f1 = Fraction(1, 2)
        f2 = Fraction(x, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f1.numerator = f1
            f2.denominator = f1
        f_coefficients = f2.get_coefficient_of_unknowns([x], True)

        # We create a new Fraction object to avoid the influence
        # of the original f1 and f2.
        f_expected = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f_expected.numerator = f_expected

        expected = Fraction(Integer(1), f_expected, simplify=True)

        self.assertIsInstance(f_coefficients, list)
        self.assertListEqual(f_coefficients, [expected])

        # _do_simplify = False
        f1 = Fraction(1, 2)
        f2 = Fraction(x, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f1.numerator = f1
            f2.denominator = f1
        f_coefficients = f2.get_coefficient_of_unknowns([x], False)

        # Here, we can also use:
        # f_expected = f1
        # or
        # f_expected = f2.denominator
        f_expected = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f_expected.numerator = f_expected

        expected = Fraction(Integer(1), f_expected, simplify=False)

        self.assertIsInstance(f_coefficients, list)
        self.assertListEqual(f_coefficients, [expected])

    @unittest.skip("Not implemented yet")
    def test_get_coefficient_of_unknowns_denomiator_is_unknown(self):
        with self.assertRaises(NotImplementedError):
            Fraction(1, x).get_coefficient_of_unknowns([x])

    def tes_contain_unknown_num_basic(self):
        f = Fraction(1, 2)
        self.assertFalse(f.contain_unknown_num())

        f = Fraction(x, 2)
        self.assertTrue(f.contain_unknown_num())

        f = Fraction(1, x)
        self.assertTrue(f.contain_unknown_num())

    def test_contain_unknown_num_with_circular_reference(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = f

        self.assertFalse(f.contain_unknown_num())

    def test_set_values_without_values(self):
        f = Fraction(x, 2)
        f.set_values({})

        with self.assertRaises(ValueError):
            f.do_float()

    def test_set_values_with_values(self):
        f = Fraction(x, 2)
        f.set_values({x: 3})

        self.assertEqual(f.numerator, 3)
        self.assertEqual(f.denominator, 2)

        x.set_values({x: None})

    def test_set_values_with_circular_reference(self):
        f1 = Fraction(1, 2)
        f2 = Fraction(x, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f1.numerator = f1
            f2.denominator = f1
        f2.set_values({x: 3})

        self.assertEqual(f2.numerator, 3)
        # self.assertEqual(f2.denominator, ...)

        x.set_values({x: None})

    def test_to_string_basic(self):
        f1 = Fraction(1, 2)
        output = f1.to_string()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "1/2")

        f2 = Fraction(3, 4)
        f1.numerator = f2
        f1.denominator = f2
        output = f1.to_string()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "(3/4)/(3/4)")

        f1.simplify()
        output = f1.to_string()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "1/1")

    def test_to_string_use_space_separator(self):
        f = Fraction(1, 2)
        with patch.object(Fraction, "use_space_separator", True):
            output = f.to_string()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "1 / 2")

    def test_to_string_with_circular_reference(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = f
        output = f.to_string()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "((((...)/2)/2)/2)/2")

    def test_to_string_with_both_circular_reference(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = f
            f.denominator = f
        with patch.object(constants, "MAX_REPR_CR_DEPTH", 3):
            output = f.to_string()

        self.assertIsInstance(output, str)
        self.assertEqual(
            output,
            "((((...)/(...))/((...)/(...)))/(((...)/(...))/((...)/(...))))/"
            "((((...)/(...))/((...)/(...)))/(((...)/(...))/((...)/(...))))",
        )

        with patch.object(f, "optimize_recursive_repr", True):
            output = f.to_string()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "...")

    def test_do_repr_basic(self):
        f1 = Fraction(1, 2)
        output = f1.do_repr()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "Fraction(Integer(1), Integer(2))")

        f2 = Fraction(3, 4)
        f1.numerator = f2
        f1.denominator = f2
        output = f1.do_repr()

        self.assertIsInstance(output, str)
        self.assertEqual(
            output,
            "Fraction(Fraction(Integer(3), Integer(4)), "
            "Fraction(Integer(3), Integer(4)))",
        )

        f1.simplify()
        output = f1.do_repr()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "Fraction(Integer(1), Integer(1))")

    def test_do_repr_with_circular_reference(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = f
        output = f.do_repr()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "Fraction(..., Integer(2))")

    def test_do_repr_with_circular_reference2(self):
        f1 = Fraction(1, 2)
        f2 = Fraction(3, 4)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f1.numerator = f2
            f2.numerator = f1
        with patch.object(constants, "MAX_REPR_CR_DEPTH", 3):
            output = f1.do_repr()

        self.assertIsInstance(output, str)
        self.assertEqual(
            output,
            "Fraction(Fraction(Fraction(Fraction("
            "Fraction(Fraction(Fraction(Fraction(..., "
            "Integer(4)), Integer(2)), Integer(4)), Integer(2)), "
            "Integer(4)), Integer(2)), Integer(4)), Integer(2))",
        )

    def test_do_repr_with_both_circular_reference(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = f
            f.denominator = f
        output = f.do_repr()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "Fraction(..., ...)")

        with patch.object(f, "optimize_recursive_repr", True):
            output = f.do_repr()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "...")

    def test_to_latex_auto_mode(self):
        # AUTO -> FRAC = NORMAL
        f1 = Fraction(1, 2)
        output = f1.to_latex(fraction_manual_mode=None)  # Default

        self.assertIsInstance(output, str)
        self.assertEqual(output, "\\frac{ 1 }{ 2 }")

        # AUTO -> CFRAC
        f2 = Fraction(3, 4)
        f1.numerator = f2
        output = f1.to_latex()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "\\cfrac{ \\frac{ 3 }{ 4 } }{ 2 }")

        f1 = Fraction(1, 2)
        f2.denominator = f1
        output = f2.to_latex()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "\\cfrac{ 3 }{ \\frac{ 1 }{ 2 } }")

    def test_to_latex_manual_mode(self):
        # FRAC = NORMAL
        f = Fraction(1, 2)
        output = f.to_latex(fraction_manual_mode=LatexMode.FRAC)
        # Same as:
        # output = f.to_latex(fraction_manual_mode=LatexMode.NORMAL)

        self.assertIsInstance(output, str)
        self.assertEqual(output, "\\frac{ 1 }{ 2 }")

        # CFRAC
        output = f.to_latex(fraction_manual_mode=LatexMode.CFRAC)

        self.assertIsInstance(output, str)
        self.assertEqual(output, "\\cfrac{ 1 }{ 2 }")

        # TFRAC
        output = f.to_latex(fraction_manual_mode=LatexMode.TFRAC)

        self.assertIsInstance(output, str)
        self.assertEqual(output, "\\tfrac{ 1 }{ 2 }")

    def test_to_latex_with_wrong_mode(self):
        with self.assertRaises(TypeError):
            Fraction(1, 2).to_latex(
                fraction_manual_mode="wrong_mode"  # type: ignore
            )

    def test_to_latex_with_circular_reference(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = f
        with patch.object(constants, "MAX_REPR_CR_DEPTH", 3):
            output = f.to_latex()

        self.assertIsInstance(output, str)
        self.assertEqual(
            output,
            "\\cfrac{ \\cfrac{ \\cfrac{ \\cfrac"
            "{ ... }{ 2 } }{ 2 } }{ 2 } }{ 2 }",
        )

        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.denominator = f
        with patch.object(constants, "MAX_REPR_CR_DEPTH", 3):
            output = f.to_latex()

        self.assertIsInstance(output, str)
        self.assertEqual(
            output,
            "\\cfrac{ \\cfrac{ \\cfrac{ \\cfrac"
            "{ ... }{ ... } }{ \\cfrac{ ... }{ ... } } }"
            "{ \\cfrac{ \\cfrac{ ... }{ ... } }"
            "{ \\cfrac{ ... }{ ... } } } }"
            "{ \\cfrac{ \\cfrac{ \\cfrac{ ... }{ ... } }"
            "{ \\cfrac{ ... }{ ... } } }"
            "{ \\cfrac{ \\cfrac{ ... }{ ... } }"
            "{ \\cfrac{ ... }{ ... } } } }",
        )

        with patch.object(f, "optimize_recursive_repr", True):
            output = f.to_latex()

        self.assertIsInstance(output, str)
        self.assertEqual(output, "...")

    def test_reciprocal_basic(self):
        f1 = Fraction(1, 2)
        f2 = f1.reciprocal()

        self.assertIsInstance(f2, Fraction)
        self.assertEqual(f2.numerator, 2)
        self.assertEqual(f2.denominator, 1)

    def test_reciprocal_with_circular_reference(self):
        f1 = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f1.numerator = f1
        f2 = f1.reciprocal()

        self.assertIsInstance(f2, Fraction)
        self.assertEqual(f2.numerator, 2)
        self.assertEqual(f2.denominator, f1.copy(try_deep_copy=True))

    def test_self_reciprocal_basic(self):
        f = Fraction(1, 2)
        f.self_reciprocal()

        self.assertEqual(f.numerator, 2)
        self.assertEqual(f.denominator, 1)

    def test_self_reciprocal_with_circular_reference(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = f
        f.self_reciprocal()

        self.assertEqual(f.numerator, 2)
        self.assertEqual(f.denominator, f.copy(try_deep_copy=True))

    def test_do_exactly_eq_basic(self):
        f1 = Fraction(1, 2)
        f2 = Fraction(1, 2)

        self.assertTrue(f1.do_exactly_eq(f2))

        f1 = Fraction(1, 2)
        f2 = Fraction(3, 4)

        self.assertFalse(f1.do_exactly_eq(f2))

    def test_do_exactly_eq_other_type(self):
        f = Fraction(1, 2)

        self.assertFalse(f.do_exactly_eq(1))
        self.assertFalse(f.do_exactly_eq(0.0))
        self.assertFalse(f.do_exactly_eq("1"))
        self.assertFalse(f.do_exactly_eq(None))
        self.assertFalse(f.do_exactly_eq(x))

    def test_do_exactly_eq_with_circular_reference(self):
        f = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            f.numerator = f

        expected = Fraction(1, 2)
        with patch.object(constants, "COPY_WHEN_SETTING_ATTR", False):
            expected.numerator = expected

        self.assertTrue(f.do_exactly_eq(expected))

    # Tests for calculation are in test_calculation.py
