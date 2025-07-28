"""
Constant handler for MoreReadableNumber.
"""

# `CR` means "Circular Reference"
# ===============================

# Maximum circular reference depth when making representation.
# <int> Recommended range: [2, 10] Default: 3
MAX_REPR_CR_DEPTH = 3

# Maximum circular reference depth when calculating value.
# <int> Recommended range: [6, 30] Default: 10
MAX_CALCULATION_CR_DEPTH = 10

# Maximum circular reference depth when comparing values.
# <int> Recommended range: [100, 1000] Default: 200
MAX_COMPARISON_CR_DEPTH = 200

# Maximum circular reference depth when hashing values.
# <int> Recommended range: [50, 200] Default: 50
MAX_HASH_CR_DEPTH = 50

# Maximum circular reference depth when copying values.
# <int> Recommended range: [3, 10] Default: 5
MAX_COPY_CR_DEPTH = 5

# Whether to use entire hash value that be evaluated.
# <bool> Default: False
HASH_USE_ALL = False

# Maximum force do hit count for `BasicClass`,
# When calculating hash value.
# <int> Recommended range: [3, 10] Default: 3
MAX_FORCE_DO_HIT_COUNT = 3

# Whether to copy the value when setting attribute.
# <bool> Default: True
COPY_WHEN_SETTING_ATTR = True

# The maximum simplification loop count for `BasicClass`.
# If the value is still not simplified after this count,
# the value will be returned as is.
# <int> Recommended range: [3, 20] Default: 10
SIMPLIFY_LOOP_LIMIT = 10

# ====================
# FOR `Fraction`
# ====================
# Whether to try to keep the fraction form when adding a `Number` object.
# If this option is True, when adding a `Fration` object and
# a `Number` object, it will like this:
# Fraction(Power(2, 3), Power(5, 2)) + Integer(2)
# -> Fraction(
#   Multinomial(
#     [Power(2, 3), Monomial(
#       [Power(5, 2), Integer(2)]
#     )]
#   ),
#   Power(5, 2)
# )
# <bool> Default: False
TRY_KEEP_FRACTION = False

# ====================
# FOR `Power`
# ====================

# Maximum base value for `Power`,
# if the base value is greater than this value,
# the power will try to make the base value smaller.
# !(NOT ENSURED)
# <int> Recommended range: [10, 100] Default: 30
EXPECTED_MAX_BASE = 30

# Maximum exponent value for `Power`,
# if the exponent value is greater than this value,
# the power will try to make the exponent value smaller.
# !(NOT ENSURED)
# <int> Recommended range: [10, 50] Default: 15
EXPECTED_MAX_EXPONENT = 15

# Whether to try to convert `Power` to radical form
# when doing representation.
# But whether it is True or False,
# if the exponent is a `Fraction` object whose
# numerator is 1, and denominator is a integer that
# is greater than 1, the `Power` object will be
# represented as a radical form.
# Note that the priority of this option is higher than
# "MAX_RADICAL_ROOT_EXPONENT" option.
# <bool> Default: True
TRY_TO_CONVERT_TO_RADICAL = True

# If "TRY_TO_CONVERT_TO_RADICAL", the maximum number of
# root exponent of the radical form.
# Only when the exponent is a `Fraction` object whose
# denomiator is a integer that is greater.
# <int> Recommended range: [2, 5] Default: 3
MAX_RADICAL_ROOT_EXPONENT = 3

# If "FORCE_FRACTION_AS_BASE", the `Fraction` object will
# be used as the base of `Power` object.
# By default, when the base is a `Fraction` object,
# the `Power` object is expected to convert itself
# to a `Fraction` object.
# Like this:
# Power(Fraction(2, 3), 2) -> Fraction(Power(2, 2), Power(3, 2))
# However, this is NOT possible in `Power` class,
# so this is a question for Monomial class or Multinomial class.
# <bool> Default: False
FORCE_FRACTION_AS_BASE = False

# Whether to leave the exponent negative when doing simplification.
# <bool> Default: True
ALLOW_NEGATIVE_EXPONENT = True

# ====================
# FOR `Monomial`
# ====================
# Whether to try to flatten the `Monomial` object
# when doing representation.
# <bool> Default: False
FORCE_FLATTEN_MONOMIAL = False

# ====================
# FOR `Viewer`
# ====================
# Whether to show the `Viewer` object in the representation.
# If this option is False, an object
# like `Viewer(Multinomial([Integer(2), Integer(3)]))`
# will be shown as `Multinomial([Integer(2), Integer(3)])`
# <bool> Default: True
REPR_USE_VIEWER = True
