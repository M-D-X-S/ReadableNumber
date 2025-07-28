"""
Test file for more_readable_number package.
"""

import os
from io import BytesIO

from readable_number import constants, Fraction, Power, Integer, x, LatexMode
from readable_number.latex2pic import latex_to_jpg
from readable_number.show import show_pic_on_browser


constants.COPY_WHEN_SETTING_ATTR = False

f12 = Fraction(1, 2)
f34 = Fraction(3, 4)

print(f"f12 = {f12.to_string()}")
print(f"f12 = {f12.to_latex()}")
print(f"f34 = {f34.to_string()}")
print(f"f34 = {f34.to_latex()}")

print(f"f12 + f34 = {f12 + f34}")


p23 = Power(2, 3)
p34 = Power(3, 4)

print(f"p23 = {p23.to_string()}")
print(f"p23 = {p23.to_latex()}")
print(f"p34 = {p34.to_string()}")
print(f"p34 = {p34.to_latex()}")


print(Fraction(Fraction(-1, 2), 6).to_string())

r1 = f34 / p23 * f12 / p34

print(r1.to_string())
print(r1.to_latex(fraction_auto_mode=True))

# try circular reference
f12.numerator = f34
f34.numerator = f12

f12.optimize_recursive_repr = False
f34.optimize_recursive_repr = False

print(f"f12 = {f12.to_string()}")
print(f"f12 = {f12.to_latex(auto_mode=True)}")
print(f"f34 = {f34.to_string()}")
print(f"f34 = {f34.to_latex(auto_mode=True)}")

print(f"f12 = {f12.do_float()}")
print(f"f34 = {f34.do_float()}")

# try deep copy
f12_copy = f12.copy(try_deep_copy=True)
f34_copy = f34.copy(try_deep_copy=True)

print(f"f12_copy = {f12_copy.to_string()}")
print(f"f12_copy = {f12_copy.to_latex()}")
print(f"f34_copy = {f34_copy.to_string()}")
print(f"f34_copy = {f34_copy.to_latex()}")

print("====================")

f12.optimize_recursive_repr = True
f34.optimize_recursive_repr = True

# # !Will cause RecursionError
# # try harder circular reference
# f12.denominator = f34
# f34.denominator = f12

# print(f"f12 = {f12.to_string()}")
# print(f"f12 = {f12.to_latex(auto_mode=True)}")
# print(f"f34 = {f34.to_string()}")
# print(f"f34 = {f34.to_latex(auto_mode=True)}")

# print(f"f12 = {f12.do_float()}")
# print(f"f34 = {f34.do_float()}")

# print(f"f12_simplify = {f12.simplify_without_change()}")

del f12, f34, f12_copy, f34_copy, p23, p34, r1

print("====================")

start = Integer(1)
res = ((start + 1) / 3) ** 2 + 3

print(res.to_string())
print(res.to_latex())

print(repr(Integer(Integer(2))))

print("====================")

f = Fraction(1, 2)
r = f * x + 1
r = r / 12 * f
latex_text = r.to_latex(fraction_manual_mode=LatexMode.FRAC)
print(r.to_string())
print(latex_text)

with BytesIO() as io_obj:
    latex_to_jpg(latex_text, io_obj)
    _, file_path = show_pic_on_browser(io_obj)

    input("Please check the picture and press any key to continue...")
    if file_path is not None:
        os.remove(file_path)

print("====================")

r = f * 102 // 2  # 51 // 2 = 25
print(r)
