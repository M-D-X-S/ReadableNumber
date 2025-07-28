"""
A package for converting LaTeX equations to images.

This package uses the `matplotlib` library to render the LaTeX equations.
"""

from .latex2pic import latex_to_jpg, latex_to_png

__all__ = ["latex_to_jpg", "latex_to_png"]
