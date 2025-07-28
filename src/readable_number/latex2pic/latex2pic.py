"""
A package for converting LaTeX equations to images.
"""

from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt

# Set the font
matplotlib.rcParams["mathtext.fontset"] = "stix"


def latex_to_jpg(
    latex_without_dollar_signs: str,
    io_obj: BytesIO,
    dpi: int = 300,
    *,
    facecolor: str = "white",
    pad_inches: float = 0.1,
) -> None:
    """
    Convert a LaTeX equation to a JPG image.

    Args:
        latex_without_dollar_signs (str): The LaTeX equation without
            the dollar signs.
        io_obj (BytesIO): A BytesIO object to store the image.
        dpi (int, optional): The DPI of the image.
        facecolor (str, optional): The facecolor of the image.
        pad_inches (float, optional): The padding of the image.
    """

    latex_text = r"$" + latex_without_dollar_signs + r"$"

    # Create a blank image, set the figsize to a small initial value
    fig, ax = plt.subplots(figsize=(2, 1), facecolor=facecolor)
    ax.axis("off")  # Hide the axis

    # Render the LaTeX text on the image
    ax.text(0.5, 0.5, latex_text, fontsize=30, ha="center", va="center")

    # Modify the figure size to fit the text
    tightbbox = fig.get_tightbbox(fig.canvas.get_renderer())  # type: ignore
    fig_width = tightbbox.width / fig.get_dpi()
    fig_height = tightbbox.height / fig.get_dpi()
    fig.set_size_inches(fig_width, fig_height)

    # Save the image to the BytesIO object
    fig.savefig(io_obj, bbox_inches="tight", pad_inches=pad_inches, dpi=dpi)
    io_obj.seek(0)


def latex_to_png(
    latex_without_dollar_signs: str,
    io_obj: BytesIO,
    dpi: int = 300,
    *,
    pad_inches: float = 0.1,
) -> None:
    """
    Convert a LaTeX equation to a PNG image.

    NOTE: This function actually calls the `latex_to_jpg` function with
        a facecolor of "none" to remove the background of the image.

    Args:
        latex_without_dollar_signs (str): The LaTeX equation without
            the dollar signs.
        io_obj (BytesIO): A BytesIO object to store the image.
        dpi (int, optional): The DPI of the image.
        pad_inches (float, optional): The padding of the image.
    """

    latex_to_jpg(
        latex_without_dollar_signs,
        io_obj,
        dpi,
        facecolor="none",
        pad_inches=pad_inches,
    )
