"""
A package for showing images.
"""

import base64
import tempfile
import webbrowser
from io import BytesIO
from PIL import Image

# pylint: disable=broad-exception-caught


HTML_BASE = """
<!DOCTYPE html>
<html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
        />
        <title>来自 Python 的图片</title>

        <style>
            body {{
                background-color: rgba(0, 100, 255, 0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 80vh;
                margin: 0;
            }}
            .main-container {{
                background-color: white;
                padding: 35px;
                border-radius: 20px;
                box-shadow: 10px 20px 30px rgba(0, 0, 255, 0.5);
                text-align: center;
            }}
            .pic-title {{
                display: flex;
                justify-content: center;
            }}
            .pic {{
                border-radius: 10px;
                border: 1px solid blue;
                margin-top: 20px;
            }}
            .button-container {{
                display: flex;
                justify-content: space-between;
                margin-top: 20px;
            }}
            .close-button {{
                background-color: white;
                color: red;
                border: 1px solid red;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
                font-size: 1.5em;
            }}
            .close-button:hover {{
                background-color: red;
                color: white;
            }}
            .save-button {{
                background-color: white;
                color: green;
                border: 1px solid green;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
                font-size: 1.5em;
            }}
            .save-button:hover {{
                background-color: green;
                color: white;
            }}
        </style>
    </head>

    <body>
        <div class="main-container">
            <h2 class="pic-title">
                来自 Python 的图片
            </h2>
            <p>{msg}</p>

            <img
                class="pic"
                src="data:{content_type};base64,{base64_message}"
            />

            <div class="button-container">
                <button
                    class="close-button"
                    onclick="window.close()"
                >
                    关闭窗口
                </button>
                <button
                    class="save-button"
                    onclick="saveAsImage()"
                >
                    保存图片
                </button>
            </div>
        </div>

        <script>
            function saveAsImage() {{
                var a = document.createElement("a");
                var img = document.querySelector(".pic");
                a.href = img.src;
                a.download = "image.{img_type}";
                a.click();
            }}
        </script>
    </body>
</html>
"""

type_map = {"JPEG": "image/jpeg", "JPG": "image/jpeg", "PNG": "image/png"}


def show_pic_on_browser(
    jpg: bytes | BytesIO, msg: str = ""
) -> tuple[bool, str | None]:
    """
    Show an image in a web browser.

    Args:
        jpg (bytes | BytesIO): The bytes of the image.
        msg (str, optional): The message to show. Defaults to "".

    Returns:
        tuple[bool, str | None]:
            A tuple containing two values:
             - A boolean value indicating whether the image
               is shown successfully.
             - The path of the temporary HTML file.
               (to be removed by the caller if necessary)
    """

    if isinstance(jpg, bytes):
        jpg = BytesIO(jpg)
    if not isinstance(jpg, BytesIO):
        raise TypeError("jpg must be bytes or BytesIO")
    jpg.seek(0)

    # Determine the image type
    img = Image.open(jpg)
    img_type = img.format
    if img_type is None:
        raise ValueError("Invalid image data")

    if img_type not in type_map:
        raise ValueError(f"Unsupported image type: {img_type}")

    content_type = type_map[img_type]

    # Encode the image data as a base64 string
    img_bytes = jpg.getvalue()
    base64_message = base64.b64encode(img_bytes).decode()

    # Determine the file suffix
    file_suffix = img_type.lower()

    # Create an HTML content containing the image
    html_content = HTML_BASE.format(
        msg=msg,
        content_type=content_type,
        base64_message=base64_message,
        img_type=file_suffix,
    )

    # Save the HTML content to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".html", delete=False
    ) as f:
        f.write(html_content)
        temp_file_path = f.name

    # Open the HTML file in the default web browser
    try:
        ret = webbrowser.open(f"file:///{temp_file_path}")
        return ret, temp_file_path
    except Exception as e:
        print(f"Failed to open the image in web browser: {e}")
        return False, None
