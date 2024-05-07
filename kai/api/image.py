import base64
from PIL import Image
from io import BytesIO

# convert image at path to base64
def image_to_base64(image: Image) -> str:
    with BytesIO() as buffered:
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


