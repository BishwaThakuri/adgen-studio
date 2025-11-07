from rembg import remove
from PIL import Image
import io

def remove_background(image: Image.Image) -> Image.Image:
    """
    Removes the background from a given PIL Image.

    Args:
        image: The input PIL Image object.

    Returns:
        A PIL Image object with the background removed (as a PNG).
    """
    
    # rembg's remove() function expects image bytes
    # We convert the PIL Image to bytes in memory
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0) # Go to the beginning of the byte stream
    
    # Call the remove function
    output_bytes = remove(img_byte_arr.read())
    
    # Convert the output bytes back to a PIL Image
    output_image = Image.open(io.BytesIO(output_bytes))
    
    return output_image