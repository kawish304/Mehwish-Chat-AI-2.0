import os, random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STATIC_DIR = Path("static/generated_media")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

class MediaGenerator:
    def __init__(self):
        self.pixel_api_key = os.getenv("PIXEL_API_KEY")

    def generate_pixel_image(self, prompt: str, style: str = "professional") -> str:
        """
        Pixel API (or fallback) image generator
        """
        try:
            if not self.pixel_api_key:
                raise ValueError("PIXEL_API_KEY missing, using fallback")

            filename = f"pixel_{random.randint(1000,9999)}.png"
            filepath = STATIC_DIR / filename
            filepath.write_bytes(b"fake image content")  # simulate API response

            return f"/static/generated_media/{filename}"

        except Exception as e:
            filename = f"fallback_{random.randint(1000,9999)}.png"
            filepath
