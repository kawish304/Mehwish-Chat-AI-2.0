from dotenv import load_dotenv
load_dotenv()  # Yeh alag line mein hona chahiye

import os
from pathlib import Path
from media_generator import MediaGenerator

def test_media_generator():
    # Set up static directory
    static_dir = Path("static")
    media_gen = MediaGenerator(static_dir)
    
    # Test image generation
    print("Testing image generation...")
    image_url = media_gen.generate_pixel_image("beautiful sunset")
    print(f"Generated Image URL: {image_url}")
    
    # Check if the file exists
    if image_url and image_url != "/static/placeholder.jpg":
        file_path = static_dir / image_url.lstrip("/static/")
        if file_path.exists():
            print("✅ Image successfully generated and saved!")
        else:
            print("❌ Image file not found.")
    else:
        print("❌ Image generation failed or returned placeholder.")

if __name__ == "__main__":
    test_media_generator()