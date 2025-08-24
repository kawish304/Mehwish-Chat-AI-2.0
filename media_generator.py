import os
import requests
from pathlib import Path
import uuid

class MediaGenerator:
    def __init__(self, static_dir):
        self.static_dir = static_dir
        self.media_dir = static_dir / "generated_media"
        self.media_dir.mkdir(exist_ok=True)
        self.api_key = os.getenv("PEXELS_API_KEY")
    
    def generate_pixel_image(self, prompt):
        if not self.api_key:
            print("❌ PEXELS_API_KEY not set in environment variables.")
            return "/static/placeholder.jpg"
        
        # Search for images using Pexels API
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": self.api_key}
        params = {"query": prompt, "per_page": 1}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['photos']:
                    photo = data['photos'][0]
                    image_url = photo['src']['original']
                    # Download the image
                    image_response = requests.get(image_url, timeout=10)
                    if image_response.status_code == 200:
                        filename = f"{uuid.uuid4()}.jpg"
                        filepath = self.media_dir / filename
                        with open(filepath, 'wb') as f:
                            f.write(image_response.content)
                        return f"/static/generated_media/{filename}"
                    else:
                        print(f"❌ Failed to download image: {image_response.status_code}")
                else:
                    print(f"❌ No photos found for prompt: {prompt}")
            else:
                print(f"❌ Pexels API error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error in generate_pixel_image: {e}")
        
        return "/static/placeholder.jpg"
    
    def generate_pixel_video(self, prompt):
        return "/static/placeholder.mp4"