import os
import pandas as pd
from PIL import Image
import argparse

def create_dataset_csv(image_dir, output_file):
    """Create CSV file from image directory"""
    data = []
    
    for category in os.listdir(image_dir):
        category_path = os.path.join(image_dir, category)
        if os.path.isdir(category_path):
            for brand in os.listdir(category_path):
                brand_path = os.path.join(category_path, brand)
                if os.path.isdir(brand_path):
                    for image_file in os.listdir(brand_path):
                        if image_file.endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(brand_path, image_file)
                            
                            # Get image dimensions
                            with Image.open(image_path) as img:
                                width, height = img.size
                            
                            # Create item name from filename
                            item_name = os.path.splitext(image_file)[0].replace('_', ' ').title()
                            
                            data.append({
                                'image_path': image_path,
                                'item_name': item_name,
                                'category': category,
                                'brand': brand,
                                'price': 0.0,  # You'll need to add real prices
                                'width': width,
                                'height': height
                            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Created dataset CSV with {len(df)} items")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create fashion dataset CSV')
    parser.add_argument('--image_dir', required=True, help='Directory containing fashion images')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    args = parser.parse_args()
    
    create_dataset_csv(args.image_dir, args.output)
