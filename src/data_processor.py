# src/data_processing.py
import pandas as pd
import torch
from torchvision import transforms
from fastai.vision.all import *
from PIL import Image
import os

class FashionDataProcessor:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the fashion dataset CSV."""
        try:
            data_file = os.path.join(self.config['paths']['data_dir'], 'fashion_dataset.csv')
            df = pd.read_csv(data_file)
            print(f"Loaded {len(df)} items from dataset")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def prepare_fastai_data(self, df: pd.DataFrame) -> DataLoaders:
        """Prepare data for FastAI training."""
        try:
            dls = ImageDataLoaders.from_df(
                df,
                path=self.config['paths']['data_dir'],
                valid_pct=1 - self.config['data']['train_split'],
                seed=self.config['data']['random_seed'],
                img_size=self.config['data']['image_size'],
                batch_size=self.config['model']['batch_size']
            )
            return dls
        except Exception as e:
            print(f"Error preparing FastAI data: {e}")
            return None

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
