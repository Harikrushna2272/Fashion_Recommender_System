# src/model.py
import torch
import torch.nn as nn
from torchvision import models

class FashionModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()

    def _create_model(self) -> nn.Module:
        """Create and return the modified ResNet18 model."""
        try:
            if self.config['model']['name'] == 'resnet18':
                model = models.resnet18(pretrained=self.config['model']['pretrained'])
                model.fc = nn.Linear(model.fc.in_features, self.config['model']['embedding_size'])
            return model.to(self.device)
        except Exception as e:
            print(f"Error creating model: {e}")
            return None

    def get_embeddings(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Get embeddings for an image tensor."""
        try:
            self.model.eval()
            with torch.no_grad():
                return self.model(image_tensor.to(self.device))
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None

    def save_model(self):
        """Save the model's state dictionary."""
        try:
            torch.save(self.model.state_dict(), self.config['paths']['model_save_path'])
            print(f"Model saved to {self.config['paths']['model_save_path']}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Load the model's state dictionary."""
        try:
            self.model.load_state_dict(torch.load(self.config['paths']['model_save_path'], map_location=self.device))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
