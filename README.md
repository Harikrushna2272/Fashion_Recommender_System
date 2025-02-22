# Fashion Recommendation System

A deep learning-based fashion recommendation system using ResNet and nearest neighbor search.

## Features

- Deep learning-based image feature extraction
- Fast nearest neighbor search for recommendations
- Support for custom fashion datasets
- Easy to train and deploy

# project structur

fashion-recommender/
│
├── data/               # Dataset storage
├── src/               # Source code
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks
├── requirements.txt   # Dependencies
└── README.md         # Documentation



## Installation

```bash
# Clone the repository
git clone https://github.com/Harikrushna2272/FashionistaRecommender.git
cd fashion-recommender

# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows
venv\Scripts\activate
# For Unix or MacOS
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# To prepare the dataset:
python scripts/prepare_dataset.py --image_dir data/raw/images --output data/raw/fashion_dataset.csv



# FashionistaRecommender
