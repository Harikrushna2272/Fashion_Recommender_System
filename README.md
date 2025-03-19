# Fashion Recommendation System

A deep learning-based fashion recommendation system using ResNet and nearest neighbor search.

## Project Description

The Fashion Recommendation System is an AI-powered solution that helps users discover clothing items similar to their preferences. It uses deep learning and computer vision techniques to analyze fashion images and provide personalized recommendations based on visual similarity.

## Key-Features

- Visual similarity-based recommendations
- Deep learning feature extraction
- Fast nearest neighbor search
- Support for multiple categories of clothing
- Interactive user interface
- Real-time recommendations

# project structur

fashion-recommender/
│
├── data/               # Dataset storage
├── src/               # Source code
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks
├── requirements.txt   # Dependencies
└── README.md         # Documentation

## Usage:

- Training & Building Index:
Run python main.py --train to train the model and build the recommendation index.
- Recommendation via CLI:
Run python main.py --recommend path/to/sample.jpg to get recommendations for a given image.
- Interactive API:
Run uvicorn src.web_app:app --reload to start the REST API and access the recommendation endpoint.


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
