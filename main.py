# main.py
import yaml
import argparse
from src.data_processing import FashionDataProcessor
from src.model import FashionModel
from src.recommender import FashionRecommender
from fastai.vision.all import Learner, accuracy

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_model(config):
    """Train the fashion recommendation model."""
    data_processor = FashionDataProcessor(config)
    model = FashionModel(config)
    
    # Load and prepare data
    df = data_processor.load_data()
    if df is None:
        return None
    
    dls = data_processor.prepare_fastai_data(df)
    if dls is None:
        return None
    
    learn = Learner(dls, model.model, metrics=[accuracy])
    learn.fit_one_cycle(config['model']['num_epochs'], config['model']['learning_rate'])
    
    model.save_model()
    return model

def build_recommender(config, model):
    """Build the recommendation system using FLANN."""
    data_processor = FashionDataProcessor(config)
    recommender = FashionRecommender(config)
    
    df = data_processor.load_data()
    if df is None:
        return None
    
    # Generate embeddings for each image in the dataset
    embeddings = []
    for idx, row in df.iterrows():
        img_tensor = data_processor.load_image(row['image_path'])
        if img_tensor is not None:
            embedding = model.get_embeddings(img_tensor)
            embeddings.append(embedding.cpu().numpy())
    
    import numpy as np
    embeddings = np.vstack(embeddings)
    
    recommender.build_index(embeddings, df)
    recommender.save_index()
    return recommender

def get_recommendations(config, image_path):
    """Get recommendations for a given image."""
    data_processor = FashionDataProcessor(config)
    model = FashionModel(config)
    recommender = FashionRecommender(config)
    
    # Load model and index
    model.load_model()
    recommender.load_index()
    
    img_tensor = data_processor.load_image(image_path)
    if img_tensor is None:
        return None
    
    query_embedding = model.get_embeddings(img_tensor).cpu().numpy()
    recommendations = recommender.get_recommendations(query_embedding)
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='Fashion Recommendation System')
    parser.add_argument('--train', action='store_true', help='Train the model and build the recommendation index')
    parser.add_argument('--recommend', type=str, help='Get recommendations for an image (provide image path)')
    args = parser.parse_args()
    
    config = load_config()
    
    if args.train:
        model = train_model(config)
        if model is not None:
            build_recommender(config, model)
    
    if args.recommend:
        recs = get_recommendations(config, args.recommend)
        if recs:
            print("\nRecommendations:")
            for i, rec in enumerate(recs, 1):
                print(f"{i}. {rec['item']} ({rec['category']}) - Similarity: {rec['similarity_score']:.2f}")

if __name__ == "__main__":
    main()
