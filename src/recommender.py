import numpy as np
from annoy import AnnoyIndex
import pickle
from typing import List, Tuple

class FashionRecommender:
    def __init__(self, config):
        self.config = config
        self.embedding_size = config['model']['embedding_size']
        self.annoy_index = AnnoyIndex(self.embedding_size, 'angular')
        self.items_data = None

    def build_index(self, embeddings: np.ndarray, items_data: pd.DataFrame):
        """Build the similarity index"""
        try:
            for i, embedding in enumerate(embeddings):
                self.annoy_index.add_item(i, embedding)
            self.annoy_index.build(100)  # 100 trees
            self.items_data = items_data
            print("Similarity index built successfully")
        except Exception as e:
            print(f"Error building index: {str(e)}")

    def get_recommendations(self, query_embedding: np.ndarray, n_recommendations: int = 5) -> List[dict]:
        """Get recommendations for a query embedding"""
        try:
            similar_indices = self.annoy_index.get_nns_by_vector(
                query_embedding, n_recommendations + 1)[1:]
            recommendations = []
            for idx in similar_indices:
                item = self.items_data.iloc[idx]
                recommendations.append({
                    'id': idx,
                    'item': item['item_name'],
                    'category': item['category'],
                    'similarity_score': 1 - self.annoy_index.get_distance(0, idx)
                })
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

    def save_index(self):
        """Save the index and items data"""
        try:
            self.annoy_index.save(f"{self.config['paths']['processed_dir']}/fashion.ann")
            with open(f"{self.config['paths']['processed_dir']}/items_data.pkl", 'wb') as f:
                pickle.dump(self.items_data, f)
            print("Index saved successfully")
        except Exception as e:
            print(f"Error saving index: {str(e)}")

    def load_index(self):
        """Load the index and items data"""
        try:
            self.annoy_index.load(f"{self.config['paths']['processed_dir']}/fashion.ann")
            with open(f"{self.config['paths']['processed_dir']}/items_data.pkl", 'rb') as f:
                self.items_data = pickle.load(f)
            print("Index loaded successfully")
        except Exception as e:
            print(f"Error loading index: {str(e)}")
