# src/recommender.py
import cv2
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class FashionRecommender:
    def __init__(self, config):
        self.config = config
        self.embedding_size = config['model']['embedding_size']
        # FLANN parameters: using KDTree (algorithm=1)
        self.index_params = dict(algorithm=1, trees=5)
        self.search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.items_data = None
        self.embeddings = None

    def build_index(self, embeddings: np.ndarray, items_data: pd.DataFrame):
        """Build the similarity index using FLANN."""
        try:
            self.embeddings = embeddings.astype(np.float32)
            # FLANN requires a list of arrays; we pass the embeddings array as one dataset
            self.flann.add([self.embeddings])
            self.flann.train()
            self.items_data = items_data.reset_index(drop=True)
            print("FLANN index built successfully.")
        except Exception as e:
            print(f"Error building FLANN index: {e}")

    def get_recommendations(self, query_embedding: np.ndarray, n_recommendations: int = 5) -> List[Dict]:
        """Get recommendations for a query embedding using FLANN."""
        try:
            query_embedding = query_embedding.astype(np.float32)
            matches = self.flann.knnMatch(query_embedding, self.embeddings, k=n_recommendations)
            recommendations = []
            # Re-rank using cosine similarity if desired
            candidate_embeddings = np.array([self.embeddings[m.trainIdx] for m in matches[0]])
            cos_sim = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()
            sorted_indices = np.argsort(-cos_sim)
            for idx in sorted_indices:
                actual_index = matches[0][idx].trainIdx
                item = self.items_data.iloc[actual_index]
                recommendations.append({
                    'item': item['item_name'],
                    'category': item['category'],
                    'similarity_score': cos_sim[idx]
                })
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    def save_index(self):
        """Save the FLANN index and associated items data."""
        try:
            index_path = self.config['paths']['index_path']
            with open(index_path, 'wb') as f:
                pickle.dump((self.flann, self.items_data, self.embeddings), f)
            print("FLANN index saved successfully.")
        except Exception as e:
            print(f"Error saving FLANN index: {e}")

    def load_index(self):
        """Load the FLANN index and associated items data."""
        try:
            index_path = self.config['paths']['index_path']
            with open(index_path, 'rb') as f:
                self.flann, self.items_data, self.embeddings = pickle.load(f)
            print("FLANN index loaded successfully.")
        except Exception as e:
            print(f"Error loading FLANN index: {e}")
