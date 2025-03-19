# src/web_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import yaml
from src.data_processing import FashionDataProcessor
from src.model import FashionModel
from src.recommender import FashionRecommender

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="Fashion Recommender API")

class RecommendationRequest(BaseModel):
    image_path: str

# Initialize global components
data_processor = FashionDataProcessor(config)
model = FashionModel(config)
recommender = FashionRecommender(config)

@app.on_event("startup")
def load_assets():
    try:
        model.load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
    try:
        recommender.load_index()
    except Exception as e:
        print(f"Error loading index: {e}")

@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    img_tensor = data_processor.load_image(request.image_path)
    if img_tensor is None:
        raise HTTPException(status_code=400, detail="Invalid image path or unable to process image.")
    
    query_embedding = model.get_embeddings(img_tensor).cpu().numpy()
    recs = recommender.get_recommendations(query_embedding)
    if not recs:
        raise HTTPException(status_code=404, detail="No recommendations found.")
    
    return {"recommendations": recs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
