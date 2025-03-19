# **👗 Fashion Recommendation System**  
A **Deep Learning-based Fashion Recommendation System** that utilizes a **ResNet18 model** for feature extraction and **FLANN (Fast Library for Approximate Nearest Neighbors)** for similarity search. This system enables personalized fashion recommendations based on image similarity.

---

## **🚀 Features**
✅ Uses a **pretrained ResNet18 model** for feature extraction.  
✅ Implements **FLANN** for fast and efficient similarity search.  
✅ Provides a **REST API** with FastAPI for interactive recommendations.  
✅ Supports **cosine similarity re-ranking** for better recommendations.  
✅ Fully **configurable** via `config.yaml`.  
✅ **Modular and scalable** design.  

---

## **👤 Project Structure**
```
Fashion_Recommender_System/
├── config.yaml                # Configuration file
├── main.py                    # Main script for training and recommendations
├── requirements.txt            # Required dependencies
└── src/
    ├── model.py                # ResNet18 feature extractor
    ├── data_processing.py       # Data processing utilities
    ├── recommender.py           # FLANN-based recommendation engine
    └── web_app.py               # FastAPI-based REST API
```

---

## **🛠️ Installation**
### **1️⃣ Clone the repository**  
```bash
git clone https://github.com/Harikrushna2272/Fashion_Recommender_System.git
cd Fashion_Recommender_System
```
  
### **2️⃣ Create a virtual environment (Recommended)**  
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```
  
### **3️⃣ Install dependencies**  
```bash
pip install -r requirements.txt
```

---

## **📂 Dataset Format**
The system expects a **CSV file (`fashion_dataset.csv`)** in the dataset directory (`data/images`).  
Example CSV structure:
```csv
item_name,category,image_path
"Red Dress","Dresses","data/images/red_dress.jpg"
"Blue Jeans","Bottoms","data/images/blue_jeans.jpg"
"White T-Shirt","Tops","data/images/white_tshirt.jpg"
"Black Shoes","Footwear","data/images/black_shoes.jpg"
```
- **item_name** → Name of the fashion item.  
- **category** → Category (e.g., Dresses, Tops, Bottoms, Footwear).  
- **image_path** → Relative path to the image.  

---

## **⚙️ Usage**

### **1️⃣ Train the Model & Build the FLANN Index**
```bash
python main.py --train
```
- Trains a **ResNet18** model on the dataset.
- Saves the trained model in **models/resnet18_fashion.pth**.
- Extracts **image embeddings** and builds a **FLANN index**.

---

### **2️⃣ Get Recommendations via CLI**
```bash
python main.py --recommend "data/sample.jpg"
```
- Extracts features from `sample.jpg`.
- Finds **top 5 similar images** using FLANN.
- Prints recommended items.

---

### **3️⃣ Start the FastAPI Server**
```bash
uvicorn src.web_app:app --reload
```
- Runs a **REST API** for real-time recommendations.
- API is accessible at:  
  **http://127.0.0.1:8000/docs** (Swagger UI)

---

### **4️⃣ Get Recommendations via API**
Send a **POST request** to `/recommend`:
```json
{
  "image_path": "data/sample.jpg"
}
```
Response:
```json
{
  "recommendations": [
    {"item": "Blue Jeans", "category": "Bottoms", "similarity_score": 0.89},
    {"item": "Black Shoes", "category": "Footwear", "similarity_score": 0.85}
  ]
}
```

