"""
FastAPI REST API for Tree Species Classification
Provides endpoints for predictions and tree recommendations
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import logging
from PIL import Image
from io import BytesIO
import os
from pathlib import Path

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tree Species Classification API",
    description="AI-powered tree identification and location recommendations",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Load Models and Data ==========

BASE_DIR = Path(__file__).parent

try:
    df = pd.read_pickle(BASE_DIR / 'tree_data.pkl')
    scaler = joblib.load(BASE_DIR / 'scaler.joblib')
    nn_model = joblib.load(BASE_DIR / 'nn_model.joblib')
    logger.info("✅ KNN model and data loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading KNN model: {e}")
    df, scaler, nn_model = None, None, None

try:
    import tensorflow as tf
    cnn_model = tf.keras.models.load_model(BASE_DIR / 'basic_cnn_tree_species.h5')
    
    # Load class labels
    dataset_path = BASE_DIR / 'Tree_Species_Dataset'
    if dataset_path.exists():
        class_labels = sorted([d for d in os.listdir(dataset_path) 
                              if os.path.isdir(dataset_path / d)])
    else:
        class_labels = sorted(df['common_name'].unique()) if df is not None else []
    
    logger.info(f"✅ CNN model loaded successfully with {len(class_labels)} classes")
except Exception as e:
    logger.error(f"❌ Error loading CNN model: {e}")
    cnn_model = None
    class_labels = []

# ========== Request/Response Models ==========

class LocationRequest(BaseModel):
    """Request model for location-based recommendations"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    diameter: float = Field(..., ge=0, le=1000, description="Tree diameter in cm")
    native: int = Field(..., ge=0, le=1, description="Native status (0 or 1)")
    city: str = Field(..., description="City name")
    state: str = Field(..., description="State name")
    top_n: int = Field(5, ge=1, le=20, description="Number of recommendations")

class TreeRecommendation(BaseModel):
    """Response model for tree recommendation"""
    species: str
    count: int
    rank: int

class LocationRecommendationResponse(BaseModel):
    """Response model for location-based recommendations"""
    recommendations: List[TreeRecommendation]
    input_location: dict

class SpeciesLocationsRequest(BaseModel):
    """Request model for species location query"""
    species_name: str = Field(..., description="Tree species name")
    top_n: int = Field(10, ge=1, le=50, description="Number of top locations")

class LocationInfo(BaseModel):
    """Location information model"""
    city: str
    state: str
    count: int

class SpeciesLocationsResponse(BaseModel):
    """Response model for species locations"""
    species: str
    locations: List[LocationInfo]
    total_trees: int

class PredictionResult(BaseModel):
    """Single prediction result"""
    species: str
    confidence: float
    rank: int

class ImagePredictionResponse(BaseModel):
    """Response model for image prediction"""
    predicted_species: str
    confidence: float
    top_predictions: List[PredictionResult]

class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    knn_model_loaded: bool
    cnn_model_loaded: bool
    data_loaded: bool
    total_tree_records: int
    supported_species: int

# ========== API Endpoints ==========

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🌳 Tree Species Classification API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/api/recommend",
            "locations": "/api/locations",
            "predict": "/api/predict"
        },
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy" if all([df is not None, nn_model is not None, scaler is not None]) else "degraded",
        knn_model_loaded=nn_model is not None,
        cnn_model_loaded=cnn_model is not None,
        data_loaded=df is not None,
        total_tree_records=len(df) if df is not None else 0,
        supported_species=len(class_labels)
    )

@app.post("/api/recommend", response_model=LocationRecommendationResponse)
async def recommend_trees(request: LocationRequest):
    """
    Get tree species recommendations based on location and environmental factors
    """
    if df is None or nn_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Encode categorical variables
        native_code = request.native
        
        # Find city and state codes
        cities = df['city'].astype('category').cat.categories
        states = df['state'].astype('category').cat.categories
        
        if request.city not in cities or request.state not in states:
            raise HTTPException(status_code=400, detail="Invalid city or state")
        
        city_code = cities.get_loc(request.city)
        state_code = states.get_loc(request.state)
        
        # Prepare input
        input_data = np.array([[
            request.latitude,
            request.longitude,
            request.diameter,
            native_code,
            city_code,
            state_code
        ]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        distances, indices = nn_model.kneighbors(input_scaled, n_neighbors=request.top_n)
        
        # Get recommendations
        neighbors = df.iloc[indices[0]]
        from collections import Counter
        species_counts = Counter(neighbors['common_name'])
        
        recommendations = [
            TreeRecommendation(
                species=species,
                count=count,
                rank=i+1
            )
            for i, (species, count) in enumerate(species_counts.most_common(request.top_n))
        ]
        
        logger.info(f"Recommendations generated for location: ({request.latitude}, {request.longitude})")
        
        return LocationRecommendationResponse(
            recommendations=recommendations,
            input_location={
                "latitude": request.latitude,
                "longitude": request.longitude,
                "city": request.city,
                "state": request.state
            }
        )
        
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/locations", response_model=SpeciesLocationsResponse)
async def get_species_locations(request: SpeciesLocationsRequest):
    """
    Get top locations where a specific tree species is found
    """
    if df is None:
        raise HTTPException(status_code=503, detail="Data not available")
    
    try:
        # Filter by species
        species_df = df[df['common_name'] == request.species_name]
        
        if species_df.empty:
            raise HTTPException(status_code=404, detail=f"Species '{request.species_name}' not found")
        
        # Group by location
        location_counts = species_df.groupby(['city', 'state']) \
                                   .size() \
                                   .reset_index(name='count') \
                                   .sort_values(by='count', ascending=False) \
                                   .head(request.top_n)
        
        locations = [
            LocationInfo(
                city=row['city'],
                state=row['state'],
                count=int(row['count'])
            )
            for _, row in location_counts.iterrows()
        ]
        
        logger.info(f"Locations retrieved for species: {request.species_name}")
        
        return SpeciesLocationsResponse(
            species=request.species_name,
            locations=locations,
            total_trees=len(species_df)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in locations endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict", response_model=ImagePredictionResponse)
async def predict_tree_species(
    file: UploadFile = File(..., description="Tree image (JPG/PNG)")
):
    """
    Predict tree species from uploaded image
    """
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="CNN model not available")
    
    # Validate file type
    if file.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG/PNG allowed")
    
    try:
        # Read and process image
        contents = await file.read()
        
        # Check file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum 10MB")
        
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        # Resize and normalize
        img_resized = image.resize((224, 224))
        
        import tensorflow as tf
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = cnn_model.predict(img_array, verbose=0)
        pred_idx = np.argmax(predictions[0])
        
        # Get top 3 predictions
        top_3_idx = predictions[0].argsort()[-3:][::-1]
        
        top_predictions = [
            PredictionResult(
                species=class_labels[i] if i < len(class_labels) else f"Class_{i}",
                confidence=float(predictions[0][i]),
                rank=rank+1
            )
            for rank, i in enumerate(top_3_idx)
            if i < len(class_labels)
        ]
        
        logger.info(f"Prediction made: {class_labels[pred_idx]} ({predictions[0][pred_idx]:.2%})")
        
        return ImagePredictionResponse(
            predicted_species=class_labels[pred_idx] if pred_idx < len(class_labels) else f"Class_{pred_idx}",
            confidence=float(predictions[0][pred_idx]),
            top_predictions=top_predictions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/species", response_model=List[str])
async def list_species():
    """Get list of all supported tree species"""
    if not class_labels:
        raise HTTPException(status_code=503, detail="Species data not available")
    
    return class_labels

@app.get("/api/stats", response_model=dict)
async def get_statistics():
    """Get dataset statistics"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not available")
    
    try:
        stats = {
            "total_trees": len(df),
            "unique_species": df['common_name'].nunique(),
            "unique_cities": df['city'].nunique(),
            "unique_states": df['state'].nunique(),
            "top_species": df['common_name'].value_counts().head(10).to_dict(),
            "top_cities": df['city'].value_counts().head(10).to_dict()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
