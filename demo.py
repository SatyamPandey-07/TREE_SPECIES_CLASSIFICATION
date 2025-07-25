#!/usr/bin/env python3
"""
Tree Species Classification Demo Script

This script demonstrates the key features of the Tree Species Classification system
without requiring the full Streamlit interface. Perfect for quick testing and
showcasing the machine learning capabilities.

Usage:
    python demo.py

Author: Satyam Pandey
"""

import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_models():
    """Load all trained models and data"""
    try:
        print("🔄 Loading models and data...")
        
        # Load tree data
        with open('tree_data.pkl', 'rb') as f:
            tree_data = pickle.load(f)
        print(f"   ✅ Tree data loaded: {len(tree_data):,} records")
        
        # Load scaler and KNN model
        scaler = joblib.load('scaler.joblib')
        nn_model = joblib.load('nn_model.joblib')
        print("   ✅ KNN recommender model loaded")
        
        # Check for CNN model
        cnn_available = os.path.exists('basic_cnn_tree_species.h5')
        if cnn_available:
            print("   ✅ CNN model available")
        else:
            print("   ⚠️  CNN model not found (optional for this demo)")
        
        return tree_data, scaler, nn_model, cnn_available
    
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return None, None, None, False

def demo_location_recommendation(tree_data, scaler, nn_model):
    """Demonstrate location-based tree recommendations"""
    print("\n🌲 DEMO 1: Location-Based Tree Recommendations")
    print("=" * 50)
    
    # Example location: Louisville, KY
    latitude = 38.2527
    longitude = -85.7585
    diameter = 25.4
    native_status = 1  # 1 for native, 0 for non-native
    city = "Louisville"
    state = "Kentucky"
    
    print(f"📍 Location: {city}, {state}")
    print(f"   Coordinates: ({latitude}, {longitude})")
    print(f"   Tree diameter: {diameter} cm")
    print(f"   Native status: {'Native' if native_status else 'Non-native'}")
    
    try:
        # Prepare features for prediction
        features = np.array([[latitude, longitude, diameter, native_status]])
        features_scaled = scaler.transform(features)
        
        # Get recommendations
        distances, indices = nn_model.kneighbors(features_scaled, n_neighbors=5)
        
        print(f"\n🌳 Top 5 Recommended Tree Species:")
        print("-" * 40)
        
        for i, idx in enumerate(indices[0]):
            tree_info = tree_data.iloc[idx]
            distance = distances[0][i]
            confidence = max(0, 100 - (distance * 50))  # Simple confidence calculation
            
            print(f"{i+1}. {tree_info['common_name']}")
            print(f"   Scientific: {tree_info.get('scientific_name', 'N/A')}")
            print(f"   Confidence: {confidence:.1f}%")
            print(f"   Location: {tree_info.get('city', 'N/A')}, {tree_info.get('state', 'N/A')}")
            print()
    
    except Exception as e:
        print(f"   ❌ Error in recommendation: {e}")

def demo_species_distribution(tree_data):
    """Demonstrate species distribution analysis"""
    print("\n📍 DEMO 2: Species Distribution Analysis")
    print("=" * 50)
    
    # Example species
    species_name = "Red Oak"
    print(f"🔍 Analyzing distribution for: {species_name}")
    
    try:
        # Filter data for the species
        species_data = tree_data[tree_data['common_name'].str.contains(species_name, case=False, na=False)]
        
        if len(species_data) > 0:
            print(f"\n📊 Found {len(species_data):,} {species_name} trees")
            print("-" * 40)
            
            # City distribution
            city_counts = species_data['city'].value_counts().head(10)
            print("Top 10 cities with most trees:")
            for i, (city, count) in enumerate(city_counts.items(), 1):
                state = species_data[species_data['city'] == city]['state'].iloc[0] if not species_data[species_data['city'] == city]['state'].empty else 'Unknown'
                print(f"{i:2d}. {city}, {state}: {count:,} trees")
            
            # Geographic spread
            if 'latitude_coordinate' in species_data.columns and 'longitude_coordinate' in species_data.columns:
                lat_range = (species_data['latitude_coordinate'].min(), species_data['latitude_coordinate'].max())
                lon_range = (species_data['longitude_coordinate'].min(), species_data['longitude_coordinate'].max())
                print(f"\n🗺️  Geographic Range:")
                print(f"   Latitude: {lat_range[0]:.2f}° to {lat_range[1]:.2f}°")
                print(f"   Longitude: {lon_range[0]:.2f}° to {lon_range[1]:.2f}°")
        else:
            print(f"   ❌ No data found for {species_name}")
    
    except Exception as e:
        print(f"   ❌ Error in distribution analysis: {e}")

def demo_dataset_overview(tree_data):
    """Show overview of the dataset"""
    print("\n📊 DEMO 3: Dataset Overview")
    print("=" * 50)
    
    try:
        print(f"🗃️  Total Records: {len(tree_data):,}")
        
        # Species diversity
        if 'common_name' in tree_data.columns:
            unique_species = tree_data['common_name'].nunique()
            print(f"🌿 Unique Species: {unique_species:,}")
            
            # Top species
            top_species = tree_data['common_name'].value_counts().head(5)
            print(f"\n🏆 Top 5 Most Common Species:")
            for i, (species, count) in enumerate(top_species.items(), 1):
                percentage = (count / len(tree_data)) * 100
                print(f"{i}. {species}: {count:,} trees ({percentage:.1f}%)")
        
        # Geographic coverage
        if 'city' in tree_data.columns:
            unique_cities = tree_data['city'].nunique()
            print(f"\n🏙️  Cities Covered: {unique_cities:,}")
            
            top_cities = tree_data['city'].value_counts().head(5)
            print(f"\n🌆 Top 5 Cities by Tree Count:")
            for i, (city, count) in enumerate(top_cities.items(), 1):
                print(f"{i}. {city}: {count:,} trees")
        
        # Native vs Non-native
        if 'native' in tree_data.columns:
            native_counts = tree_data['native'].value_counts()
            print(f"\n🌱 Native vs Non-native Distribution:")
            for status, count in native_counts.items():
                status_label = "Native" if status == 1 else "Non-native"
                percentage = (count / len(tree_data)) * 100
                print(f"   {status_label}: {count:,} trees ({percentage:.1f}%)")
    
    except Exception as e:
        print(f"   ❌ Error in dataset overview: {e}")

def demo_cnn_info(cnn_available):
    """Show CNN model information"""
    print("\n📷 DEMO 4: CNN Image Classification")
    print("=" * 50)
    
    if cnn_available:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model('basic_cnn_tree_species.h5')
            
            print("🧠 CNN Model Information:")
            print(f"   Model type: Sequential CNN")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output classes: {model.output_shape[-1]}")
            print(f"   Total parameters: {model.count_params():,}")
            
            # Model size
            file_size = os.path.getsize('basic_cnn_tree_species.h5') / (1024 * 1024)
            print(f"   Model file size: {file_size:.1f} MB")
            
            print("\n📸 Image Classification Features:")
            print("   • Accepts 224×224 RGB images")
            print("   • Predicts from 30 tree species")
            print("   • Confidence scoring included")
            print("   • Real-time inference capability")
            
        except ImportError:
            print("⚠️  TensorFlow not available for CNN demonstration")
        except Exception as e:
            print(f"   ❌ Error loading CNN model: {e}")
    else:
        print("⚠️  CNN model not found. Train using tree_CNN.ipynb")
        print("\n🎯 To enable image classification:")
        print("   1. Run: jupyter notebook tree_CNN.ipynb")
        print("   2. Execute all cells to train the CNN")
        print("   3. The model will be saved as basic_cnn_tree_species.h5")

def main():
    """Main demo function"""
    print("🌳 Tree Species Classification Demo")
    print("=" * 60)
    print("Demonstrating AI-powered tree identification and recommendations")
    print("=" * 60)
    
    # Load models and data
    tree_data, scaler, nn_model, cnn_available = load_models()
    
    if tree_data is None:
        print("❌ Failed to load required models. Please ensure:")
        print("   • tree_data.pkl exists")
        print("   • scaler.joblib exists") 
        print("   • nn_model.joblib exists")
        print("\n💡 Run 5M_trees.ipynb to generate these files")
        return
    
    # Run demonstrations
    demo_location_recommendation(tree_data, scaler, nn_model)
    demo_species_distribution(tree_data)
    demo_dataset_overview(tree_data)
    demo_cnn_info(cnn_available)
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("\n🚀 To run the full web application:")
    print("   streamlit run streamlit_integrated.py")
    print("\n📚 For more information, see README.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
