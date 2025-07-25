import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import joblib
import os

# Conditional imports for image classification (if TensorFlow available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    from PIL import Image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ========== Load Data and Models ==========

@st.cache_data
def load_data():
    """Load the tree data from pickle file"""
    data_path = os.path.join(os.path.dirname(__file__), 'tree_data.pkl')
    if not os.path.exists(data_path):
        st.error(f"❌ Data file not found: {data_path}")
        st.info("💡 Please run the '5M_trees.ipynb' notebook first to generate the required data files.")
        st.stop()
    return pd.read_pickle(data_path)

@st.cache_resource
def load_nn_models():
    """Load the machine learning models"""
    base_dir = os.path.dirname(__file__)
    scaler_path = os.path.join(base_dir, 'scaler.joblib')
    model_path = os.path.join(base_dir, 'nn_model.joblib')
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        st.error("❌ ML model files not found!")
        st.info("💡 Please run the '5M_trees.ipynb' notebook first to generate the required model files.")
        st.stop()
        
    scaler = joblib.load(scaler_path)
    nn_model = joblib.load(model_path)
    return scaler, nn_model

@st.cache_resource
def load_cnn_model():
    """Load CNN model if available"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model_path = os.path.join(os.path.dirname(__file__), "basic_cnn_tree_species.h5")
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception as e:
            st.error(f"Error loading CNN model: {e}")
            return None
    else:
        return None

# ========== Utility Functions ==========

def recommend_species(input_data, nn_model, scaler, df, top_n=5):
    input_scaled = scaler.transform([input_data])
    distances, indices = nn_model.kneighbors(input_scaled)
    neighbors = df.iloc[indices[0]]
    species_counts = Counter(neighbors['common_name'])
    top_species = species_counts.most_common(top_n)
    return top_species

def get_common_locations_for_species(df, tree_name, top_n=10):
    species_df = df[df['common_name'] == tree_name]
    if species_df.empty:
        return pd.DataFrame(columns=['city', 'state', 'count'])
    location_counts = species_df.groupby(['city', 'state']) \
                                .size().reset_index(name='count') \
                                .sort_values(by='count', ascending=False) \
                                .head(top_n)
    return location_counts

# ========== Main App ==========

def main():
    st.title("🌿 Tree Intelligence Assistant")

    df = load_data()
    scaler, nn_model = load_nn_models()
    cnn_model = load_cnn_model()

    class_labels = sorted(df['common_name'].unique())

    # Determine available modes based on loaded models
    available_modes = [
        "🌲 Recommend Trees by Location",
        "📍 Find Locations for a Tree"
    ]
    
    if cnn_model is not None:
        available_modes.append("📷 Identify Tree from Image")
    
    mode = st.sidebar.radio("Choose Mode", available_modes)
    
    # Show info about image classification availability
    if cnn_model is None:
        st.sidebar.info("💡 Image classification feature will be available when CNN model is trained!")

    if mode == "🌲 Recommend Trees by Location":
        st.sidebar.header("Input Tree Features")
        lat = st.sidebar.number_input("Latitude", -90.0, 90.0, 38.2274, format="%.6f")
        lon = st.sidebar.number_input("Longitude", -180.0, 180.0, -85.8009, format="%.6f")
        diameter = st.sidebar.number_input("Diameter (cm)", 0.0, 1000.0, 1.0)

        native = st.sidebar.selectbox("Native Status", df['native'].astype('category').cat.categories)
        city = st.sidebar.selectbox("City", df['city'].astype('category').cat.categories)
        state = st.sidebar.selectbox("State", df['state'].astype('category').cat.categories)

        native_code = df['native'].astype('category').cat.categories.get_loc(native)
        city_code = df['city'].astype('category').cat.categories.get_loc(city)
        state_code = df['state'].astype('category').cat.categories.get_loc(state)

        input_data = [lat, lon, diameter, native_code, city_code, state_code]

        if st.button("Recommend Tree Species"):
            recommendations = recommend_species(input_data, nn_model, scaler, df, top_n=5)
            st.subheader("🌳 Top Tree Species in This Area:")
            for i, (species, count) in enumerate(recommendations, 1):
                st.write(f"{i}. {species} (seen {count} times nearby)")

    elif mode == "📍 Find Locations for a Tree":
        tree_name = st.sidebar.selectbox("Tree Species", sorted(df['common_name'].unique()))
        if st.button("Show Common Locations"):
            top_locations = get_common_locations_for_species(df, tree_name)
            if top_locations.empty:
                st.warning(f"No location data found for '{tree_name}'")
            else:
                st.subheader(f"📌 Top Locations for '{tree_name}':")
                st.dataframe(top_locations)

    elif mode == "📷 Identify Tree from Image":
        if cnn_model is None:
            st.error("❌ Image classification is not available. CNN model needs to be trained first.")
            st.info("💡 Run the 'tree_CNN.ipynb' notebook to train the image classification model.")
            return
            
        st.write("Upload a tree image to predict its species and see common locations.")
        uploaded_file = st.file_uploader("Choose a tree image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            if not TENSORFLOW_AVAILABLE:
                st.error("❌ TensorFlow is not available. Please install TensorFlow to use image classification.")
                return
                
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

            IMG_SIZE = (224, 224)
            img = image.resize(IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = cnn_model.predict(img_array)
            pred_idx = np.argmax(predictions)
            pred_label = class_labels[pred_idx]
            confidence = predictions[0][pred_idx]

            st.success(f"🌳 Predicted Tree Species: **{pred_label}**")
            st.write(f"🔍 Confidence: **{confidence:.2%}**")

            # Show top-3
            st.subheader("🔝 Top 3 Predictions:")
            top_3_idx = predictions[0].argsort()[-3:][::-1]
            for i in top_3_idx:
                st.write(f"{class_labels[i]} - {predictions[0][i]:.2%}")

            # Recommend locations
            st.subheader(f"📌 Common Locations for '{pred_label}'")
            location_info = get_common_locations_for_species(df, pred_label)
            if location_info.empty:
                st.warning("This species is not found in the dataset.")
            else:
                st.dataframe(location_info)

if __name__ == "__main__":
    main()

