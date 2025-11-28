import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import joblib
import os
import logging
import hashlib
from datetime import datetime
from pathlib import Path

# Conditional imports for image classification (if TensorFlow available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    from PIL import Image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import folium
    from streamlit_folium import folium_static
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    st.warning("⚠️ Visualization libraries not available. Install: pip install plotly folium streamlit-folium")

# ========== Logging Configuration ==========
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'app_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== Load Data and Models ==========

@st.cache_data
def load_data():
    """Load the tree data from pickle file"""
    data_path = os.path.join(os.path.dirname(__file__), 'tree_data.pkl')
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        st.error(f"❌ Data file not found: {data_path}")
        st.info("💡 Please run the '5M_trees.ipynb' notebook first to generate the required data files.")
        st.stop()
    
    logger.info(f"Loading tree data from {data_path}")
    data = pd.read_pickle(data_path)
    logger.info(f"Loaded {len(data)} tree records")
    return data

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
    
    # Try to load improved model first, then fall back to basic model
    improved_model_path = os.path.join(os.path.dirname(__file__), "improved_cnn_tree_species.h5")
    basic_model_path = os.path.join(os.path.dirname(__file__), "basic_cnn_tree_species.h5")
    
    model_path = improved_model_path if os.path.exists(improved_model_path) else basic_model_path
    
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            model_name = "Improved Transfer Learning" if "improved" in model_path else "Basic CNN"
            logger.info(f"Loaded {model_name} model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading CNN model: {e}")
            st.error(f"Error loading CNN model: {e}")
            return None
    else:
        return None

# ========== Security & Validation Functions ==========

def validate_image_upload(uploaded_file):
    """Validate uploaded image for security"""
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg']
    MIN_DIMENSION = 50
    MAX_DIMENSION = 4096
    
    try:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {uploaded_file.size} bytes")
            raise ValueError("File too large. Maximum size is 10MB.")
        
        # Check file type
        if uploaded_file.type not in ALLOWED_TYPES:
            logger.warning(f"Invalid file type: {uploaded_file.type}")
            raise ValueError("Invalid file type. Only JPG and PNG images are allowed.")
        
        # Verify it's a valid image and check dimensions
        img = Image.open(uploaded_file)
        width, height = img.size
        
        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            raise ValueError(f"Image too small. Minimum dimensions: {MIN_DIMENSION}x{MIN_DIMENSION}px")
        
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            raise ValueError(f"Image too large. Maximum dimensions: {MAX_DIMENSION}x{MAX_DIMENSION}px")
        
        # Verify image integrity
        img.verify()
        
        # Reset file pointer after verify
        uploaded_file.seek(0)
        
        logger.info(f"Image validated successfully: {width}x{height}px, {uploaded_file.size} bytes")
        return True
        
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise ValueError(f"Invalid or corrupted image: {str(e)}")

def get_image_hash(image):
    """Generate hash for image caching"""
    return hashlib.md5(image.tobytes()).hexdigest()

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
    # Page configuration
    st.set_page_config(
        page_title="Tree Species Classification",
        page_icon="🌳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🌿 Tree Species Classification & Intelligence Assistant")
    st.markdown("**AI-powered tree identification and location recommendations**")
    st.markdown("---")

    df = load_data()
    scaler, nn_model = load_nn_models()
    cnn_model = load_cnn_model()

    # Get class labels from dataset directory if available, otherwise from data
    dataset_path = os.path.join(os.path.dirname(__file__), 'Tree_Species_Dataset')
    if os.path.exists(dataset_path):
        class_labels = sorted([d for d in os.listdir(dataset_path) 
                              if os.path.isdir(os.path.join(dataset_path, d))])
    else:
        class_labels = sorted(df['common_name'].unique())

    # Add custom CSS for better metric visibility
    st.markdown("""
    <style>
    /* Main content metrics */
    [data-testid="stMetricValue"] {
        color: #0e1117 !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        color: #31333f !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    /* Sidebar metrics - different colors for dark background */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #fafafa !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar with app info
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg", width=60)
        st.title("Navigation")
        
        # Show statistics
        st.metric("Total Trees", f"{len(df):,}")
        st.metric("Unique Species", f"{df['common_name'].nunique():,}")
        st.metric("Cities Covered", f"{df['city'].nunique():,}")
        
        st.markdown("---")
    
    # Determine available modes based on loaded models
    available_modes = [
        "📊 Dashboard & Statistics",
        "🌲 Recommend Trees by Location",
        "📍 Find Locations for a Tree"
    ]
    
    if cnn_model is not None:
        available_modes.append("📷 Identify Tree from Image")
    
    mode = st.sidebar.radio("Choose Mode", available_modes)
    
    # Show info about image classification availability
    if cnn_model is None:
        st.sidebar.info("💡 Image classification feature will be available when CNN model is trained!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Quick Links")
    st.sidebar.markdown("[📖 Documentation](https://github.com/SatyamPandey-07/TREE_SPECIES_CLASSIFICATION)")
    st.sidebar.markdown("[🐛 Report Issue](https://github.com/SatyamPandey-07/TREE_SPECIES_CLASSIFICATION/issues)")
    st.sidebar.markdown("[⭐ Star on GitHub](https://github.com/SatyamPandey-07/TREE_SPECIES_CLASSIFICATION)")

    if mode == "📊 Dashboard & Statistics":
        st.header("📊 Dataset Statistics & Insights")
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌳 Total Trees", f"{len(df):,}")
        with col2:
            st.metric("🌿 Unique Species", f"{df['common_name'].nunique()}")
        with col3:
            st.metric("🏙️ Cities", f"{df['city'].nunique()}")
        with col4:
            st.metric("🗺️ States", f"{df['state'].nunique()}")
        
        st.markdown("---")
        
        if VISUALIZATION_AVAILABLE:
            # Top species chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top 15 Most Common Species")
                top_species = df['common_name'].value_counts().head(15)
                fig_species = px.bar(
                    x=top_species.values,
                    y=top_species.index,
                    orientation='h',
                    labels={'x': 'Number of Trees', 'y': 'Species'},
                    color=top_species.values,
                    color_continuous_scale='Greens'
                )
                fig_species.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig_species, use_container_width=True)
            
            with col2:
                st.subheader("🏙️ Top 15 Cities by Tree Count")
                top_cities = df['city'].value_counts().head(15)
                fig_cities = px.bar(
                    x=top_cities.values,
                    y=top_cities.index,
                    orientation='h',
                    labels={'x': 'Number of Trees', 'y': 'City'},
                    color=top_cities.values,
                    color_continuous_scale='Blues'
                )
                fig_cities.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig_cities, use_container_width=True)
            
            # Additional insights
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌱 Native vs Non-Native Distribution")
                if 'native' in df.columns:
                    native_counts = df['native'].value_counts()
                    fig_native = px.pie(
                        values=native_counts.values,
                        names=['Non-Native' if x == 0 else 'Native' for x in native_counts.index],
                        title='Tree Origin Distribution',
                        color_discrete_sequence=['#ff7f0e', '#2ca02c']
                    )
                    st.plotly_chart(fig_native, use_container_width=True)
            
            with col2:
                st.subheader("📏 Tree Diameter Distribution")
                if 'diameter_breast_height_CM' in df.columns:
                    fig_diameter = px.histogram(
                        df.sample(min(10000, len(df))),
                        x='diameter_breast_height_CM',
                        nbins=50,
                        title='Diameter Distribution (Sample)',
                        labels={'diameter_breast_height_CM': 'Diameter (cm)'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_diameter.update_layout(showlegend=False)
                    st.plotly_chart(fig_diameter, use_container_width=True)
            
            # Geographic distribution map
            if 'latitude_coordinate' in df.columns and 'longitude_coordinate' in df.columns:
                st.markdown("---")
                st.subheader("🗺️ Geographic Distribution (Sample)")
                
                # Sample for performance
                sample_df = df.sample(min(2000, len(df)))
                
                fig_map = px.scatter_mapbox(
                    sample_df,
                    lat='latitude_coordinate',
                    lon='longitude_coordinate',
                    color='city',
                    hover_name='common_name',
                    hover_data=['city', 'state'],
                    zoom=3,
                    height=500
                )
                fig_map.update_layout(mapbox_style="open-street-map")
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
        else:
            # Fallback without visualizations
            st.subheader("Top 10 Most Common Species")
            top_species = df['common_name'].value_counts().head(10)
            st.dataframe(top_species)
            
            st.subheader("Top 10 Cities")
            top_cities = df['city'].value_counts().head(10)
            st.dataframe(top_cities)
    
    elif mode == "🌲 Recommend Trees by Location":
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
                
                # Display data table
                st.dataframe(top_locations)
                
                # Visualizations
                if VISUALIZATION_AVAILABLE:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart of top locations
                        fig_bar = px.bar(
                            top_locations,
                            x='count',
                            y='city',
                            orientation='h',
                            title=f'{tree_name} Distribution by City',
                            labels={'count': 'Number of Trees', 'city': 'City'},
                            color='count',
                            color_continuous_scale='Greens'
                        )
                        fig_bar.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col2:
                        # Pie chart
                        fig_pie = px.pie(
                            top_locations,
                            values='count',
                            names='city',
                            title='Distribution Percentage'
                        )
                        fig_pie.update_layout(height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Interactive map
                    if 'latitude_coordinate' in df.columns and 'longitude_coordinate' in df.columns:
                        st.subheader("🗺️ Geographic Distribution Map")
                        
                        species_df = df[df['common_name'] == tree_name].copy()
                        
                        if len(species_df) > 0:
                            # Sample data for performance (max 1000 points)
                            if len(species_df) > 1000:
                                species_df = species_df.sample(1000)
                            
                            # Calculate center
                            center_lat = species_df['latitude_coordinate'].mean()
                            center_lon = species_df['longitude_coordinate'].mean()
                            
                            # Create map
                            m = folium.Map(
                                location=[center_lat, center_lon],
                                zoom_start=4,
                                tiles='OpenStreetMap'
                            )
                            
                            # Add markers in clusters
                            from folium.plugins import MarkerCluster
                            marker_cluster = MarkerCluster().add_to(m)
                            
                            for idx, row in species_df.iterrows():
                                folium.CircleMarker(
                                    location=[row['latitude_coordinate'], row['longitude_coordinate']],
                                    radius=3,
                                    color='green',
                                    fill=True,
                                    popup=f"{tree_name}<br>{row.get('city', 'Unknown')}, {row.get('state', '')}",
                                    tooltip=row.get('city', 'Unknown')
                                ).add_to(marker_cluster)
                            
                            folium_static(m, width=800, height=400)
                        else:
                            st.info("No location coordinates available for mapping")

    elif mode == "📷 Identify Tree from Image":
        if cnn_model is None:
            st.error("❌ Image classification is not available. CNN model needs to be trained first.")
            st.info("💡 Run the 'tree_CNN.ipynb' notebook to train the image classification model.")
            return
            
        st.write("Upload a tree image to predict its species and see common locations.")
        st.info("📏 Requirements: JPG/PNG, max 10MB, min 50x50px, max 4096x4096px")
        uploaded_file = st.file_uploader("Choose a tree image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            if not TENSORFLOW_AVAILABLE:
                logger.error("TensorFlow not available")
                st.error("❌ TensorFlow is not available. Please install TensorFlow to use image classification.")
                return
            
            # Validate image
            try:
                validate_image_upload(uploaded_file)
                logger.info(f"Processing uploaded file: {uploaded_file.name}")
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                logger.warning(f"Invalid upload: {str(e)}")
                return
                
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)

            IMG_SIZE = (224, 224)
            img = image.resize(IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            with st.spinner('🔍 Analyzing image...'):
                predictions = cnn_model.predict(img_array, verbose=0)
                pred_idx = np.argmax(predictions)
                pred_label = class_labels[pred_idx]
                confidence = predictions[0][pred_idx]
                
                logger.info(f"Prediction: {pred_label} (confidence: {confidence:.2%})")

            st.success(f"🌳 Predicted Tree Species: **{pred_label}**")
            st.write(f"🔍 Confidence: **{confidence:.2%}**")

            # Show top-3
            st.subheader("🔝 Top 3 Predictions:")
            top_3_idx = predictions[0].argsort()[-3:][::-1]
            for i in top_3_idx:
                if i < len(class_labels):
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

