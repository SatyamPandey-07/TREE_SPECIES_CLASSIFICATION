"""
Unit tests for machine learning models
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestKNNModel:
    """Test KNN recommendation model"""
    
    def test_model_loading(self):
        """Test that KNN model can be loaded"""
        import joblib
        model_path = Path(__file__).parent.parent / 'nn_model.joblib'
        
        if model_path.exists():
            model = joblib.load(model_path)
            assert model is not None
            assert hasattr(model, 'kneighbors')
    
    def test_scaler_loading(self):
        """Test that scaler can be loaded"""
        import joblib
        scaler_path = Path(__file__).parent.parent / 'scaler.joblib'
        
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            assert scaler is not None
            assert hasattr(scaler, 'transform')
    
    def test_recommendation_shape(self):
        """Test that recommendations return correct shape"""
        import joblib
        from collections import Counter
        
        model_path = Path(__file__).parent.parent / 'nn_model.joblib'
        scaler_path = Path(__file__).parent.parent / 'scaler.joblib'
        
        if model_path.exists() and scaler_path.exists():
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Test input
            test_input = np.array([[38.0, -85.0, 20.0, 1, 0, 0]])
            scaled_input = scaler.transform(test_input)
            
            distances, indices = model.kneighbors(scaled_input)
            
            assert distances.shape[1] == 5  # Default k=5
            assert indices.shape[1] == 5


class TestCNNModel:
    """Test CNN image classification model"""
    
    def test_model_file_exists(self):
        """Test that model file exists"""
        model_path = Path(__file__).parent.parent / 'basic_cnn_tree_species.h5'
        assert model_path.exists(), "CNN model file not found"
    
    def test_model_loading(self):
        """Test that CNN model can be loaded"""
        try:
            import tensorflow as tf
            model_path = Path(__file__).parent.parent / 'basic_cnn_tree_species.h5'
            
            if model_path.exists():
                model = tf.keras.models.load_model(model_path)
                assert model is not None
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_model_input_shape(self):
        """Test model expects correct input shape"""
        try:
            import tensorflow as tf
            model_path = Path(__file__).parent.parent / 'basic_cnn_tree_species.h5'
            
            if model_path.exists():
                model = tf.keras.models.load_model(model_path)
                
                # Check input shape
                expected_shape = (None, 224, 224, 3)
                assert model.input_shape == expected_shape
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_model_output_shape(self):
        """Test model output has correct number of classes"""
        try:
            import tensorflow as tf
            model_path = Path(__file__).parent.parent / 'basic_cnn_tree_species.h5'
            
            if model_path.exists():
                model = tf.keras.models.load_model(model_path)
                
                # Test prediction
                test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
                predictions = model.predict(test_input, verbose=0)
                
                assert predictions.shape[0] == 1  # Batch size
                assert predictions.shape[1] == 30  # 30 classes
                assert np.isclose(predictions.sum(), 1.0, atol=1e-5)  # Probabilities sum to 1
        except ImportError:
            pytest.skip("TensorFlow not available")


class TestDataLoading:
    """Test data loading functions"""
    
    def test_tree_data_exists(self):
        """Test that tree data file exists"""
        data_path = Path(__file__).parent.parent / 'tree_data.pkl'
        assert data_path.exists(), "Tree data file not found"
    
    def test_tree_data_loading(self):
        """Test that tree data can be loaded"""
        data_path = Path(__file__).parent.parent / 'tree_data.pkl'
        
        if data_path.exists():
            df = pd.read_pickle(data_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_tree_data_columns(self):
        """Test that tree data has required columns"""
        data_path = Path(__file__).parent.parent / 'tree_data.pkl'
        
        if data_path.exists():
            df = pd.read_pickle(data_path)
            
            required_columns = ['common_name', 'latitude_coordinate', 
                              'longitude_coordinate', 'city', 'state']
            
            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
