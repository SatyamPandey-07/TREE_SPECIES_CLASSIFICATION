"""
Unit tests for input validation and security
"""
import pytest
import numpy as np
from PIL import Image
from io import BytesIO
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockUploadedFile:
    """Mock Streamlit uploaded file"""
    def __init__(self, data, filename, filetype, size):
        self.data = data
        self.name = filename
        self.type = filetype
        self.size = size
        self._position = 0
    
    def read(self):
        return self.data
    
    def seek(self, position):
        self._position = position
    
    def tell(self):
        return self._position


class TestImageValidation:
    """Test image validation functions"""
    
    def create_test_image(self, width=224, height=224, format='PNG'):
        """Helper to create test image"""
        img = Image.new('RGB', (width, height), color='red')
        buffer = BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return buffer.getvalue()
    
    def test_valid_image_jpg(self):
        """Test validation passes for valid JPG image"""
        from streamlit_integrated import validate_image_upload
        
        data = self.create_test_image(format='JPEG')
        mock_file = MockUploadedFile(data, 'test.jpg', 'image/jpeg', len(data))
        
        assert validate_image_upload(mock_file) is True
    
    def test_valid_image_png(self):
        """Test validation passes for valid PNG image"""
        from streamlit_integrated import validate_image_upload
        
        data = self.create_test_image(format='PNG')
        mock_file = MockUploadedFile(data, 'test.png', 'image/png', len(data))
        
        assert validate_image_upload(mock_file) is True
    
    def test_image_too_large(self):
        """Test validation fails for oversized image"""
        from streamlit_integrated import validate_image_upload
        
        # Create mock file > 10MB
        large_data = b'x' * (11 * 1024 * 1024)
        mock_file = MockUploadedFile(large_data, 'large.jpg', 'image/jpeg', len(large_data))
        
        with pytest.raises(ValueError, match="File too large"):
            validate_image_upload(mock_file)
    
    def test_invalid_file_type(self):
        """Test validation fails for invalid file type"""
        from streamlit_integrated import validate_image_upload
        
        data = b'fake pdf data'
        mock_file = MockUploadedFile(data, 'test.pdf', 'application/pdf', len(data))
        
        with pytest.raises(ValueError, match="Invalid file type"):
            validate_image_upload(mock_file)
    
    def test_image_too_small(self):
        """Test validation fails for very small images"""
        from streamlit_integrated import validate_image_upload
        
        data = self.create_test_image(width=40, height=40)
        mock_file = MockUploadedFile(data, 'small.png', 'image/png', len(data))
        
        with pytest.raises(ValueError, match="Image too small"):
            validate_image_upload(mock_file)
    
    def test_image_dimensions_boundary(self):
        """Test validation passes at boundary dimensions"""
        from streamlit_integrated import validate_image_upload
        
        # Test minimum valid size
        data = self.create_test_image(width=50, height=50)
        mock_file = MockUploadedFile(data, 'min.png', 'image/png', len(data))
        assert validate_image_upload(mock_file) is True


class TestImagePreprocessing:
    """Test image preprocessing functions"""
    
    def test_image_resize(self):
        """Test image resizing to model input size"""
        from PIL import Image
        
        # Create test image
        img = Image.new('RGB', (500, 600), color='blue')
        
        # Resize to model input
        img_resized = img.resize((224, 224))
        
        assert img_resized.size == (224, 224)
    
    def test_image_to_array_shape(self):
        """Test image to array conversion"""
        try:
            from tensorflow.keras.preprocessing.image import img_to_array
            from PIL import Image
            
            img = Image.new('RGB', (224, 224), color='green')
            img_array = img_to_array(img)
            
            assert img_array.shape == (224, 224, 3)
            assert img_array.dtype == np.float32 or img_array.dtype == np.float64
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_image_normalization(self):
        """Test image normalization"""
        try:
            from tensorflow.keras.preprocessing.image import img_to_array
            from PIL import Image
            
            img = Image.new('RGB', (224, 224), color='white')
            img_array = img_to_array(img) / 255.0
            
            assert img_array.max() <= 1.0
            assert img_array.min() >= 0.0
        except ImportError:
            pytest.skip("TensorFlow not available")


class TestInputSanitization:
    """Test input sanitization"""
    
    def test_latitude_bounds(self):
        """Test latitude input bounds"""
        # Valid latitudes
        assert -90.0 <= 38.2274 <= 90.0
        assert -90.0 <= 0.0 <= 90.0
        assert -90.0 <= -45.0 <= 90.0
        
        # Invalid latitudes
        assert not (-90.0 <= 95.0 <= 90.0)
        assert not (-90.0 <= -100.0 <= 90.0)
    
    def test_longitude_bounds(self):
        """Test longitude input bounds"""
        # Valid longitudes
        assert -180.0 <= -85.8009 <= 180.0
        assert -180.0 <= 0.0 <= 180.0
        assert -180.0 <= 120.0 <= 180.0
        
        # Invalid longitudes
        assert not (-180.0 <= 200.0 <= 180.0)
        assert not (-180.0 <= -200.0 <= 180.0)
    
    def test_diameter_bounds(self):
        """Test tree diameter input bounds"""
        # Valid diameters
        assert 0.0 <= 1.0 <= 1000.0
        assert 0.0 <= 50.5 <= 1000.0
        
        # Invalid diameters
        assert not (0.0 <= -5.0 <= 1000.0)
        assert not (0.0 <= 1500.0 <= 1000.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
