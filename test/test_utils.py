import pytest
import numpy as np
import cv2
import sys
import os
from unittest.mock import Mock, patch
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nnar.utils import (
    calculate_affine_transform,
    calculate_transformed_points, 
    calculate_predicted_position,
    landmark2numpy,
    numpy2landmark,
    convert_opencv_to_pixmap,
    convert_arrays_to_electrode_dict,
    apply_registration_to_atlas_points,
    update_mov_avg_buffer,
    apply_slider_adjustment,
    ATLAS_MAPPING,
    ELECTRODE_KEYS,
    ELECTRODE_LENGTHS
)

class TestMathematicalFunctions:
    """Test mathematical transformation functions"""
    
    def test_calculate_affine_transform(self):
        """Test affine transformation calculation"""
        # Create simple test points
        source_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        target_points = np.array([[1, 1, 0], [2, 1, 0], [1, 2, 0]])  # Translation by [1,1,0]
        
        A, b = calculate_affine_transform(source_points, target_points)
        
        # Verify transformation matrix is approximately identity
        assert A.shape == (3, 3)
        assert b.shape == (3, 1)
        
        # Test that transformation works correctly
        transformed = calculate_transformed_points(A, b, source_points)
        np.testing.assert_allclose(transformed, target_points, atol=1e-10)
    
    def test_calculate_transformed_points(self):
        """Test point transformation"""
        # Identity transformation
        A = np.eye(3)
        b = np.array([[1], [2], [3]])
        points = [[0, 0, 0], [1, 1, 1]]
        
        result = calculate_transformed_points(A, b, points)
        expected = np.array([[1, 2, 3], [2, 3, 4]])
        
        np.testing.assert_allclose(result, expected)
    
    def test_calculate_predicted_position(self):
        """Test ML prediction position calculation"""
        predictor = np.array([[1, 2, 3]])
        x_pa = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        
        result = calculate_predicted_position(predictor, x_pa)
        
        # Should return 3D coordinates
        assert result.shape == (3,)
        assert not np.isnan(result).any()


class TestDataConversion:
    """Test data conversion utilities"""
    
    def test_landmark2numpy_numpy2landmark_roundtrip(self):
        """Test conversion between numpy arrays and MediaPipe landmarks"""
        # Create mock MediaPipe landmark
        mock_landmark = Mock()
        mock_point1 = Mock()
        mock_point1.x, mock_point1.y, mock_point1.z = 0.1, 0.2, 0.3
        mock_point2 = Mock() 
        mock_point2.x, mock_point2.y, mock_point2.z = 0.4, 0.5, 0.6
        mock_landmark.landmark = [mock_point1, mock_point2]
        
        # Convert to numpy
        numpy_pts = landmark2numpy(mock_landmark)
        expected = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        np.testing.assert_allclose(numpy_pts, expected)
        
        # Convert back to landmarks
        landmark_result = numpy2landmark(numpy_pts)
        
        # Verify structure
        assert len(landmark_result.landmark) == 2
        assert abs(landmark_result.landmark[0].x - 0.1) < 1e-6
        assert abs(landmark_result.landmark[1].z - 0.6) < 1e-6
    
    def test_convert_arrays_to_electrode_dict(self):
        """Test conversion from array list to electrode dictionary"""
        # Create test data matching expected lengths
        total_length = sum(ELECTRODE_LENGTHS)
        test_data = list(range(total_length))
        
        result = convert_arrays_to_electrode_dict(test_data)
        
        # Verify all keys are present
        assert set(result.keys()) == set(ELECTRODE_KEYS)
        
        # Verify correct lengths
        for key, expected_length in zip(ELECTRODE_KEYS, ELECTRODE_LENGTHS):
            assert len(result[key]) == expected_length
        
        # Verify data continuity
        start_idx = 0
        for key, length in zip(ELECTRODE_KEYS, ELECTRODE_LENGTHS):
            expected_slice = test_data[start_idx:start_idx + length]
            assert result[key] == expected_slice
            start_idx += length

    @patch('PyQt5.QtGui.QPixmap')
    @patch('PyQt5.QtGui.QImage')
    def test_convert_opencv_to_pixmap(self, mock_qimage, mock_qpixmap):
        """Test OpenCV to QPixmap conversion"""
        # Create dummy BGR image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # Blue channel
        
        mock_qimg_instance = Mock()
        mock_qimage.return_value = mock_qimg_instance
        mock_qimg_instance.scaled.return_value = mock_qimg_instance
        
        mock_pixmap_instance = Mock()
        mock_qpixmap.fromImage.return_value = mock_pixmap_instance
        
        # Test without target size
        result = convert_opencv_to_pixmap(test_image)
        
        # Verify QImage was created with correct parameters
        mock_qimage.assert_called_once()
        mock_qpixmap.fromImage.assert_called_once_with(mock_qimg_instance)


class TestAtlasProcessing:
    """Test atlas and brain processing functions"""
    
    def test_apply_registration_to_atlas_points(self):
        """Test registration transformation application"""
        # Create test atlas data
        test_atlas = {
            'aal': [[0, 0, 0], [1, 0, 0]],
            'aar': [[0, 1, 0], [1, 1, 0]],
            'sm': [[0, 0, 1]] * 17  # SM needs 17 points
        }
        
        # Identity transformation with translation
        A = np.eye(3)
        b = np.array([[1], [1], [1]])
        
        result = apply_registration_to_atlas_points(A, b, test_atlas)
        
        # Verify all groups are processed
        assert 'aal' in result
        assert 'aar' in result
        assert 'sm' in result
        
        # Verify transformation applied correctly
        expected_aal = np.array([[1, 1, 1], [2, 1, 1]])
        np.testing.assert_allclose(result['aal'], expected_aal)
    
    def test_apply_slider_adjustment(self):
        """Test vertical position adjustment"""
        brain_data = {
            'aal': [[0, 0, 0], [1, 1, 1]],
            'aar': [[2, 2, 2]]
        }
        
        adjustment = 0.5
        result = apply_slider_adjustment(brain_data, adjustment)
        
        # Verify y-coordinates were adjusted
        expected_aal = [[0, 0.5, 0], [1, 1.5, 1]]
        expected_aar = [[2, 2.5, 2]]
        
        np.testing.assert_allclose(result['aal'], expected_aal)
        np.testing.assert_allclose(result['aar'], expected_aar)


class TestMovingAverage:
    """Test moving average buffer functionality"""
    
    def test_update_mov_avg_buffer_window_not_full(self):
        """Test moving average when window is not full"""
        data_list = []
        new_data = np.array([1, 2, 3, 4, 5])
        window_size = 3
        
        result, updated_list = update_mov_avg_buffer(data_list, new_data, window_size)
        
        # Should return None when window not full
        assert result is None
        assert len(updated_list) == 1
        np.testing.assert_array_equal(updated_list[0], new_data)
    
    def test_update_mov_avg_buffer_window_full(self):
        """Test moving average when window is full"""
        # Create test data that matches electrode array structure
        total_length = sum(ELECTRODE_LENGTHS)
        
        data_list = [
            np.arange(total_length),
            np.arange(total_length) + 1,
            np.arange(total_length) + 2
        ]
        new_data = np.arange(total_length) + 3
        window_size = 3
        
        result, updated_list = update_mov_avg_buffer(data_list, new_data, window_size)
        
        # Should return averaged result when window is full
        assert result is not None
        assert isinstance(result, dict)
        assert len(updated_list) == window_size
        
        # Verify it's properly structured as electrode dict
        assert set(result.keys()) == set(ELECTRODE_KEYS)
    
    def test_update_mov_avg_buffer_window_overflow(self):
        """Test moving average buffer maintains window size"""
        total_length = sum(ELECTRODE_LENGTHS)
        window_size = 2
        
        # Fill beyond window size
        data_list = [np.ones(total_length) * i for i in range(5)]
        new_data = np.ones(total_length) * 10
        
        result, updated_list = update_mov_avg_buffer(data_list, new_data, window_size)
        
        # Should maintain only window_size elements
        assert len(updated_list) == window_size
        assert result is not None


class TestConfiguration:
    """Test configuration constants and mappings"""
    
    def test_atlas_mapping_structure(self):
        """Test atlas mapping configuration"""
        assert isinstance(ATLAS_MAPPING, dict)
        assert len(ATLAS_MAPPING) > 0
        
        # Verify each mapping has correct structure
        for key, (atlas_file, atlas_5p_file) in ATLAS_MAPPING.items():
            assert isinstance(key, str)
            assert isinstance(atlas_file, str)
            assert atlas_file.endswith('.json')
            assert atlas_5p_file is None or atlas_5p_file.endswith('.json')
    
    def test_electrode_keys_lengths_consistency(self):
        """Test electrode keys and lengths are consistent"""
        assert len(ELECTRODE_KEYS) == len(ELECTRODE_LENGTHS)
        
        # Verify all are positive integers
        for length in ELECTRODE_LENGTHS:
            assert isinstance(length, int)
            assert length > 0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_data_handling(self):
        """Test functions handle empty data gracefully"""
        # Empty moving average
        result, updated = update_mov_avg_buffer([], np.array([]), 1)
        assert result is None
        
        # Empty slider adjustment
        result = apply_slider_adjustment({}, 0.5)
        assert result == {}
    
    def test_invalid_input_shapes(self):
        """Test handling of invalid input shapes"""
        with pytest.raises((ValueError, IndexError, TypeError)):
            calculate_affine_transform(
                np.array([[1, 2]]),  # Wrong shape (2D instead of 3D)
                np.array([[1, 2, 3]])
            )


# Fixtures for integration testing
@pytest.fixture
def sample_atlas_data():
    """Provide sample atlas data for testing"""
    return {
        'aal': [[0, 0, 0]] * 9,
        'aar': [[1, 1, 1]] * 9,
        'sm': [[0.5, 0.5, 0.5]] * 17,
        'cm': [[0, 1, 0]] * 17,
        'nz': [0, 0, 1],
        'lpa': [-1, 0, 0],
        'rpa': [1, 0, 0],
        'iz': [0, -1, 0],
        'cz': [0, 0, 0]
    }

@pytest.fixture 
def mock_mediapipe_results():
    """Provide mock MediaPipe results for testing"""
    mock_results = Mock()
    mock_results.face_landmarks = Mock()
    
    # Create mock landmarks
    landmarks = []
    for i in range(500):  # MediaPipe face has ~468 landmarks
        mock_point = Mock()
        mock_point.x = i * 0.001
        mock_point.y = i * 0.001 + 0.1
        mock_point.z = i * 0.001 + 0.2
        landmarks.append(mock_point)
    
    mock_results.face_landmarks.landmark = landmarks
    return mock_results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])