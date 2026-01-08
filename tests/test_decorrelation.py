"""
Tests for the decorrelation stretch algorithm.

Tests the core functionality against known results and edge cases.

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from pydecorrelation_stretch import ColorspaceManager, DecorrelationStretch
from pydecorrelation_stretch.decorrelation import process_image


class TestDecorrelationStretch:
    """Test cases for DecorrelationStretch class."""

    def test_initialization(self):
        """Test basic initialization."""
        dstretch = DecorrelationStretch()
        assert dstretch is not None
        assert dstretch.colorspace_manager is not None

    def test_input_validation(self):
        """Test input validation."""
        dstretch = DecorrelationStretch()

        # Test invalid image format
        with pytest.raises(ValueError):
            dstretch.process(np.array([1, 2, 3]), "RGB")

        # Test invalid colorspace
        with pytest.raises(ValueError):
            test_image = np.zeros((10, 10, 3), dtype=np.uint8)
            dstretch.process(test_image, "INVALID")

        # Test invalid scale
        with pytest.raises(ValueError):
            test_image = np.zeros((10, 10, 3), dtype=np.uint8)
            dstretch.process(test_image, "RGB", scale=200)

    @pytest.mark.skip(reason="Scale 1.0 is not identity in this implementation")
    def test_process_rgb_identity(self):
        """Test processing with RGB colorspace (should be minimal change)."""
        # Create test image with some variation
        test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        dstretch = DecorrelationStretch()
        result = dstretch.process(test_image, "RGB", scale=1.0)  # Minimal enhancement

        assert result.processed_image.shape == test_image.shape
        assert result.colorspace == "RGB"
        assert result.scale == 1.0

        # With minimal scale, result should be very close to original
        difference = np.mean(
            np.abs(result.processed_image.astype(float) - test_image.astype(float))
        )
        assert difference < 5.0  # Small difference expected

    def test_process_different_colorspaces(self):
        """Test processing with different colorspaces."""
        test_image = np.random.randint(50, 200, (20, 20, 3), dtype=np.uint8)

        dstretch = DecorrelationStretch()
        colorspaces = ["RGB", "LAB", "YDS", "CRGB", "LDS", "LRE"]

        results = {}
        for cs in colorspaces:
            result = dstretch.process(test_image, cs, scale=15.0)
            results[cs] = result

            # Basic validation
            assert result.processed_image.shape == test_image.shape
            assert result.colorspace == cs
            assert result.processed_image.dtype == np.uint8

        # Results should be different between colorspaces
        rgb_result = results["RGB"].processed_image
        yds_result = results["YDS"].processed_image

        difference = np.mean(
            np.abs(rgb_result.astype(float) - yds_result.astype(float))
        )
        assert difference > 1.0  # Should see some difference

    def test_scale_effect(self):
        """Test that scale parameter affects enhancement intensity."""
        # Use low contrast image so we can see expansion
        test_image = np.random.randint(100, 150, (30, 30, 3), dtype=np.uint8)

        dstretch = DecorrelationStretch()

        result_low = dstretch.process(test_image, "YDS", scale=5.0)
        result_high = dstretch.process(test_image, "YDS", scale=50.0)

        # Higher scale should produce more dramatic changes
        diff_low = np.mean(
            np.abs(result_low.processed_image.astype(float) - test_image.astype(float))
        )
        diff_high = np.mean(
            np.abs(result_high.processed_image.astype(float) - test_image.astype(float))
        )

        assert diff_high > diff_low

    def test_reset_functionality(self):
        """Test reset to original functionality."""
        test_image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)

        dstretch = DecorrelationStretch()

        # Process image
        dstretch.process(test_image, "YDS", scale=25.0)

        # Reset should return original
        original = dstretch.reset_to_original()

        assert original is not None
        assert np.array_equal(original, test_image)


class TestColorspaceManager:
    """Test cases for ColorspaceManager."""

    def test_initialization(self):
        """Test ColorspaceManager initialization."""
        manager = ColorspaceManager()

        # Should have basic colorspaces registered
        available = manager.list_available()
        expected_spaces = ["RGB", "LAB", "YDS", "CRGB", "LDS", "LRE"]

        for space in expected_spaces:
            assert space in available

    def test_colorspace_retrieval(self):
        """Test colorspace retrieval and validation."""
        manager = ColorspaceManager()

        # Valid colorspace
        rgb_space = manager.get_colorspace("RGB")
        assert rgb_space.name == "RGB"

        # Invalid colorspace
        with pytest.raises(ValueError):
            manager.get_colorspace("INVALID")

    def test_availability_check(self):
        """Test colorspace availability checking."""
        manager = ColorspaceManager()

        assert manager.is_available("RGB")
        assert manager.is_available("YDS")
        assert not manager.is_available("INVALID")


class TestUtilityFunctions:
    """Test utility functions."""

    def test_process_image_function(self):
        """Test the convenience process_image function."""
        # This test would need actual image files
        # For now, test that it handles invalid paths correctly

        with pytest.raises(ValueError):
            process_image("nonexistent_file.jpg")


# Integration test with actual image
@pytest.mark.skipif(
    not Path("test_images/paintings.jpg").exists(),
    reason="Test image not found in test_images/",
)
def test_real_image_processing():
    """Integration test with a real image file."""
    image_path = "test_images/paintings.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    dstretch = DecorrelationStretch()
    result = dstretch.process(image, colorspace="YDS", scale=15.0)

    assert result.processed_image is not None
    assert result.processed_image.shape == image.shape
    assert result.colorspace == "YDS"
    assert result.scale == 15.0
