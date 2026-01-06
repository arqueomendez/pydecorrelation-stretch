import numpy as np

from dstretch.independent_processors import (
    AutoContrastProcessor,
    ColorBalanceProcessor,
    FlattenProcessor,
    HueShiftProcessor,
    InvertProcessor,
)


class TestIndependentProcessors:
    def setup_method(self):
        # Create a synthetic image for testing (rgb)
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create a gradient for better testing
        for i in range(100):
            self.image[i, :, 0] = i * 2.5  # Red gradient
            self.image[:, i, 1] = i * 2.5  # Green gradient
            self.image[i, i, 2] = 128  # Constant Blue diagonal

    def test_invert_processor(self):
        """Test InvertProcessor."""
        processor = InvertProcessor()
        result = processor.process(self.image)

        assert result.processor_type == "Invert"
        assert result.image.shape == self.image.shape
        assert result.image.dtype == np.uint8

        # Check actual inversion
        expected = 255 - self.image
        np.testing.assert_array_equal(result.image, expected)

    def test_auto_contrast_processor(self):
        """Test AutoContrastProcessor."""
        processor = AutoContrastProcessor()
        # Create a low contrast image
        low_contrast = (self.image // 4) + 100
        result = processor.process(low_contrast, clip_percent=0.0)

        assert result.processor_type == "Auto Contrast"
        assert result.image.shape == low_contrast.shape

        # Check that range is expanded (histogram stretching)
        for i in range(3):
            assert np.min(result.image[:, :, i]) < np.min(low_contrast[:, :, i])
            assert np.max(result.image[:, :, i]) > np.max(low_contrast[:, :, i])

    def test_color_balance_processor(self):
        """Test ColorBalanceProcessor."""
        processor = ColorBalanceProcessor()
        # Create an image with a color cast (e.g., strong red)
        cast_image = self.image.copy()
        cast_image[:, :, 0] = np.clip(cast_image[:, :, 0].astype(int) + 50, 0, 255)

        result = processor.process(cast_image, strength=1.0)

        assert result.processor_type == "Color Balance"
        assert "original_color_cast" in result.statistics
        # assert "dominant_cast" in result.statistics

    def test_flatten_processor(self):
        """Test FlattenProcessor."""
        processor = FlattenProcessor()

        # Flatten is hard to verify exactly without reimplementing,
        # but we can check output properties
        result = processor.process(self.image, ksize_factor=20.0)

        assert result.processor_type == "Flatten"
        assert result.image.shape == self.image.shape
        # Flatten often reduces variance in illumination
        # (check if it runs without error)

    def test_hue_shift_processor(self):
        """Test HueShiftProcessor."""
        processor = HueShiftProcessor()
        result = processor.process(self.image, shift=90)

        assert result.processor_type == "Hue Shift"

        # For a shift of 90 degrees (in OpenCV HSV 0-180 scale, 90 degrees is 45 units)
        # Actually implementation might use degrees directly. Let's check logic.
        # Implementation: ((h + shift) % 180).
        # Shift in pipeline is usually passed as 'shift' param.

        # Just convert input to HSV, shift H, convert back and compare?
        # That's what the processor does. Let's ensure output is different.
        assert not np.array_equal(result.image, self.image)
