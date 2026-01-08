#!/usr/bin/env python3
"""
DStretch Python - Auto Contrast Processor
Implements DStretch's lEnhance method with luminance-based histogram stretch

Based on DStretch ImageJ implementation:
- Calculates luminance as (min(R,G,B) + max(R,G,B)) / 2
- Finds min/max luminance values ignoring small percentage of outliers
- Applies linear transformation to each RGB channel using same luminance cutoffs
- Preserves color relationships while stretching brightness uniformly

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1
"""

import logging

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class AutoContrastProcessor:
    """
    Implements DStretch's lEnhance auto contrast functionality.

    Uses luminance-based histogram stretch that preserves color relationships
    while optimizing the dynamic range of the image.
    """

    def __init__(self, clip_percentage: float = 0.1, preserve_colors: bool = True):
        """
        Initialize Auto Contrast Processor.

        Args:
            clip_percentage: Percentage of pixels to ignore at extremes (0.0-5.0)
            preserve_colors: Whether to preserve color relationships during stretch
        """
        self.clip_percentage = clip_percentage
        self.preserve_colors = preserve_colors

        # Validate parameters
        if not 0.0 <= clip_percentage <= 5.0:
            raise ValueError("clip_percentage must be between 0.0 and 5.0")

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply auto contrast enhancement using DStretch's lEnhance algorithm.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W)

        Returns:
            numpy.ndarray: Auto contrast enhanced image with same dtype as input

        Raises:
            ValueError: If image format is invalid
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if len(image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D grayscale or 3D color")

        # Log processing info
        logger.info(
            f"Processing auto contrast - Shape: {image.shape}, "
            f"Clip: {self.clip_percentage}%, Preserve colors: {self.preserve_colors}"
        )

        # Store original dtype for output
        original_dtype = image.dtype

        # Convert to working format (float64 for precision)
        working_image = image.astype(np.float64)

        if len(image.shape) == 2:
            # Grayscale image - direct histogram stretch
            result = self._stretch_grayscale(working_image)
        else:
            # Color image - use luminance-based method
            if self.preserve_colors:
                result = self._stretch_luminance_based(working_image)
            else:
                result = self._stretch_independent_channels(working_image)

        # Convert back to original dtype
        if original_dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif original_dtype == np.uint16:
            result = np.clip(result, 0, 65535).astype(np.uint16)
        else:
            result = result.astype(original_dtype)

        logger.info("Auto contrast completed successfully")
        return result

    def _stretch_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram stretch to grayscale image.
        """
        # Calculate histogram
        flat_image = image.flatten()
        hist, bin_edges = np.histogram(flat_image, bins=256, range=(0, 255))

        # Find cutoff points
        min_val, max_val = self._find_histogram_cutoffs(hist, len(flat_image))

        # Apply linear stretch
        return self._apply_linear_stretch(image, min_val, max_val)

    def _stretch_luminance_based(self, image: np.ndarray) -> np.ndarray:
        """
        Apply DStretch's lEnhance: luminance-based stretch preserving color.

        This is the core algorithm from DStretch ImageJ plugin.
        """
        # Calculate luminance using DStretch formula: (min(R,G,B) + max(R,G,B)) / 2
        luminance = self._calculate_luminance_dstretch(image)

        # Calculate luminance histogram
        hist, bin_edges = np.histogram(luminance.flatten(), bins=256, range=(0, 255))

        # Find luminance cutoff points
        min_lum, max_lum = self._find_histogram_cutoffs(hist, luminance.size)

        # Apply stretch to each RGB channel using same luminance cutoffs
        result = np.zeros_like(image)
        for channel in range(3):
            result[:, :, channel] = self._apply_linear_stretch(
                image[:, :, channel], min_lum, max_lum
            )

        return result

    def _stretch_independent_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Apply independent histogram stretch to each channel.
        """
        result = np.zeros_like(image)

        for channel in range(3):
            channel_data = image[:, :, channel]
            hist, bin_edges = np.histogram(
                channel_data.flatten(), bins=256, range=(0, 255)
            )
            min_val, max_val = self._find_histogram_cutoffs(hist, channel_data.size)
            result[:, :, channel] = self._apply_linear_stretch(
                channel_data, min_val, max_val
            )

        return result

    def _calculate_luminance_dstretch(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate luminance using DStretch formula: (min(R,G,B) + max(R,G,B)) / 2

        This differs from standard luminance calculations and is specific to DStretch.
        """
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Find min and max for each pixel across RGB channels
        min_rgb = np.minimum(np.minimum(R, G), B)
        max_rgb = np.maximum(np.maximum(R, G), B)

        # DStretch luminance formula
        luminance = (min_rgb + max_rgb) / 2.0

        return luminance

    def _find_histogram_cutoffs(
        self, histogram: np.ndarray, total_pixels: int
    ) -> tuple[float, float]:
        """
        Find histogram cutoff points ignoring specified percentage of outliers.

        Uses DStretch's approach of finding first and last bins with significant content.
        """
        # Calculate threshold based on clip percentage
        # DStretch uses a dynamic threshold, not fixed percentile
        threshold = int(total_pixels * (self.clip_percentage / 100.0))

        # Find minimum value (first bin with count > threshold)
        min_val = 0
        for i in range(len(histogram)):
            if histogram[i] > threshold:
                min_val = i
                break

        # Find maximum value (last bin with count > threshold)
        max_val = 255
        for i in range(len(histogram) - 1, -1, -1):
            if histogram[i] > threshold:
                max_val = i
                break

        # Ensure valid range
        if max_val <= min_val:
            min_val = 0
            max_val = 255

        return float(min_val), float(max_val)

    def _apply_linear_stretch(
        self, channel: np.ndarray, min_val: float, max_val: float
    ) -> np.ndarray:
        """
        Apply linear stretch to a single channel.

        Maps [min_val, max_val] to [0, 255] linearly.
        """
        if max_val > min_val:
            # Linear mapping: output = (input - min) * (255 / (max - min))
            stretched = (channel - min_val) * (255.0 / (max_val - min_val))
            return np.clip(stretched, 0, 255)
        else:
            # No stretch needed if range is invalid
            return channel

    def get_contrast_statistics(self, image: np.ndarray) -> dict:
        """
        Get statistics about image contrast for analysis.

        Args:
            image: Input image

        Returns:
            Dictionary with contrast statistics
        """
        if len(image.shape) == 2:
            # Grayscale
            std_dev = np.std(image)
            min_val, max_val = np.min(image), np.max(image)
            dynamic_range = max_val - min_val
        else:
            # Color - use luminance
            luminance = self._calculate_luminance_dstretch(image.astype(np.float64))
            std_dev = np.std(luminance)
            min_val, max_val = np.min(luminance), np.max(luminance)
            dynamic_range = max_val - min_val

        return {
            "standard_deviation": float(std_dev),
            "dynamic_range": float(dynamic_range),
            "min_value": float(min_val),
            "max_value": float(max_val),
            "needs_enhancement": std_dev < 30.0 or dynamic_range < 100.0,
        }


def main():
    """
    Example usage and testing of AutoContrastProcessor.
    """
    # Create test image with low contrast
    test_image = np.random.randint(
        80, 120, (100, 100, 3), dtype=np.uint8
    )  # Low contrast

    # Test different contrast modes
    processor_preserve = AutoContrastProcessor(
        clip_percentage=0.1, preserve_colors=True
    )
    processor_independent = AutoContrastProcessor(
        clip_percentage=0.1, preserve_colors=False
    )

    # Get statistics before enhancement
    stats_before = processor_preserve.get_contrast_statistics(test_image)
    print("Statistics before enhancement:")
    for key, value in stats_before.items():
        print(f"  {key}: {value}")

    # Apply auto contrast
    enhanced_preserve = processor_preserve.process(test_image)
    enhanced_independent = processor_independent.process(test_image)

    # Get statistics after enhancement
    stats_after = processor_preserve.get_contrast_statistics(enhanced_preserve)
    print("\nStatistics after enhancement (preserve colors):")
    for key, value in stats_after.items():
        print(f"  {key}: {value}")

    print(f"\nOriginal range: {np.min(test_image)} - {np.max(test_image)}")
    print(
        f"Enhanced range (preserve): {np.min(enhanced_preserve)} - {np.max(enhanced_preserve)}"
    )
    print(
        f"Enhanced range (independent): {np.min(enhanced_independent)} - {np.max(enhanced_independent)}"
    )

    # Verify enhancement
    improvement = stats_after["dynamic_range"] - stats_before["dynamic_range"]
    print(f"\nDynamic range improvement: {improvement:.1f}")


if __name__ == "__main__":
    main()
