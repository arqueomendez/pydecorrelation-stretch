#!/usr/bin/env python3
"""
Flatten Processor for DStretch Python.

Implements background subtraction for uneven illumination correction.
Uses bandpass filtering approach functionally equivalent to ImageJ's Sliding Paraboloid algorithm.
Optimized for archaeological image enhancement.

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


class FlattenMethod(Enum):
    """Available flatten methods."""

    BANDPASS_FILTER = "bandpass_filter"
    GAUSSIAN_BACKGROUND = "gaussian_background"
    SLIDING_PARABOLOID = "sliding_paraboloid"
    ROLLING_BALL = "rolling_ball"


@dataclass
class FlattenParams:
    """Parameters for flatten processing."""

    method: FlattenMethod = FlattenMethod.BANDPASS_FILTER
    filter_large: float = 40.0  # Large structures to remove (pixels)
    filter_small: float = 3.0  # Small structures to preserve (pixels)
    suppress_stripes: bool = True  # Suppress horizontal/vertical stripes
    tolerance: float = 5.0  # Direction tolerance for stripe suppression (%)
    autoscale_result: bool = True  # Automatically scale result to full range
    preview_background: bool = False  # Show background instead of corrected image
    ball_radius: float = 50.0  # Radius for rolling ball method (pixels)
    paraboloid_radius: float = 50.0  # Radius for sliding paraboloid (pixels)


class FlattenProcessor:
    """
    Advanced flatten processor for archaeological images.

    Corrects uneven illumination while preserving archaeological details.
    Implements multiple algorithms optimized for different types of illumination problems.
    """

    def __init__(self):
        self.last_params = FlattenParams()
        self.last_background = None
        self.last_stats = None

    def process(self, image: np.ndarray, params: FlattenParams | None = None) -> np.ndarray:
        """
        Apply flatten correction to an image.

        Args:
            image: Input image (H, W, 3) in RGB format, uint8
            params: Flatten parameters

        Returns:
            Flattened image (H, W, 3) in RGB format, uint8
        """
        if params is None:
            params = FlattenParams()

        # Store parameters for analysis
        self.last_params = params

        # Convert to float for processing
        image_float = image.astype(np.float64) / 255.0

        # Apply selected flatten method
        if params.method == FlattenMethod.BANDPASS_FILTER:
            flattened = self._bandpass_filter_flatten(image_float, params)
        elif params.method == FlattenMethod.GAUSSIAN_BACKGROUND:
            flattened = self._gaussian_background_flatten(image_float, params)
        elif params.method == FlattenMethod.SLIDING_PARABOLOID:
            flattened = self._sliding_paraboloid_flatten(image_float, params)
        elif params.method == FlattenMethod.ROLLING_BALL:
            flattened = self._rolling_ball_flatten(image_float, params)
        else:
            flattened = image_float.copy()

        # Auto-scale result if requested
        if params.autoscale_result:
            flattened = self._autoscale_image(flattened)

        # Convert back to uint8
        result = np.clip(flattened * 255.0, 0, 255).astype(np.uint8)

        return result

    def _bandpass_filter_flatten(
        self, image: np.ndarray, params: FlattenParams
    ) -> np.ndarray:
        """
        Apply bandpass filter approach (ImageJ default since v1.39f).

        This is functionally equivalent to subtracting a large Gaussian blur
        from the original image, preserving details while removing gradual illumination changes.
        """
        # Process each channel separately
        flattened = np.zeros_like(image)
        background = np.zeros_like(image)

        for channel in range(3):
            channel_data = image[:, :, channel]

            # Large structure removal (background estimation)
            if params.filter_large > 0:
                large_gaussian = gaussian_filter(
                    channel_data, sigma=params.filter_large
                )
            else:
                large_gaussian = np.zeros_like(channel_data)

            # Small structure preservation
            if params.filter_small > 0:
                small_gaussian = gaussian_filter(
                    channel_data, sigma=params.filter_small
                )
            else:
                small_gaussian = channel_data.copy()

            # Bandpass result: original - large_blur + small_blur_mean
            small_mean = np.mean(small_gaussian)
            bandpass_result = channel_data - large_gaussian + small_mean

            # Store background for analysis
            background[:, :, channel] = large_gaussian

            # Apply stripe suppression if requested
            if params.suppress_stripes:
                bandpass_result = self._suppress_stripes(
                    bandpass_result, params.tolerance
                )

            flattened[:, :, channel] = bandpass_result

        # Store background and statistics
        self.last_background = background
        self._calculate_flatten_statistics(image, flattened, background)

        # Return background preview if requested
        if params.preview_background:
            return background

        return flattened

    def _gaussian_background_flatten(
        self, image: np.ndarray, params: FlattenParams
    ) -> np.ndarray:
        """
        Apply simple Gaussian background subtraction.

        Estimates background using large Gaussian blur and subtracts it.
        """
        flattened = np.zeros_like(image)
        background = np.zeros_like(image)

        for channel in range(3):
            channel_data = image[:, :, channel]

            # Estimate background using large Gaussian
            bg_estimate = gaussian_filter(channel_data, sigma=params.filter_large)
            background[:, :, channel] = bg_estimate

            # Subtract background, preserve mean
            original_mean = np.mean(channel_data)
            corrected = channel_data - bg_estimate + original_mean

            # Apply stripe suppression if requested
            if params.suppress_stripes:
                corrected = self._suppress_stripes(corrected, params.tolerance)

            flattened[:, :, channel] = corrected

        # Store results
        self.last_background = background
        self._calculate_flatten_statistics(image, flattened, background)

        if params.preview_background:
            return background

        return flattened

    def _sliding_paraboloid_flatten(
        self, image: np.ndarray, params: FlattenParams
    ) -> np.ndarray:
        """
        Apply sliding paraboloid algorithm (ImageJ BackgroundSubtracter.java).

        This is a morphological approach that fits a paraboloid to local minima.
        More accurate than Gaussian but computationally intensive.
        """
        flattened = np.zeros_like(image)
        background = np.zeros_like(image)

        for channel in range(3):
            channel_data = image[:, :, channel]

            # Apply sliding paraboloid
            bg_estimate = self._apply_sliding_paraboloid(
                channel_data, params.paraboloid_radius
            )
            background[:, :, channel] = bg_estimate

            # Subtract background
            original_mean = np.mean(channel_data)
            corrected = channel_data - bg_estimate + original_mean

            # Apply stripe suppression if requested
            if params.suppress_stripes:
                corrected = self._suppress_stripes(corrected, params.tolerance)

            flattened[:, :, channel] = corrected

        # Store results
        self.last_background = background
        self._calculate_flatten_statistics(image, flattened, background)

        if params.preview_background:
            return background

        return flattened

    def _rolling_ball_flatten(
        self, image: np.ndarray, params: FlattenParams
    ) -> np.ndarray:
        """
        Apply rolling ball background subtraction.

        Classical morphological approach using ball-shaped structuring element.
        """
        flattened = np.zeros_like(image)
        background = np.zeros_like(image)

        # Create ball-shaped structuring element
        radius = int(params.ball_radius)
        if radius < 1:
            radius = 1

        # Create ball structuring element
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        ball = x * x + y * y <= radius * radius

        for channel in range(3):
            channel_data = image[:, :, channel]

            # Apply morphological opening (erosion followed by dilation)
            bg_estimate = ndimage.grey_opening(channel_data, structure=ball)
            background[:, :, channel] = bg_estimate

            # Subtract background
            original_mean = np.mean(channel_data)
            corrected = channel_data - bg_estimate + original_mean

            # Apply stripe suppression if requested
            if params.suppress_stripes:
                corrected = self._suppress_stripes(corrected, params.tolerance)

            flattened[:, :, channel] = corrected

        # Store results
        self.last_background = background
        self._calculate_flatten_statistics(image, flattened, background)

        if params.preview_background:
            return background

        return flattened

    def _apply_sliding_paraboloid(self, image: np.ndarray, radius: float) -> np.ndarray:
        """
        Apply sliding paraboloid algorithm.

        This is a simplified version of the ImageJ algorithm.
        For full implementation, would need to replicate the exact ImageJ BackgroundSubtracter logic.
        """
        # For simplicity, approximate with morphological opening using disk
        # A full implementation would use the exact paraboloid fitting algorithm
        radius_int = max(1, int(radius))

        # Create disk structuring element
        y, x = np.ogrid[-radius_int : radius_int + 1, -radius_int : radius_int + 1]
        disk = x * x + y * y <= radius_int * radius_int

        # Apply morphological opening
        result = ndimage.grey_opening(image, structure=disk)

        return result

    def _suppress_stripes(self, image: np.ndarray, tolerance: float) -> np.ndarray:
        """
        Suppress horizontal and vertical stripes in the image.

        This addresses common artifacts in scanned or photographed images.
        """
        result = image.copy()

        # Calculate FFT to analyze frequencies
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)

        # Get image dimensions
        height, width = image.shape
        center_y, center_x = height // 2, width // 2

        # Create mask for stripe suppression
        mask = np.ones_like(fft_shift)

        # Tolerance as fraction
        tol_frac = tolerance / 100.0

        # Suppress horizontal stripes (vertical lines in frequency domain)
        stripe_width = max(1, int(width * tol_frac))
        mask[:, center_x - stripe_width : center_x + stripe_width] *= 0.1

        # Suppress vertical stripes (horizontal lines in frequency domain)
        stripe_height = max(1, int(height * tol_frac))
        mask[center_y - stripe_height : center_y + stripe_height, :] *= 0.1

        # Apply mask and convert back
        fft_filtered = fft_shift * mask
        fft_ishifted = np.fft.ifftshift(fft_filtered)
        result = np.real(np.fft.ifft2(fft_ishifted))

        return result

    def _autoscale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Auto-scale image to full dynamic range.

        Stretches the image so that the darkest pixel becomes 0 and brightest becomes 1.
        """
        result = image.copy()

        for channel in range(image.shape[2]):
            channel_data = image[:, :, channel]

            # Find min and max values
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)

            # Avoid division by zero
            if max_val > min_val:
                # Scale to [0, 1] range
                result[:, :, channel] = (channel_data - min_val) / (max_val - min_val)
            else:
                # If uniform, set to middle gray
                result[:, :, channel] = 0.5

        return result

    def _calculate_flatten_statistics(
        self, original: np.ndarray, flattened: np.ndarray, background: np.ndarray
    ):
        """Calculate statistics about the flatten operation."""

        # Calculate means for each channel
        orig_means = [np.mean(original[:, :, i]) for i in range(3)]
        flat_means = [np.mean(flattened[:, :, i]) for i in range(3)]
        bg_means = [np.mean(background[:, :, i]) for i in range(3)]

        # Calculate standard deviations (measure of contrast)
        orig_stds = [np.std(original[:, :, i]) for i in range(3)]
        flat_stds = [np.std(flattened[:, :, i]) for i in range(3)]

        # Calculate uniformity improvement
        orig_uniformity = [
            self._calculate_uniformity(original[:, :, i]) for i in range(3)
        ]
        flat_uniformity = [
            self._calculate_uniformity(flattened[:, :, i]) for i in range(3)
        ]

        self.last_stats = {
            "original_means": orig_means,
            "flattened_means": flat_means,
            "background_means": bg_means,
            "original_stds": orig_stds,
            "flattened_stds": flat_stds,
            "original_uniformity": orig_uniformity,
            "flattened_uniformity": flat_uniformity,
            "uniformity_improvement": [
                f - o for f, o in zip(flat_uniformity, orig_uniformity, strict=False)
            ],
        }

    def _calculate_uniformity(self, image: np.ndarray) -> float:
        """
        Calculate image uniformity measure.

        Higher values indicate more uniform illumination.
        Based on coefficient of variation (inverse).
        """
        mean_val = np.mean(image)
        std_val = np.std(image)

        if std_val == 0:
            return 1.0  # Perfect uniformity

        # Coefficient of variation (lower is more uniform)
        cv = std_val / (mean_val + 1e-8)

        # Return inverse (higher is more uniform)
        return float(1.0 / (1.0 + cv))

    def get_flatten_statistics(self) -> dict:
        """Get statistics from the last flatten operation."""
        return self.last_stats if self.last_stats else {}

    def get_background_estimate(self) -> np.ndarray | None:
        """Get the background estimate from the last flatten operation."""
        return self.last_background

    def analyze_illumination(self, image: np.ndarray) -> dict:
        """
        Analyze illumination uniformity without applying correction.

        Returns:
            Dictionary with illumination analysis information.
        """
        image_float = image.astype(np.float64) / 255.0

        # Calculate statistics for each channel
        analysis = {
            "channels": [],
            "overall_uniformity": 0.0,
            "needs_correction": False,
            "recommended_method": FlattenMethod.BANDPASS_FILTER,
            "recommended_large_filter": 40.0,
        }

        total_uniformity = 0.0

        for channel in range(3):
            channel_data = image_float[:, :, channel]

            # Basic statistics
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)

            # Uniformity measure
            uniformity = self._calculate_uniformity(channel_data)
            total_uniformity += uniformity

            # Gradient analysis (illumination variation)
            grad_y, grad_x = np.gradient(channel_data)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(grad_magnitude)

            channel_info = {
                "channel": channel,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "range": max_val - min_val,
                "uniformity": uniformity,
                "avg_gradient": avg_gradient,
            }

            analysis["channels"].append(channel_info)

        # Overall analysis
        analysis["overall_uniformity"] = total_uniformity / 3.0
        analysis["needs_correction"] = analysis["overall_uniformity"] < 0.7

        # Recommend method based on analysis
        avg_gradient = np.mean([ch["avg_gradient"] for ch in analysis["channels"]])

        if avg_gradient > 0.1:
            analysis["recommended_method"] = FlattenMethod.SLIDING_PARABOLOID
            analysis["recommended_large_filter"] = 60.0
        elif avg_gradient > 0.05:
            analysis["recommended_method"] = FlattenMethod.BANDPASS_FILTER
            analysis["recommended_large_filter"] = 40.0
        else:
            analysis["recommended_method"] = FlattenMethod.GAUSSIAN_BACKGROUND
            analysis["recommended_large_filter"] = 30.0

        return analysis


def create_test_image_with_uneven_illumination() -> np.ndarray:
    """Create a test image with uneven illumination for validation."""
    height, width = 400, 600

    # Create base pattern
    y, x = np.meshgrid(
        np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij"
    )

    # Create test pattern with archaeological-like features
    base_pattern = np.zeros((height, width))

    # Add some "rock art" features
    for _ in range(10):
        center_y = np.random.randint(height // 4, 3 * height // 4)
        center_x = np.random.randint(width // 4, 3 * width // 4)
        radius = np.random.randint(10, 30)

        y_circ, x_circ = np.ogrid[:height, :width]
        mask = (x_circ - center_x) ** 2 + (y_circ - center_y) ** 2 <= radius**2
        base_pattern[mask] = 0.8

    # Add uneven illumination (gradient from top-left to bottom-right)
    illumination = 0.3 + 0.7 * (0.5 * x + 0.5 * y)

    # Add some radial illumination variation
    center_illum_y, center_illum_x = height // 3, 2 * width // 3
    radial_dist = np.sqrt(
        (y * height - center_illum_y) ** 2 + (x * width - center_illum_x) ** 2
    )
    radial_illum = 0.8 + 0.4 * np.exp(-radial_dist / 200)

    # Combine illumination effects
    total_illumination = illumination * radial_illum

    # Apply to base pattern
    final_pattern = (base_pattern + 0.3) * total_illumination

    # Convert to RGB
    rgb_image = np.stack([final_pattern, final_pattern, final_pattern], axis=-1)

    # Add some color variation
    rgb_image[:, :, 0] *= 1.1  # Slightly more red
    rgb_image[:, :, 2] *= 0.9  # Slightly less blue

    # Convert to uint8
    rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

    return rgb_image


# Utility functions for archaeological workflow
def recommend_flatten_method(image: np.ndarray) -> FlattenMethod:
    """
    Recommend optimal flatten method based on image characteristics.

    For archaeological images:
    - Bandpass Filter: General purpose, good for most cases
    - Gaussian Background: Simple gradients
    - Sliding Paraboloid: Complex illumination patterns
    - Rolling Ball: Textured backgrounds
    """
    processor = FlattenProcessor()
    analysis = processor.analyze_illumination(image)

    return analysis["recommended_method"]


def estimate_optimal_filter_size(image: np.ndarray) -> float:
    """
    Estimate optimal filter size based on image dimensions and content.

    Returns filter size as fraction of image width.
    """
    height, width = image.shape[:2]

    # Base filter size as fraction of smaller dimension
    base_size = min(height, width) * 0.1

    # Analyze spatial frequency content
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

    # Calculate local variance to estimate detail size
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)

    mean_img = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sqr_img = cv2.filter2D((gray.astype(np.float32)) ** 2, -1, kernel)
    variance_img = sqr_img - mean_img**2

    avg_variance = np.mean(variance_img)

    # Adjust filter size based on variance
    if avg_variance > 1000:  # High detail
        return base_size * 0.5
    elif avg_variance < 100:  # Low detail
        return base_size * 2.0
    else:
        return base_size
