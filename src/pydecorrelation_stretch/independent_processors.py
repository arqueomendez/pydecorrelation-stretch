#!/usr/bin/env python3
"""
DStretch Advanced Independent Processors
=========================================

Independent processing tools that operate on RGB images before decorrelation stretch.
These processors implement the advanced features from ImageJ DStretch plugin.

Architecture:
- Pre-processing: Applied to original RGB image
- Core Algorithm: Decorrelation Stretch (separate module)
- Post-processing: Optional effects on enhanced image

Author: Claude Sonnet 4 (Based on ImageJ DStretch v6.3 by Jon Harman)
Version: 2.0
Date: January 2025

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1
"""

import logging
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, cast

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=".*divide by zero encountered.*")


class ProcessorType(Enum):
    """Types of processors available."""

    AUTO_CONTRAST = "auto_contrast"
    COLOR_BALANCE = "color_balance"
    FLATTEN = "flatten"
    INVERT = "invert"
    HUE_SHIFT = "hue_shift"


class ProcessingResult:
    """Result object for processing operations."""

    def __init__(
        self,
        image: np.ndarray,
        processor_type: str,
        parameters: dict[str, Any],
        statistics: dict | None = None,
    ):
        self.image = image
        self.processor_type = processor_type
        self.parameters = parameters
        self.statistics = statistics or {}

    def __repr__(self):
        return f"ProcessingResult({self.processor_type}, shape={self.image.shape})"


class BaseProcessor(ABC):
    """Abstract base class for all image processors."""

    def __init__(self, name: str):
        self.name = name
        self._last_result = None

    def get_last_result(self) -> object | None:
        """Get the result of the last processing operation."""
        return self._last_result

    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """Process the input image and return a ProcessingResult."""
        pass

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Validate input image format."""
        if image is None:
            raise ValueError("Image cannot be None")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB with 3 channels")

        if image.dtype != np.uint8:
            raise ValueError("Image must be uint8 format")

    @staticmethod
    def _ensure_uint8_range(image: np.ndarray) -> np.ndarray:
        """Ensure image values are in valid uint8 range."""
        return np.clip(image, 0, 255).astype(np.uint8)


class InvertProcessor(BaseProcessor):
    """
    Simple color inversion processor.
    Equivalent to ImageJ Edit > Invert command.
    """

    def __init__(self):
        super().__init__("Invert")

    def process(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """
        Invert image colors: output = 255 - input

        Args:
            image: Input RGB image (uint8)

        Returns:
            ProcessingResult with inverted image
        """
        self._validate_image(image)

        # Simple inversion
        inverted = 255 - image

        # Statistics
        stats = {
            "mean_change": np.mean(
                np.abs(inverted.astype(float) - image.astype(float))
            ),
            "processing_time": "instant",
        }

        result = ProcessingResult(
            image=inverted, processor_type=self.name, parameters={}, statistics=stats
        )
        self._last_result = result
        return result


class AutoContrastProcessor(BaseProcessor):
    """
    Automatic contrast enhancement using histogram stretching.
    Based on ImageJ ContrastAdjuster.java with percentile clipping.
    """

    def __init__(self):
        super().__init__("Auto Contrast")

    def process(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """
        Apply automatic contrast enhancement.

        Args:
            image: Input RGB image
            saturated_pixels: Percentage of pixels to saturate (default: 0.35)
            normalize: Whether to normalize after stretching (default: True)
            equalize: Whether to apply histogram equalization (default: False)

        Returns:
            ProcessingResult with contrast enhanced image
        """
        self._validate_image(image)

        # Parameters
        saturated_pixels = kwargs.get("saturated_pixels", 0.35)
        normalize = kwargs.get("normalize", True)
        equalize = kwargs.get("equalize", False)

        # Process each channel independently
        enhanced_image = np.zeros_like(image)
        stats = {"channels": {}, "overall": {}}

        for channel in range(3):
            channel_data = image[:, :, channel]

            if equalize:
                # Histogram equalization
                enhanced_channel = cv2.equalizeHist(channel_data)
            else:
                # Percentile-based stretching
                enhanced_channel = self._stretch_histogram(
                    channel_data, saturated_pixels, normalize
                )

            enhanced_image[:, :, channel] = enhanced_channel

            # Channel statistics
            stats["channels"][f"channel_{channel}"] = {
                "original_range": (np.min(channel_data), np.max(channel_data)),
                "enhanced_range": (np.min(enhanced_channel), np.max(enhanced_channel)),
                "contrast_improvement": np.std(cast(np.ndarray, enhanced_channel))
                / max(np.std(cast(np.ndarray, channel_data)), 1e-6),
            }

        # Overall statistics
        original_contrast = np.std(cast(np.ndarray, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)))
        enhanced_contrast = np.std(cast(np.ndarray, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)))

        stats["overall"] = {
            "contrast_improvement": enhanced_contrast / max(original_contrast, 1e-6),
            "saturated_pixels_percent": saturated_pixels,
            "equalization_applied": equalize,
        }

        parameters = {
            "saturated_pixels": saturated_pixels,
            "normalize": normalize,
            "equalize": equalize,
        }

        result = ProcessingResult(
            image=enhanced_image,
            processor_type=self.name,
            parameters=parameters,
            statistics=stats,
        )
        self._last_result = result
        return result

    def _stretch_histogram(
        self, channel: np.ndarray, saturated_pixels: float, normalize: bool
    ) -> np.ndarray:
        """Apply histogram stretching to a single channel."""

        # Calculate percentiles
        lower_percentile = saturated_pixels / 2
        upper_percentile = 100 - saturated_pixels / 2

        lower_bound = np.percentile(channel, lower_percentile)
        upper_bound = np.percentile(channel, upper_percentile)

        # Prevent division by zero
        if upper_bound - lower_bound < 1e-6:
            return channel

        # Stretch histogram
        stretched = (channel.astype(float) - lower_bound) / (upper_bound - lower_bound)

        if normalize:
            stretched = stretched * 255

        return self._ensure_uint8_range(stretched)


class ColorBalanceProcessor(BaseProcessor):
    """
    Color balance processor implementing Gray World algorithm.
    Based on ImageJ ContrastAdjuster.java with archaeological optimizations.
    """

    def __init__(self):
        super().__init__("Color Balance")

    def process(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """
        Apply color balance correction.

        Args:
            image: Input RGB image
            method: 'gray_world', 'white_patch', or 'manual' (default: 'gray_world')
            strength: Balance strength 0.0-1.0 (default: 0.8)
            temperature: Color temperature adjustment (default: 0.0)
            tint: Tint adjustment (default: 0.0)
            preserve_luminance: Whether to preserve original luminance (default: True)
            percentile_clip: Percentile for clipping outliers (default: 1.0)

        Returns:
            ProcessingResult with color balanced image
        """
        self._validate_image(image)

        # Parameters
        method = kwargs.get("method", "gray_world")
        strength = kwargs.get("strength", 0.8)
        temperature = kwargs.get("temperature", 0.0)
        tint = kwargs.get("tint", 0.0)
        preserve_luminance = kwargs.get("preserve_luminance", True)
        percentile_clip = kwargs.get("percentile_clip", 1.0)

        # Store original for statistics
        original_image = image.copy()

        # Convert to float for processing
        float_image = image.astype(np.float64)

        # Apply color balance method
        if method == "gray_world":
            balanced_image = self._gray_world_balance(
                float_image, strength, percentile_clip
            )
        elif method == "white_patch":
            balanced_image = self._white_patch_balance(
                float_image, strength, percentile_clip
            )
        elif method == "manual":
            balanced_image = self._manual_balance(
                float_image, temperature, tint, strength
            )
        else:
            raise ValueError(f"Unknown color balance method: {method}")

        # Preserve luminance if requested
        if preserve_luminance:
            balanced_image = self._preserve_luminance(original_image, balanced_image)

        # Convert back to uint8
        result_image = self._ensure_uint8_range(balanced_image)

        # Calculate statistics
        stats = self._calculate_balance_statistics(original_image, result_image)

        parameters = {
            "method": method,
            "strength": strength,
            "temperature": temperature,
            "tint": tint,
            "preserve_luminance": preserve_luminance,
            "percentile_clip": percentile_clip,
        }

        result = ProcessingResult(
            image=result_image,
            processor_type=self.name,
            parameters=parameters,
            statistics=stats,
        )
        self._last_result = result
        return result

    def _gray_world_balance(
        self, image: np.ndarray, strength: float, percentile_clip: float
    ) -> np.ndarray:
        """Apply Gray World algorithm with percentile clipping."""

        # Calculate means with percentile clipping
        channel_means = []
        for channel in range(3):
            channel_data = image[:, :, channel]

            # Remove outliers using percentiles
            lower_bound = np.percentile(channel_data, percentile_clip)
            upper_bound = np.percentile(channel_data, 100 - percentile_clip)

            # Calculate mean of clipped data
            clipped_data = channel_data[
                (channel_data >= lower_bound) & (channel_data <= upper_bound)
            ]
            channel_means.append(np.mean(clipped_data))

        # Calculate global mean
        global_mean = np.mean(channel_means)

        # Calculate correction factors
        correction_factors = []
        for mean in channel_means:
            if mean > 1e-6:  # Avoid division by zero
                factor = global_mean / mean
                # Apply strength
                factor = 1.0 + strength * (factor - 1.0)
                correction_factors.append(factor)
            else:
                correction_factors.append(1.0)

        # Apply corrections
        corrected_image = image.copy()
        for channel in range(3):
            corrected_image[:, :, channel] *= correction_factors[channel]

        return corrected_image

    def _white_patch_balance(
        self, image: np.ndarray, strength: float, percentile_clip: float
    ) -> np.ndarray:
        """Apply White Patch algorithm."""

        # Find maximum values with percentile clipping
        max_values = []
        for channel in range(3):
            channel_data = image[:, :, channel]
            max_val = np.percentile(channel_data, 100 - percentile_clip)
            max_values.append(max_val)

        # Calculate global maximum
        global_max = max(max_values)

        # Calculate correction factors
        correction_factors = []
        for max_val in max_values:
            if max_val > 1e-6:
                factor = global_max / max_val
                # Apply strength
                factor = 1.0 + strength * (factor - 1.0)
                correction_factors.append(factor)
            else:
                correction_factors.append(1.0)

        # Apply corrections
        corrected_image = image.copy()
        for channel in range(3):
            corrected_image[:, :, channel] *= correction_factors[channel]

        return corrected_image

    def _manual_balance(
        self, image: np.ndarray, temperature: float, tint: float, strength: float
    ) -> np.ndarray:
        """Apply manual color temperature and tint adjustments."""

        corrected_image = image.copy()

        # Temperature adjustment (blue-yellow axis)
        if temperature != 0:
            temp_factor = 1.0 + temperature * strength
            if temperature > 0:  # Warmer (more yellow, less blue)
                corrected_image[:, :, 0] *= temp_factor  # Red
                corrected_image[:, :, 1] *= temp_factor  # Green
                corrected_image[:, :, 2] /= temp_factor  # Blue
            else:  # Cooler (more blue, less yellow)
                corrected_image[:, :, 0] /= abs(temp_factor)  # Red
                corrected_image[:, :, 1] /= abs(temp_factor)  # Green
                corrected_image[:, :, 2] *= abs(temp_factor)  # Blue

        # Tint adjustment (green-magenta axis)
        if tint != 0:
            tint_factor = 1.0 + tint * strength
            if tint > 0:  # More magenta (less green)
                corrected_image[:, :, 0] *= tint_factor  # Red
                corrected_image[:, :, 2] *= tint_factor  # Blue
                corrected_image[:, :, 1] /= tint_factor  # Green
            else:  # More green (less magenta)
                corrected_image[:, :, 0] /= abs(tint_factor)  # Red
                corrected_image[:, :, 2] /= abs(tint_factor)  # Blue
                corrected_image[:, :, 1] *= abs(tint_factor)  # Green

        return corrected_image

    def _preserve_luminance(
        self, original: np.ndarray, processed: np.ndarray
    ) -> np.ndarray:
        """Preserve original luminance in processed image."""

        # Convert to float
        original_float = original.astype(np.float64)
        processed_float = processed.astype(np.float64)

        # Calculate luminance (using ITU-R BT.709 coefficients)
        original_lum = (
            0.2126 * original_float[:, :, 0]
            + 0.7152 * original_float[:, :, 1]
            + 0.0722 * original_float[:, :, 2]
        )

        processed_lum = (
            0.2126 * processed_float[:, :, 0]
            + 0.7152 * processed_float[:, :, 1]
            + 0.0722 * processed_float[:, :, 2]
        )

        # Calculate luminance ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            lum_ratio = np.divide(
                original_lum,
                processed_lum,
                out=np.ones_like(original_lum),
                where=processed_lum > 1e-6,
            )

        # Apply luminance preservation
        result = processed_float.copy()
        for channel in range(3):
            result[:, :, channel] *= lum_ratio

        return result

    def _calculate_balance_statistics(
        self, original: np.ndarray, processed: np.ndarray
    ) -> dict:
        """Calculate color balance statistics."""

        # Color cast analysis
        original_means = [np.mean(original[:, :, i]) for i in range(3)]
        processed_means = [np.mean(processed[:, :, i]) for i in range(3)]

        # Calculate color cast before and after
        original_cast = {
            "red_bias": original_means[0] - np.mean(original_means),
            "green_bias": original_means[1] - np.mean(original_means),
            "blue_bias": original_means[2] - np.mean(original_means),
        }

        processed_cast = {
            "red_bias": processed_means[0] - np.mean(processed_means),
            "green_bias": processed_means[1] - np.mean(processed_means),
            "blue_bias": processed_means[2] - np.mean(processed_means),
        }

        # Calculate correction strength
        cast_reduction = {
            "red": abs(original_cast["red_bias"]) - abs(processed_cast["red_bias"]),
            "green": abs(original_cast["green_bias"])
            - abs(processed_cast["green_bias"]),
            "blue": abs(original_cast["blue_bias"]) - abs(processed_cast["blue_bias"]),
        }

        return {
            "original_color_cast": original_cast,
            "processed_color_cast": processed_cast,
            "cast_reduction": cast_reduction,
            "overall_improvement": np.mean(list(cast_reduction.values())),
        }


class FlattenProcessor(BaseProcessor):
    """
    Flatten illumination processor for correcting uneven lighting.
    Based on ImageJ BackgroundSubtracter.java with multiple algorithms.
    """

    def __init__(self):
        super().__init__("Flatten")

    def process(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """
        Apply flatten illumination correction.

        Args:
            image: Input RGB image
            method: 'bandpass', 'gaussian', 'sliding_paraboloid', 'rolling_ball' (default: 'bandpass')
            large_structures: Size of large structures to filter (default: 40)
            small_structures: Size of small structures to preserve (default: 3)
            suppress_stripes: Whether to suppress horizontal/vertical stripes (default: True)
            auto_scale: Whether to auto-scale the result (default: True)

        Returns:
            ProcessingResult with illumination corrected image
        """
        self._validate_image(image)

        # Parameters
        method = kwargs.get("method", "bandpass")
        large_structures = kwargs.get("large_structures", 40)
        small_structures = kwargs.get("small_structures", 3)
        suppress_stripes = kwargs.get("suppress_stripes", True)
        auto_scale = kwargs.get("auto_scale", True)

        # Store original for statistics
        original_image = image.copy()

        # Apply flattening method
        if method == "bandpass":
            flattened_image = self._bandpass_filter(
                image, large_structures, small_structures, suppress_stripes
            )
        elif method == "gaussian":
            flattened_image = self._gaussian_background_subtraction(
                image, large_structures
            )
        elif method == "sliding_paraboloid":
            flattened_image = self._sliding_paraboloid(image, large_structures)
        elif method == "rolling_ball":
            flattened_image = self._rolling_ball_background(image, large_structures)
        else:
            raise ValueError(f"Unknown flatten method: {method}")

        # Auto-scale if requested
        if auto_scale:
            flattened_image = self._auto_scale_result(original_image, flattened_image)

        # Calculate statistics
        stats = self._calculate_flatten_statistics(original_image, flattened_image)

        parameters = {
            "method": method,
            "large_structures": large_structures,
            "small_structures": small_structures,
            "suppress_stripes": suppress_stripes,
            "auto_scale": auto_scale,
        }

        result = ProcessingResult(
            image=flattened_image,
            processor_type=self.name,
            parameters=parameters,
            statistics=stats,
        )
        self._last_result = result
        return result

    def _bandpass_filter(
        self, image: np.ndarray, large: int, small: int, suppress_stripes: bool
    ) -> np.ndarray:
        """
        Apply bandpass filter (ImageJ method since v1.39f).
        Equivalent to large Gaussian blur - small Gaussian blur.
        """
        result = np.zeros_like(image, dtype=np.float64)

        for channel in range(3):
            channel_data = image[:, :, channel].astype(np.float64)

            # Large Gaussian (background estimation)
            if large > 0:
                large_blur = gaussian_filter(channel_data, sigma=large / 3.0)
            else:
                large_blur = channel_data

            # Small Gaussian (detail preservation)
            if small > 0:
                small_blur = gaussian_filter(channel_data, sigma=small / 3.0)
            else:
                small_blur = channel_data

            # Bandpass = original - (large - small)
            background = large_blur - small_blur
            filtered = channel_data - background

            # Suppress stripes if requested
            if suppress_stripes:
                filtered = self._suppress_stripes_fft(filtered)

            result[:, :, channel] = filtered

        return self._ensure_uint8_range(result + 128)  # Add offset for display

    def _gaussian_background_subtraction(
        self, image: np.ndarray, sigma: int
    ) -> np.ndarray:
        """Simple Gaussian background subtraction."""
        result = np.zeros_like(image, dtype=np.float64)

        for channel in range(3):
            channel_data = image[:, :, channel].astype(np.float64)
            background = gaussian_filter(channel_data, sigma=sigma / 3.0)
            result[:, :, channel] = channel_data - background

        return self._ensure_uint8_range(result + 128)

    def _sliding_paraboloid(self, image: np.ndarray, radius: int) -> np.ndarray:
        """
        Sliding paraboloid background subtraction.
        Based on ImageJ BackgroundSubtracter implementation.
        """
        result = np.zeros_like(image, dtype=np.float64)

        for channel in range(3):
            channel_data = image[:, :, channel].astype(np.float64)
            background = self._apply_sliding_paraboloid(channel_data, radius)
            result[:, :, channel] = channel_data - background

        return self._ensure_uint8_range(result + 128)

    def _apply_sliding_paraboloid(self, data: np.ndarray, radius: int) -> np.ndarray:
        """Apply sliding paraboloid algorithm to single channel."""
        height, width = data.shape
        background = np.zeros_like(data)

        # Create paraboloid kernel
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size))
        center = radius

        for i in range(kernel_size):
            for j in range(kernel_size):
                dist_sq = (i - center) ** 2 + (j - center) ** 2
                if dist_sq <= radius**2:
                    kernel[i, j] = radius**2 - dist_sq

        # Apply sliding paraboloid
        for y in range(height):
            for x in range(width):
                y_start = max(0, y - radius)
                y_end = min(height, y + radius + 1)
                x_start = max(0, x - radius)
                x_end = min(width, x + radius + 1)

                region = data[y_start:y_end, x_start:x_end]
                ky_start = max(0, radius - y)
                ky_end = ky_start + region.shape[0]
                kx_start = max(0, radius - x)
                kx_end = kx_start + region.shape[1]

                kernel_region = kernel[ky_start:ky_end, kx_start:kx_end]

                # Find maximum (paraboloid touches data from below)
                adjusted_region = region + kernel_region
                background[y, x] = np.max(adjusted_region) - kernel[center, center]

        return background

    def _rolling_ball_background(self, image: np.ndarray, radius: int) -> np.ndarray:
        """
        Rolling ball background subtraction.
        Simplified implementation of ImageJ's rolling ball algorithm.
        """
        result = np.zeros_like(image, dtype=np.float64)

        for channel in range(3):
            channel_data = image[:, :, channel].astype(np.float64)
            background = self._apply_rolling_ball(channel_data, radius)
            result[:, :, channel] = channel_data - background

        return self._ensure_uint8_range(result + 128)

    def _apply_rolling_ball(self, data: np.ndarray, radius: int) -> np.ndarray:
        """Apply rolling ball algorithm to single channel."""
        # Simplified rolling ball using morphological opening
        # Create ball-shaped structuring element
        ball_radius = max(1, radius // 2)

        # Use disk-shaped structuring element as approximation

        # Create circular structuring element
        y, x = np.ogrid[: 2 * ball_radius + 1, : 2 * ball_radius + 1]
        mask = (x - ball_radius) ** 2 + (y - ball_radius) ** 2 <= ball_radius**2
        structuring_element = mask.astype(np.uint8)

        # Apply morphological opening (erosion followed by dilation)
        from scipy.ndimage import grey_opening

        background = grey_opening(data, structure=structuring_element)

        return background

    def _suppress_stripes_fft(self, data: np.ndarray) -> np.ndarray:
        """Suppress horizontal and vertical stripes using FFT."""
        # Apply FFT
        fft_data = np.fft.fft2(data)
        fft_shifted = np.fft.fftshift(fft_data)

        # Create mask to suppress horizontal and vertical frequencies
        rows, cols = data.shape
        mask = np.ones((rows, cols))

        # Suppress central horizontal and vertical lines
        center_row, center_col = rows // 2, cols // 2

        # Suppress horizontal stripes (vertical frequencies)
        mask[center_row - 2 : center_row + 3, :] *= 0.1

        # Suppress vertical stripes (horizontal frequencies)
        mask[:, center_col - 2 : center_col + 3] *= 0.1

        # Apply mask and inverse FFT
        fft_filtered = fft_shifted * mask
        fft_ishifted = np.fft.ifftshift(fft_filtered)
        result = np.real(np.fft.ifft2(fft_ishifted))

        return result

    def _auto_scale_result(
        self, original: np.ndarray, processed: np.ndarray
    ) -> np.ndarray:
        """Auto-scale the result to match original dynamic range."""
        processed_float = processed.astype(np.float64)

        # Calculate scaling factor to match original range
        original_range = np.max(original) - np.min(original)
        processed_range = np.max(processed_float) - np.min(processed_float)

        if processed_range > 1e-6:
            scale_factor = original_range / processed_range
            offset = np.min(original) - scale_factor * np.min(processed_float)

            scaled = processed_float * scale_factor + offset
        else:
            scaled = processed_float

        return self._ensure_uint8_range(scaled)

    def _calculate_flatten_statistics(
        self, original: np.ndarray, processed: np.ndarray
    ) -> dict:
        """Calculate flattening statistics."""

        # Convert to grayscale for analysis
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float64)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY).astype(np.float64)

        # Calculate illumination uniformity
        original_uniformity = self._calculate_uniformity(original_gray)
        processed_uniformity = self._calculate_uniformity(processed_gray)

        # Calculate local contrast
        original_contrast = self._calculate_local_contrast(original_gray)
        processed_contrast = self._calculate_local_contrast(processed_gray)

        return {
            "illumination_uniformity": {
                "original": original_uniformity,
                "processed": processed_uniformity,
                "improvement": processed_uniformity - original_uniformity,
            },
            "local_contrast": {
                "original": original_contrast,
                "processed": processed_contrast,
                "improvement": processed_contrast - original_contrast,
            },
            "overall_improvement": (processed_uniformity - original_uniformity)
            + (processed_contrast - original_contrast),
        }

    def _calculate_uniformity(self, image: np.ndarray) -> float:
        """Calculate illumination uniformity (inverse of variance)."""
        return float(1.0 / (1.0 + np.var(image)))

    def _calculate_local_contrast(self, image: np.ndarray) -> float:
        """Calculate local contrast using Laplacian variance."""
        laplacian = cv2.Laplacian(image.astype(np.uint8), cv2.CV_64F)
        return float(np.var(cast(np.ndarray, laplacian)))


class HueShiftProcessor(BaseProcessor):
    """
    Hue shift processor for color separation enhancement.
    Based on HSL color space manipulation.
    """

    def __init__(self):
        super().__init__("Hue Shift")

    def process(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """
        Apply hue shift for color enhancement.

        Args:
            image: Input RGB image
            hue_shift: Hue shift in degrees (-180 to 180, default: 0)
            target_hue: Target specific hue range (optional)
            hue_range: Range around target hue to affect (default: 60)
            saturation_boost: Saturation boost factor (default: 1.0)
            selective: Whether to apply shift selectively (default: False)

        Returns:
            ProcessingResult with hue shifted image
        """
        self._validate_image(image)

        # Parameters
        hue_shift = kwargs.get("hue_shift", 0)
        target_hue = kwargs.get("target_hue", None)
        hue_range = kwargs.get("hue_range", 60)
        saturation_boost = kwargs.get("saturation_boost", 1.0)
        selective = kwargs.get("selective", False)

        # Convert RGB to HSV for hue manipulation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float64)

        # Store original for statistics
        original_hsv = hsv_image.copy()

        # Apply hue shift
        if selective and target_hue is not None:
            # Selective hue shift
            hsv_image = self._selective_hue_shift(
                hsv_image, hue_shift, target_hue, hue_range, saturation_boost
            )
        else:
            # Global hue shift
            hsv_image = self._global_hue_shift(hsv_image, hue_shift, saturation_boost)

        # Convert back to RGB
        # Ensure HSV values are in correct range
        hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0], 0, 179)  # Hue: 0-179
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)  # Saturation: 0-255
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)  # Value: 0-255

        result_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Calculate statistics
        stats = self._calculate_hue_statistics(
            original_hsv, hsv_image, image, result_image
        )

        parameters = {
            "hue_shift": hue_shift,
            "target_hue": target_hue,
            "hue_range": hue_range,
            "saturation_boost": saturation_boost,
            "selective": selective,
        }

        return ProcessingResult(
            image=result_image,
            processor_type=self.name,
            parameters=parameters,
            statistics=stats,
        )

    def _global_hue_shift(
        self, hsv_image: np.ndarray, hue_shift: float, saturation_boost: float
    ) -> np.ndarray:
        """Apply global hue shift to entire image."""

        result = hsv_image.copy()

        # Shift hue (OpenCV uses 0-179 range for hue)
        hue_shift_cv = (hue_shift * 179) / 360  # Convert degrees to OpenCV range
        result[:, :, 0] = (result[:, :, 0] + hue_shift_cv) % 180

        # Boost saturation if requested
        if saturation_boost != 1.0:
            result[:, :, 1] = np.clip(result[:, :, 1] * saturation_boost, 0, 255)

        return result

    def _selective_hue_shift(
        self,
        hsv_image: np.ndarray,
        hue_shift: float,
        target_hue: float,
        hue_range: float,
        saturation_boost: float,
    ) -> np.ndarray:
        """Apply selective hue shift to specific hue range."""

        result = hsv_image.copy()

        # Convert target hue to OpenCV range
        target_hue_cv = (target_hue * 179) / 360
        hue_range_cv = (hue_range * 179) / 360
        hue_shift_cv = (hue_shift * 179) / 360

        # Create mask for target hue range
        hue_channel = hsv_image[:, :, 0]

        # Handle wraparound in hue space
        lower_bound = (target_hue_cv - hue_range_cv / 2) % 180
        upper_bound = (target_hue_cv + hue_range_cv / 2) % 180

        if lower_bound < upper_bound:
            mask = (hue_channel >= lower_bound) & (hue_channel <= upper_bound)
        else:  # Wraparound case
            mask = (hue_channel >= lower_bound) | (hue_channel <= upper_bound)

        # Apply shift only to masked areas
        result[:, :, 0] = np.where(
            mask, (hue_channel + hue_shift_cv) % 180, hue_channel
        )

        # Apply saturation boost to masked areas
        if saturation_boost != 1.0:
            result[:, :, 1] = np.where(
                mask,
                np.clip(hsv_image[:, :, 1] * saturation_boost, 0, 255),
                hsv_image[:, :, 1],
            )

        return result

    def _calculate_hue_statistics(
        self,
        original_hsv: np.ndarray,
        processed_hsv: np.ndarray,
        original_rgb: np.ndarray,
        processed_rgb: np.ndarray,
    ) -> dict:
        """Calculate hue shift statistics."""

        # Hue distribution analysis
        original_hues = original_hsv[:, :, 0].flatten()
        processed_hues = processed_hsv[:, :, 0].flatten()

        # Calculate hue histogram
        hue_bins = np.arange(0, 181, 10)  # 18 bins for hue
        original_hist, _ = np.histogram(original_hues, bins=hue_bins)
        processed_hist, _ = np.histogram(processed_hues, bins=hue_bins)

        # Calculate dominant hues
        original_dominant = hue_bins[np.argmax(original_hist)]
        processed_dominant = hue_bins[np.argmax(processed_hist)]

        # Color separation metrics
        original_color_variance = self._calculate_color_variance(original_rgb)
        processed_color_variance = self._calculate_color_variance(processed_rgb)

        return {
            "hue_distribution": {
                "original_dominant_hue": float(original_dominant * 360 / 179),
                "processed_dominant_hue": float(processed_dominant * 360 / 179),
                "hue_shift_applied": float(
                    (processed_dominant - original_dominant) * 360 / 179
                ),
            },
            "color_separation": {
                "original_variance": original_color_variance,
                "processed_variance": processed_color_variance,
                "improvement_ratio": processed_color_variance
                / max(original_color_variance, 1e-6),
            },
            "saturation_change": {
                "original_mean": float(np.mean(original_hsv[:, :, 1])),
                "processed_mean": float(np.mean(processed_hsv[:, :, 1])),
                "boost_factor": float(
                    np.mean(processed_hsv[:, :, 1])
                    / max(np.mean(original_hsv[:, :, 1]), 1e-6)
                ),
            },
        }

    def _calculate_color_variance(self, rgb_image: np.ndarray) -> float:
        """Calculate color variance as measure of color separation."""
        # Calculate variance in each channel
        r_var = np.var(rgb_image[:, :, 0])
        g_var = np.var(rgb_image[:, :, 1])
        b_var = np.var(rgb_image[:, :, 2])

        # Return total color variance
        return float(r_var + g_var + b_var)


class ProcessorFactory:
    """Factory class for creating processor instances."""

    _processors = {
        ProcessorType.AUTO_CONTRAST: AutoContrastProcessor,
        ProcessorType.COLOR_BALANCE: ColorBalanceProcessor,
        ProcessorType.FLATTEN: FlattenProcessor,
        ProcessorType.INVERT: InvertProcessor,
        ProcessorType.HUE_SHIFT: HueShiftProcessor,
    }

    @classmethod
    def create_processor(cls, processor_type: ProcessorType) -> BaseProcessor:
        """Create a processor instance by type."""
        if processor_type not in cls._processors:
            raise ValueError(f"Unknown processor type: {processor_type}")

        return cls._processors[processor_type]()

    @classmethod
    def get_available_processors(cls) -> list[ProcessorType]:
        """Get list of available processor types."""
        return list(cls._processors.keys())

    @classmethod
    def create_all_processors(cls) -> dict[ProcessorType, BaseProcessor]:
        """Create instances of all available processors."""
        return {
            processor_type: cls.create_processor(processor_type)
            for processor_type in cls._processors.keys()
        }


class PreprocessingPipeline:
    """
    Pipeline for applying multiple preprocessing steps in sequence.
    This is the main interface for the advanced processing tools.
    """

    def __init__(self):
        self.processors = ProcessorFactory.create_all_processors()
        self.processing_history = []

    def process(
        self, image: np.ndarray, config: dict[str, Any]
    ) -> tuple[np.ndarray, list[ProcessingResult]]:
        """
        Apply preprocessing pipeline to image.

        Args:
            image: Input RGB image
            config: Configuration dictionary with processor settings

        Returns:
            Tuple of (processed_image, processing_results)
        """
        if image is None or len(image.shape) != 3:
            raise ValueError("Invalid input image")

        current_image = image.copy()
        results = []

        # Define processing order (important for optimal results)
        processing_order = [
            ProcessorType.FLATTEN,  # First: correct illumination
            ProcessorType.COLOR_BALANCE,  # Second: correct color cast
            ProcessorType.AUTO_CONTRAST,  # Third: optimize contrast
            ProcessorType.HUE_SHIFT,  # Fourth: adjust hue if needed
            ProcessorType.INVERT,  # Last: invert if requested
        ]

        logger.info(f"Starting preprocessing pipeline with {len(config)} processors")

        for processor_type in processing_order:
            processor_name = processor_type.value

            if processor_name in config and config[processor_name].get(
                "enabled", False
            ):
                try:
                    logger.info(f"Applying {processor_name} processor")

                    processor = self.processors[processor_type]
                    processor_params = config[processor_name].copy()
                    processor_params.pop("enabled", None)  # Remove 'enabled' flag

                    result = processor.process(current_image, **processor_params)
                    current_image = result.image
                    results.append(result)

                    logger.info(f"Successfully applied {processor_name}")

                except Exception as e:
                    logger.error(f"Error applying {processor_name}: {str(e)}")
                    # Continue with next processor instead of failing completely
                    continue

        self.processing_history.append(
            {
                "config": config,
                "results": results,
                "final_image_shape": current_image.shape,
            }
        )

        logger.info(
            f"Preprocessing pipeline completed with {len(results)} processors applied"
        )

        return current_image, results

    def get_processor_info(self, processor_type: ProcessorType) -> dict[str, Any]:
        """Get information about a specific processor."""
        if processor_type not in self.processors:
            raise ValueError(f"Unknown processor type: {processor_type}")

        processor = self.processors[processor_type]

        return {
            "name": processor.name,
            "type": processor_type.value,
            "description": processor.__doc__ or "No description available",
            "available": True,
        }

    def get_all_processors_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all available processors."""
        return {
            processor_type.value: self.get_processor_info(processor_type)
            for processor_type in self.processors.keys()
        }

    def clear_history(self):
        """Clear processing history."""
        self.processing_history.clear()
        logger.info("Processing history cleared")


# Utility functions for external use
def create_preprocessing_config(**kwargs) -> dict[str, Any]:
    """
    Create a preprocessing configuration dictionary.

    Args:
        **kwargs: Processor-specific parameters

    Returns:
        Configuration dictionary for preprocessing pipeline
    """
    config = {}

    # Auto Contrast configuration
    if kwargs.get("auto_contrast", False):
        config["auto_contrast"] = {
            "enabled": True,
            "saturated_pixels": kwargs.get("auto_contrast_saturated", 0.35),
            "normalize": kwargs.get("auto_contrast_normalize", True),
            "equalize": kwargs.get("auto_contrast_equalize", False),
        }

    # Color Balance configuration
    if kwargs.get("color_balance", False):
        config["color_balance"] = {
            "enabled": True,
            "method": kwargs.get("color_balance_method", "gray_world"),
            "strength": kwargs.get("color_balance_strength", 0.8),
            "temperature": kwargs.get("color_balance_temperature", 0.0),
            "tint": kwargs.get("color_balance_tint", 0.0),
            "preserve_luminance": kwargs.get("color_balance_preserve_lum", True),
            "percentile_clip": kwargs.get("color_balance_clip", 1.0),
        }

    # Flatten configuration
    if kwargs.get("flatten", False):
        config["flatten"] = {
            "enabled": True,
            "method": kwargs.get("flatten_method", "bandpass"),
            "large_structures": kwargs.get("flatten_large", 40),
            "small_structures": kwargs.get("flatten_small", 3),
            "suppress_stripes": kwargs.get("flatten_suppress_stripes", True),
            "auto_scale": kwargs.get("flatten_auto_scale", True),
        }

    # Hue Shift configuration
    if kwargs.get("hue_shift", False):
        config["hue_shift"] = {
            "enabled": True,
            "hue_shift": kwargs.get("hue_shift_degrees", 0),
            "target_hue": kwargs.get("hue_shift_target", None),
            "hue_range": kwargs.get("hue_shift_range", 60),
            "saturation_boost": kwargs.get("hue_shift_saturation", 1.0),
            "selective": kwargs.get("hue_shift_selective", False),
        }

    # Invert configuration
    if kwargs.get("invert", False):
        config["invert"] = {"enabled": True}

    return config


def quick_enhance(image: np.ndarray, enhancement_type: str = "balanced") -> np.ndarray:
    """
    Apply quick enhancement presets.

    Args:
        image: Input RGB image
        enhancement_type: 'balanced', 'contrast', 'color', 'illumination'

    Returns:
        Enhanced image
    """
    pipeline = PreprocessingPipeline()

    if enhancement_type == "balanced":
        config = create_preprocessing_config(
            auto_contrast=True, color_balance=True, flatten=True
        )
    elif enhancement_type == "contrast":
        config = create_preprocessing_config(
            auto_contrast=True, auto_contrast_saturated=0.5
        )
    elif enhancement_type == "color":
        config = create_preprocessing_config(
            color_balance=True, color_balance_strength=1.0
        )
    elif enhancement_type == "illumination":
        config = create_preprocessing_config(flatten=True, flatten_method="bandpass")
    else:
        raise ValueError(f"Unknown enhancement type: {enhancement_type}")

    enhanced_image, _ = pipeline.process(image, config)
    return enhanced_image


# Export main classes and functions
__all__ = [
    "BaseProcessor",
    "InvertProcessor",
    "AutoContrastProcessor",
    "ColorBalanceProcessor",
    "FlattenProcessor",
    "HueShiftProcessor",
    "ProcessorFactory",
    "PreprocessingPipeline",
    "ProcessorType",
    "ProcessingResult",
    "create_preprocessing_config",
    "quick_enhance",
]


if __name__ == "__main__":
    # Example usage and testing

    # Test with a simple synthetic image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Create pipeline
    pipeline = PreprocessingPipeline()

    # Test each processor individually
    for processor_type in ProcessorType:
        try:
            processor = ProcessorFactory.create_processor(processor_type)
            result = processor.process(test_image)
            print(f"✓ {processor.name} processor working correctly")
            print(f"  Output shape: {result.image.shape}")
            print(f"  Statistics: {len(result.statistics)} metrics")
        except Exception as e:
            print(f"✗ {processor_type.name} processor failed: {e}")

    # Test full pipeline
    try:
        config = create_preprocessing_config(
            auto_contrast=True, color_balance=True, flatten=True
        )

        processed_image, results = pipeline.process(test_image, config)
        print("\n✓ Full pipeline working correctly")
        print(f"  Applied {len(results)} processors")
        print(f"  Output shape: {processed_image.shape}")

    except Exception as e:
        print(f"\n✗ Full pipeline failed: {e}")

    print(
        f"\n📊 Available processors: {len(ProcessorFactory.get_available_processors())}"
    )
    for processor_type in ProcessorFactory.get_available_processors():
        info = pipeline.get_processor_info(processor_type)
        print(f"   - {info['name']}: {processor_type.value}")

    print("\n🎯 Independent processors module loaded successfully!")
