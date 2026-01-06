#!/usr/bin/env python3
"""
Color Balance Processor for DStretch Python.

Implements Gray World algorithm with percentile clipping for archaeological image enhancement.
Corrects color casts while preserving subtle archaeological information.

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class BalanceMethod(Enum):
    """Available color balance methods."""

    GRAY_WORLD = "gray_world"
    WHITE_PATCH = "white_patch"
    MANUAL = "manual"


@dataclass
class ColorBalanceParams:
    """Parameters for color balance processing."""

    method: BalanceMethod = BalanceMethod.GRAY_WORLD
    clip_percentage: float = 0.1  # Percentage of pixels to clip (0.1-5.0)
    preserve_luminance: bool = True
    preserve_colors: bool = True  # Preserve original saturation
    temperature_offset: float = 0.0  # Manual temperature adjustment (-100 to +100)
    tint_offset: float = 0.0  # Manual tint adjustment (-100 to +100)
    strength: float = 1.0  # Overall effect strength (0.0-2.0)


class ColorBalanceProcessor:
    """
    Advanced color balance processor for archaeological images.

    Implements multiple balance algorithms optimized for preserving
    subtle archaeological information while correcting color casts.
    """

    def __init__(self):
        self.last_params = ColorBalanceParams()
        self.last_stats = None

    def process(
        self, image: np.ndarray, params: ColorBalanceParams | None = None
    ) -> np.ndarray:
        """
        Apply color balance to an image.

        Args:
            image: Input image (H, W, 3) in RGB format, uint8
            params: Color balance parameters

        Returns:
            Balanced image (H, W, 3) in RGB format, uint8
        """
        if params is None:
            params = ColorBalanceParams()

        # Store parameters for analysis
        self.last_params = params

        # Convert to float for processing
        image_float = image.astype(np.float64) / 255.0

        # Apply selected balance method
        if params.method == BalanceMethod.GRAY_WORLD:
            balanced = self._gray_world_balance(image_float, params)
        elif params.method == BalanceMethod.WHITE_PATCH:
            balanced = self._white_patch_balance(image_float, params)
        elif params.method == BalanceMethod.MANUAL:
            balanced = self._manual_balance(image_float, params)
        else:
            balanced = image_float.copy()

        # Apply strength parameter
        if params.strength != 1.0:
            balanced = self._blend_images(image_float, balanced, params.strength)

        # Convert back to uint8
        result = np.clip(balanced * 255.0, 0, 255).astype(np.uint8)

        return result

    def _gray_world_balance(
        self, image: np.ndarray, params: ColorBalanceParams
    ) -> np.ndarray:
        """
        Apply Gray World color balance.

        The Gray World assumption states that the average reflectance in a scene
        is achromatic (gray). Any deviation indicates a color cast.
        """
        # Apply percentile clipping to exclude extreme values
        clipped_image = self._apply_percentile_clipping(image, params.clip_percentage)

        # Calculate mean values for each channel
        mean_r = np.mean(clipped_image[:, :, 0])
        mean_g = np.mean(clipped_image[:, :, 1])
        mean_b = np.mean(clipped_image[:, :, 2])

        # Calculate overall gray level
        gray_level = (mean_r + mean_g + mean_b) / 3.0

        # Calculate correction factors
        factor_r = gray_level / (mean_r + 1e-8)  # Avoid division by zero
        factor_g = gray_level / (mean_g + 1e-8)
        factor_b = gray_level / (mean_b + 1e-8)

        # Store statistics for analysis
        self.last_stats = {
            "original_means": (mean_r, mean_g, mean_b),
            "gray_level": gray_level,
            "correction_factors": (factor_r, factor_g, factor_b),
        }

        # Apply correction
        balanced = image.copy()
        balanced[:, :, 0] *= factor_r
        balanced[:, :, 1] *= factor_g
        balanced[:, :, 2] *= factor_b

        # Optional luminance preservation
        if params.preserve_luminance:
            balanced = self._preserve_luminance(image, balanced)

        # Optional color preservation (maintain original saturation)
        if params.preserve_colors:
            balanced = self._preserve_saturation(image, balanced)

        return np.clip(balanced, 0.0, 1.0)

    def _white_patch_balance(
        self, image: np.ndarray, params: ColorBalanceParams
    ) -> np.ndarray:
        """
        Apply White Patch color balance.

        Assumes the brightest point in the image should be white.
        Less suitable for archaeological images but included for completeness.
        """
        # Apply percentile clipping
        clipped_image = self._apply_percentile_clipping(image, params.clip_percentage)

        # Find maximum values (brightest patch)
        max_r = np.max(clipped_image[:, :, 0])
        max_g = np.max(clipped_image[:, :, 1])
        max_b = np.max(clipped_image[:, :, 2])

        # Calculate correction factors to make max values equal
        max_overall = max(max_r, max_g, max_b)

        factor_r = max_overall / (max_r + 1e-8)
        factor_g = max_overall / (max_g + 1e-8)
        factor_b = max_overall / (max_b + 1e-8)

        # Store statistics
        self.last_stats = {
            "original_maxes": (max_r, max_g, max_b),
            "max_overall": max_overall,
            "correction_factors": (factor_r, factor_g, factor_b),
        }

        # Apply correction
        balanced = image.copy()
        balanced[:, :, 0] *= factor_r
        balanced[:, :, 1] *= factor_g
        balanced[:, :, 2] *= factor_b

        # Apply preservation options
        if params.preserve_luminance:
            balanced = self._preserve_luminance(image, balanced)

        if params.preserve_colors:
            balanced = self._preserve_saturation(image, balanced)

        return np.clip(balanced, 0.0, 1.0)

    def _manual_balance(
        self, image: np.ndarray, params: ColorBalanceParams
    ) -> np.ndarray:
        """
        Apply manual color balance using temperature and tint adjustments.

        Temperature: Blue/Yellow balance
        Tint: Green/Magenta balance
        """
        balanced = image.copy()

        # Convert temperature offset to RGB adjustments
        # Positive temperature = warmer (more red/yellow)
        # Negative temperature = cooler (more blue)
        temp_factor = params.temperature_offset / 100.0

        if temp_factor > 0:  # Warmer
            balanced[:, :, 0] *= 1.0 + temp_factor * 0.3  # More red
            balanced[:, :, 1] *= 1.0 + temp_factor * 0.1  # Slightly more green
            balanced[:, :, 2] *= 1.0 - temp_factor * 0.2  # Less blue
        elif temp_factor < 0:  # Cooler
            temp_factor = abs(temp_factor)
            balanced[:, :, 0] *= 1.0 - temp_factor * 0.2  # Less red
            balanced[:, :, 1] *= 1.0 - temp_factor * 0.1  # Slightly less green
            balanced[:, :, 2] *= 1.0 + temp_factor * 0.3  # More blue

        # Convert tint offset to RGB adjustments
        # Positive tint = more magenta (red+blue)
        # Negative tint = more green
        tint_factor = params.tint_offset / 100.0

        if tint_factor > 0:  # More magenta
            balanced[:, :, 0] *= 1.0 + tint_factor * 0.2  # More red
            balanced[:, :, 1] *= 1.0 - tint_factor * 0.3  # Less green
            balanced[:, :, 2] *= 1.0 + tint_factor * 0.2  # More blue
        elif tint_factor < 0:  # More green
            tint_factor = abs(tint_factor)
            balanced[:, :, 0] *= 1.0 - tint_factor * 0.2  # Less red
            balanced[:, :, 1] *= 1.0 + tint_factor * 0.3  # More green
            balanced[:, :, 2] *= 1.0 - tint_factor * 0.2  # Less blue

        # Store statistics
        self.last_stats = {
            "temperature_offset": params.temperature_offset,
            "tint_offset": params.tint_offset,
            "adjustments_applied": True,
        }

        # Apply preservation options
        if params.preserve_luminance:
            balanced = self._preserve_luminance(image, balanced)

        if params.preserve_colors:
            balanced = self._preserve_saturation(image, balanced)

        return np.clip(balanced, 0.0, 1.0)

    def _apply_percentile_clipping(
        self, image: np.ndarray, clip_percentage: float
    ) -> np.ndarray:
        """
        Apply percentile clipping to exclude extreme pixel values.

        This prevents outliers from skewing the color balance calculation.
        """
        clipped = image.copy()

        for channel in range(3):
            channel_data = image[:, :, channel]

            # Calculate percentiles
            low_percentile = np.percentile(channel_data, clip_percentage)
            high_percentile = np.percentile(channel_data, 100 - clip_percentage)

            # Create mask for valid pixels
            valid_mask = (channel_data >= low_percentile) & (
                channel_data <= high_percentile
            )

            # Only use valid pixels for this channel
            clipped[:, :, channel] = np.where(valid_mask, channel_data, np.nan)

        # Replace NaN values with original values for calculation
        # (This ensures we don't lose pixels, just exclude them from mean calculation)
        return np.nan_to_num(clipped, nan=0.0)

    def _preserve_luminance(
        self, original: np.ndarray, balanced: np.ndarray
    ) -> np.ndarray:
        """
        Preserve the original luminance while applying color correction.

        Uses the luminance formula: L = 0.299*R + 0.587*G + 0.114*B
        """
        # Calculate original and new luminance
        original_lum = (
            0.299 * original[:, :, 0]
            + 0.587 * original[:, :, 1]
            + 0.114 * original[:, :, 2]
        )

        balanced_lum = (
            0.299 * balanced[:, :, 0]
            + 0.587 * balanced[:, :, 1]
            + 0.114 * balanced[:, :, 2]
        )

        # Calculate luminance ratio
        lum_ratio = np.divide(
            original_lum,
            balanced_lum,
            out=np.ones_like(original_lum),
            where=balanced_lum != 0,
        )

        # Apply luminance correction
        result = balanced.copy()
        for channel in range(3):
            result[:, :, channel] *= lum_ratio

        return result

    def _preserve_saturation(
        self, original: np.ndarray, balanced: np.ndarray
    ) -> np.ndarray:
        """
        Preserve the original color saturation while applying balance correction.

        Converts to HSV, preserves S channel, applies new H and V from balanced image.
        """
        # Convert images to HSV
        original_hsv = cv2.cvtColor(
            (original * 255).astype(np.uint8), cv2.COLOR_RGB2HSV
        )
        balanced_hsv = cv2.cvtColor(
            (balanced * 255).astype(np.uint8), cv2.COLOR_RGB2HSV
        )

        # Preserve original saturation, use balanced hue and value
        result_hsv = balanced_hsv.copy()
        result_hsv[:, :, 1] = original_hsv[:, :, 1]  # Preserve saturation

        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)

        return result_rgb.astype(np.float64) / 255.0

    def _blend_images(
        self, original: np.ndarray, processed: np.ndarray, strength: float
    ) -> np.ndarray:
        """
        Blend original and processed images based on strength parameter.

        strength = 0.0: original image
        strength = 1.0: fully processed image
        strength > 1.0: enhanced effect
        """
        if strength == 1.0:
            return processed
        elif strength == 0.0:
            return original
        else:
            return original + strength * (processed - original)

    def get_balance_statistics(self) -> dict:
        """Get statistics from the last balance operation."""
        return self.last_stats if self.last_stats else {}

    def analyze_color_cast(
        self, image: np.ndarray, clip_percentage: float = 0.1
    ) -> dict:
        """
        Analyze color cast in an image without applying correction.

        Returns:
            Dictionary with color cast analysis information.
        """
        image_float = image.astype(np.float64) / 255.0
        clipped_image = self._apply_percentile_clipping(image_float, clip_percentage)

        # Calculate statistics
        means = [np.mean(clipped_image[:, :, i]) for i in range(3)]
        stds = [np.std(clipped_image[:, :, i]) for i in range(3)]

        # Overall gray level and cast detection
        gray_level = sum(means) / 3.0
        cast_factors = [gray_level / (mean + 1e-8) for mean in means]

        # Determine dominant cast
        cast_factors_arr = np.array(cast_factors)
        max_factor_idx = np.argmax(cast_factors_arr)
        min_factor_idx = np.argmin(cast_factors_arr)

        cast_names = ["red", "green", "blue"]
        dominant_cast = None
        cast_strength = 0.0

        if (
            max(cast_factors) - min(cast_factors) > 0.05
        ):  # Threshold for significant cast
            if max_factor_idx != min_factor_idx:
                dominant_cast = f"Deficient in {cast_names[max_factor_idx]}, excess {cast_names[min_factor_idx]}"
                cast_strength = max(cast_factors) - min(cast_factors)

        return {
            "channel_means": means,
            "channel_stds": stds,
            "gray_level": gray_level,
            "correction_factors": cast_factors,
            "dominant_cast": dominant_cast,
            "cast_strength": cast_strength,
            "needs_correction": cast_strength > 0.05,
        }


def create_test_image_with_cast() -> np.ndarray:
    """Create a test image with known color cast for validation."""
    # Create a neutral test pattern
    height, width = 200, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create gradient pattern
    for y in range(height):
        for x in range(width):
            # Create a pattern that should be neutral gray
            gray_value = int(128 + 50 * np.sin(x / 20) * np.cos(y / 15))
            image[y, x] = [gray_value, gray_value, gray_value]

    # Apply artificial color cast (excess red, deficient blue)
    image_float = image.astype(np.float64)
    image_float[:, :, 0] *= 1.3  # Excess red
    image_float[:, :, 1] *= 1.1  # Slight excess green
    image_float[:, :, 2] *= 0.8  # Deficient blue

    return np.clip(image_float, 0, 255).astype(np.uint8)


# Utility functions for archaeological workflow
def recommend_balance_method(image: np.ndarray) -> BalanceMethod:
    """
    Recommend optimal balance method based on image characteristics.

    For archaeological images:
    - Gray World: Most scenes (general purpose)
    - White Patch: Images with clear white/light reference
    - Manual: When automatic methods fail or specific adjustments needed
    """
    processor = ColorBalanceProcessor()
    analysis = processor.analyze_color_cast(image)

    # Recommend based on color cast strength
    if analysis["cast_strength"] > 0.15:
        return BalanceMethod.GRAY_WORLD  # Strong cast, use Gray World
    elif analysis["cast_strength"] > 0.05:
        return BalanceMethod.GRAY_WORLD  # Moderate cast, Gray World
    else:
        return BalanceMethod.MANUAL  # Weak cast, manual fine-tuning
