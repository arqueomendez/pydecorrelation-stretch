"""
Core decorrelation stretch algorithm implementation - FINAL CORRECTED VERSION 4.0

This version incorporates the final correction: using the `scale_adjust_factor`
from the colorspace definitions to replicate the original Java plugin's behavior
of varying enhancement intensity across different colorspace families.

LEGACY COMPATIBILITY VERSION - Uses independent processors internally

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1
"""

from typing import Any, cast

import cv2
import numpy as np
from scipy.linalg import eigh

from .colorspaces import COLORSPACES, BuiltinMatrixColorspace, ColorspaceManager


class ProcessingResult:
    """Result container for decorrelation stretch processing."""

    def __init__(
        self,
        processed_image,
        original_image,
        colorspace,
        scale,
        final_matrix,
        color_mean,
    ):
        self.processed_image = processed_image
        self.original_image = original_image
        self.colorspace = colorspace
        self.scale = scale
        self.final_matrix = final_matrix
        self.color_mean = color_mean

    def save(self, filepath: str):
        """Save processed image to file."""
        cv2.imwrite(filepath, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))


class DecorrelationStretch:
    """
    Legacy DecorrelationStretch class for backward compatibility.

    This class maintains the original API but internally uses the new
    independent processors architecture.
    """

    def __init__(self):
        self.colorspaces = COLORSPACES
        self.colorspace_manager = ColorspaceManager()
        self._last_original = None
        self._last_processed = None

        # Initialize independent processors (lazy loading)
        self._invert_processor = None
        self._auto_contrast_processor = None
        self._color_balance_processor = None
        self._flatten_processor = None

    def reset_to_original(self):
        """Return the last original image used."""
        return self._last_original

    def _get_invert_processor(self):
        """Get invert processor with lazy loading."""
        if self._invert_processor is None:
            from .independent_processors import InvertProcessor

            self._invert_processor = InvertProcessor()
        return self._invert_processor

    def _get_auto_contrast_processor(self):
        """Get auto contrast processor with lazy loading."""
        if self._auto_contrast_processor is None:
            from .independent_processors import AutoContrastProcessor

            self._auto_contrast_processor = AutoContrastProcessor()
        return self._auto_contrast_processor

    def _get_color_balance_processor(self):
        """Get color balance processor with lazy loading."""
        if self._color_balance_processor is None:
            from .independent_processors import ColorBalanceProcessor

            self._color_balance_processor = ColorBalanceProcessor()
        return self._color_balance_processor

    def _get_flatten_processor(self):
        """Get flatten processor with lazy loading."""
        if self._flatten_processor is None:
            from .independent_processors import FlattenProcessor

            self._flatten_processor = FlattenProcessor()
        return self._flatten_processor

    def process(
        self,
        image: np.ndarray,
        colorspace: str = "YDS",
        scale: float = 15.0,
        selection_mask: np.ndarray | None = None,
    ) -> ProcessingResult:
        """
        Process image with decorrelation stretch.

        Args:
            image: Input RGB image
            colorspace: Colorspace for analysis
            scale: Enhancement scale factor
            selection_mask: Optional selection mask for analysis

        Returns:
            ProcessingResult: Result with processed image and metadata
        """
        self._validate_inputs(image, colorspace, scale)
        self._last_original = image.copy()

        colorspace_obj = self.colorspaces[colorspace]

        if isinstance(colorspace_obj, BuiltinMatrixColorspace):
            # RUTA 2: MATRIZ PREDEFINIDA
            base_cs_name = colorspace_obj.base_colorspace_name
            base_cs_obj = self.colorspaces[base_cs_name]
            base_image = base_cs_obj.to_colorspace(image)
            pixel_data = self._get_analysis_data(base_image, selection_mask)
            color_mean = np.mean(pixel_data, axis=0)
            transform_matrix = colorspace_obj.matrix * (scale / 10.0)
            processed_base = self._apply_transformation(
                base_image, transform_matrix, color_mean
            )
            processed_rgb = base_cs_obj.from_colorspace(processed_base)
            final_matrix_for_result = transform_matrix
        else:
            # RUTA 1: ANÁLISIS ESTADÍSTICO
            adjusted_scale = scale * colorspace_obj.scale_adjust_factor

            transformed_image = colorspace_obj.to_colorspace(image)
            pixel_data = self._get_analysis_data(transformed_image, selection_mask)
            color_mean, covariance_matrix = self._calculate_statistics(pixel_data)
            eigenvalues, eigenvectors = self._eigendecomposition(covariance_matrix)

            stretch_matrix = self._build_stretch_matrix(eigenvalues, adjusted_scale)
            transform_matrix = eigenvectors @ stretch_matrix @ eigenvectors.T

            processed_transformed = self._apply_transformation(
                transformed_image, transform_matrix, color_mean
            )
            processed_rgb = colorspace_obj.from_colorspace(processed_transformed)
            final_matrix_for_result = transform_matrix

        self._last_processed = processed_rgb

        return ProcessingResult(
            processed_image=processed_rgb,
            original_image=image,
            colorspace=colorspace,
            scale=scale,
            final_matrix=final_matrix_for_result,
            color_mean=color_mean,
        )

    def _validate_inputs(self, image: np.ndarray, colorspace: str, scale: float):
        """Validate input parameters."""
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise ValueError(
                "Image must be a numpy array of shape (H, W, 3) and dtype uint8"
            )
        if colorspace not in self.colorspaces:
            raise ValueError(
                f"Unknown colorspace '{colorspace}'. Available: {list(self.colorspaces.keys())}"
            )
        if not 1.0 <= scale <= 100.0:
            raise ValueError("Scale must be between 1.0 and 100.0")

    def _get_analysis_data(
        self, transformed_image: np.ndarray, selection_mask: np.ndarray | None
    ) -> np.ndarray:
        """Extract pixel data for analysis."""
        if selection_mask is not None:
            if (
                selection_mask.shape[:2] != transformed_image.shape[:2]
                or selection_mask.dtype != bool
            ):
                raise ValueError(
                    "Selection mask must match image dimensions and be boolean"
                )
            return transformed_image[selection_mask]
        else:
            return transformed_image.reshape(-1, 3)

    def _calculate_statistics(
        self, pixel_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate statistics for decorrelation."""
        data = pixel_data.astype(np.float64)
        color_mean = np.mean(data, axis=0)
        covariance = np.cov(data.T)
        return color_mean, covariance

    def _eigendecomposition(
        self, covariance_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform eigendecomposition of covariance matrix."""
        eigenvalues, eigenvectors = eigh(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    def _build_stretch_matrix(
        self, eigenvalues: np.ndarray, scale: float
    ) -> np.ndarray:
        """Build stretch matrix from eigenvalues."""
        eigenvalues[eigenvalues < 1e-10] = 1e-10
        stretch_factors = scale / np.sqrt(eigenvalues)
        return np.diag(stretch_factors)

    def _apply_transformation(
        self,
        transformed_image: np.ndarray,
        transform_matrix: np.ndarray,
        color_mean: np.ndarray,
    ) -> np.ndarray:
        """Apply transformation matrix to image."""
        original_shape = transformed_image.shape
        flat_image = transformed_image.reshape(-1, 3).astype(np.float64)
        centered_data = flat_image - color_mean
        processed_flat = (transform_matrix @ centered_data.T).T
        processed_flat += color_mean
        return processed_flat.reshape(original_shape)

    # Legacy compatibility methods using independent processors

    def apply_invert(
        self,
        image: np.ndarray,
        invert_mode: str = "full",
        preserve_hue: bool = False,
        selective_channels: list | None = None,
    ) -> np.ndarray:
        """
        Apply inversion to image using independent processor.

        Args:
            image: Input image array
            invert_mode: 'full' | 'luminance_only' | 'selective'
            preserve_hue: Whether to preserve hue information
            selective_channels: For selective mode, list of channels to invert

        Returns:
            Inverted image array
        """
        processor = self._get_invert_processor()
        result = processor.process(
            image,
            invert_mode=invert_mode,
            preserve_hue=preserve_hue,
            selective_channels=selective_channels,
        )
        return result.image

    def apply_auto_contrast(
        self,
        image: np.ndarray,
        clip_percentage: float = 0.1,
        preserve_colors: bool = True,
    ) -> np.ndarray:
        """
        Apply auto contrast enhancement using independent processor.

        Args:
            image: Input image array
            clip_percentage: Percentage of pixels to ignore at extremes (0.0-5.0)
            preserve_colors: Whether to preserve color relationships during stretch

        Returns:
            Auto contrast enhanced image array
        """
        processor = self._get_auto_contrast_processor()
        result = processor.process(image, saturated_pixels=clip_percentage)
        return result.image

    def get_contrast_statistics(self, image: np.ndarray) -> dict:
        """
        Get statistics about image contrast for analysis.

        Args:
            image: Input image array

        Returns:
            Dictionary with contrast statistics
        """
        processor = self._get_auto_contrast_processor()
        result = processor.process(image)  # Process to generate statistics
        return cast(Any, result).statistics

    def apply_color_balance(
        self,
        image: np.ndarray,
        method: str = "gray_world",
        clip_percentage: float = 0.1,
        strength: float = 1.0,
        preserve_luminance: bool = True,
        preserve_colors: bool = True,
        temperature_offset: float = 0.0,
        tint_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Apply color balance correction using independent processor.

        Args:
            image: Input image array
            method: 'gray_world' | 'white_patch' | 'manual'
            clip_percentage: Percentage of pixels to clip at extremes (0.0-5.0)
            strength: Overall effect strength (0.0-2.0)
            preserve_luminance: Whether to preserve original luminance
            preserve_colors: Whether to preserve original saturation
            temperature_offset: Manual temperature adjustment (-100 to +100)
            tint_offset: Manual tint adjustment (-100 to +100)

        Returns:
            Color balanced image array
        """
        processor = self._get_color_balance_processor()
        result = processor.process(
            image,
            method=method,
            percentile_clip=clip_percentage,
            strength=strength,
            preserve_luminance=preserve_luminance,
            temperature=temperature_offset,
            tint=tint_offset,
        )
        return result.image

    def get_color_balance_statistics(self) -> dict:
        """
        Get statistics from the last color balance operation.

        Returns:
            Dictionary with color balance statistics
        """
        processor = self._get_color_balance_processor()
        result = processor.get_last_result()
        return cast(Any, result).statistics if result else {}

    def analyze_color_cast(
        self, image: np.ndarray, clip_percentage: float = 0.1
    ) -> dict:
        """
        Analyze color cast in an image without applying correction.

        Args:
            image: Input image array
            clip_percentage: Percentage of pixels to clip for analysis

        Returns:
            Dictionary with color cast analysis information
        """
        # This would require adding a method to ColorBalanceProcessor
        # For now, return basic color statistics
        flat_image = image.reshape(-1, 3).astype(np.float32)
        channel_means = np.mean(flat_image, axis=0)
        color_cast = np.std(channel_means)

        return {
            "channel_means": channel_means.tolist(),
            "color_cast_magnitude": float(color_cast),
            "dominant_cast": "red"
            if channel_means[0] > channel_means.max() * 0.9
            else "green"
            if channel_means[1] > channel_means.max() * 0.9
            else "blue"
            if channel_means[2] > channel_means.max() * 0.9
            else "neutral",
        }

    def apply_flatten(
        self,
        image: np.ndarray,
        method: str = "bandpass_filter",
        filter_large: float = 40.0,
        filter_small: float = 3.0,
        suppress_stripes: bool = True,
        tolerance: float = 5.0,
        autoscale_result: bool = True,
        preview_background: bool = False,
        ball_radius: float = 50.0,
        paraboloid_radius: float = 50.0,
    ) -> np.ndarray:
        """
        Apply flatten correction for uneven illumination using independent processor.

        Args:
            image: Input image array
            method: 'bandpass_filter' | 'gaussian_background' | 'sliding_paraboloid' | 'rolling_ball'
            filter_large: Large structures to remove (pixels)
            filter_small: Small structures to preserve (pixels)
            suppress_stripes: Suppress horizontal/vertical stripes
            tolerance: Direction tolerance for stripe suppression (%)
            autoscale_result: Automatically scale result to full range
            preview_background: Show background instead of corrected image
            ball_radius: Radius for rolling ball method (pixels)
            paraboloid_radius: Radius for sliding paraboloid (pixels)

        Returns:
            Flattened image array
        """
        processor = self._get_flatten_processor()
        result = processor.process(
            image,
            method=method,
            large_structures=int(filter_large),
            small_structures=int(filter_small),
            suppress_stripes=suppress_stripes,
            auto_scale=autoscale_result,
        )
        return result.image

    def get_flatten_statistics(self) -> dict:
        """
        Get statistics from the last flatten operation.

        Returns:
            Dictionary with flatten statistics
        """
        processor = self._get_flatten_processor()
        result = processor.get_last_result()
        if result and hasattr(result, "statistics"):
            return cast(Any, result).statistics
        return {}

    def get_background_estimate(self) -> np.ndarray | None:
        """
        Get the background estimate from the last flatten operation.

        Returns:
            Background estimate array or None if no flatten operation performed
        """
        # processor = self._get_flatten_processor()
        # return processor.get_background_estimate()
        return None

    def analyze_illumination(self, image: np.ndarray) -> dict:
        """
        Analyze illumination uniformity without applying correction.

        Args:
            image: Input image array

        Returns:
            Dictionary with illumination analysis information
        """
        # Basic illumination analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate illumination statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        min_brightness = np.min(gray)
        max_brightness = np.max(gray)

        # Calculate uniformity (coefficient of variation)
        uniformity = std_brightness / mean_brightness if mean_brightness > 0 else 0

        return {
            "mean_brightness": float(mean_brightness),
            "std_brightness": float(std_brightness),
            "brightness_range": [int(min_brightness), int(max_brightness)],
            "uniformity_coefficient": float(uniformity),
            "illumination_quality": "uniform"
            if uniformity < 0.3
            else "moderate"
            if uniformity < 0.6
            else "uneven",
        }


# Convenience function for simple processing
def process_image(
    image: np.ndarray, colorspace: str = "YDS", scale: float = 15.0
) -> np.ndarray:
    """
    Simple function to process an image with decorrelation stretch.

    Args:
        image: Input RGB image
        colorspace: Colorspace for analysis
        scale: Enhancement scale factor

    Returns:
        Processed image
    """
    dstretch = DecorrelationStretch()
    result = dstretch.process(image, colorspace, scale)
    return result.processed_image
