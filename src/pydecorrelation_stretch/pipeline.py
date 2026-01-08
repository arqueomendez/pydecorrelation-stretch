#!/usr/bin/env python3
"""
DStretch Pipeline - Refactored Architecture for Independent Processing

This module implements the corrected pipeline architecture where independent
processing tools operate on RGB images BEFORE decorrelation stretch, matching
the workflow of DStretch ImageJ.

Author: Claude (DStretch Migration Project)
Version: 2.0 - Independent Architecture

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1
"""

from typing import Any

import numpy as np

from .decorrelation import DecorrelationStretch, ProcessingResult
from .independent_processors import (
    AutoContrastProcessor,
    ColorBalanceProcessor,
    FlattenProcessor,
    InvertProcessor,
)
from .independent_processors import (
    ProcessingResult as ProcessorResult,
)


class DStretchPipeline:
    """
    Main DStretch pipeline with corrected architecture.

    Pipeline Order:
    1. Pre-processing (independent tools on RGB)
    2. Decorrelation Stretch (core algorithm)
    3. Post-processing (optional)
    """

    def __init__(self):
        # Core decorrelation algorithm
        self.decorrelation = DecorrelationStretch()

        # Independent processors
        self.invert_processor = InvertProcessor()
        self.auto_contrast_processor = AutoContrastProcessor()
        self.color_balance_processor = ColorBalanceProcessor()
        self.flatten_processor = FlattenProcessor()

        # Pipeline state
        self._last_original = None
        self._last_preprocessed = None
        self._last_decorrelation_result = None
        self._preprocessing_results = []

    def process_complete(
        self,
        image: np.ndarray,
        preprocessing_steps: list[dict[str, Any]] | None = None,
        colorspace: str = "YDS",
        scale: float = 15.0,
        selection_mask: np.ndarray | None = None,
    ) -> "CompletePipelineResult":
        """
        Complete processing pipeline with pre-processing and decorrelation.

        Args:
            image: Input RGB image
            preprocessing_steps: List of preprocessing configurations
            colorspace: Colorspace for decorrelation stretch
            scale: Scale factor for decorrelation stretch
            selection_mask: Optional selection mask for analysis

        Returns:
            Complete pipeline result with all intermediate steps
        """
        self._validate_inputs(image, colorspace, scale)
        self._last_original = image.copy()

        # Step 1: Apply preprocessing (if requested)
        if preprocessing_steps:
            current_image, preprocessing_results = self._apply_preprocessing(
                image, preprocessing_steps
            )
            self._preprocessing_results = preprocessing_results
        else:
            current_image = image.copy()
            self._preprocessing_results = []

        self._last_preprocessed = current_image.copy()

        # Step 2: Apply decorrelation stretch to preprocessed image
        decorrelation_result = self.decorrelation.process(
            current_image, colorspace, scale, selection_mask
        )
        self._last_decorrelation_result = decorrelation_result

        # Create complete result
        complete_result = CompletePipelineResult(
            original_image=image,
            preprocessed_image=current_image,
            final_image=decorrelation_result.processed_image,
            preprocessing_results=self._preprocessing_results,
            decorrelation_result=decorrelation_result,
            pipeline_config={
                "preprocessing_steps": preprocessing_steps or [],
                "colorspace": colorspace,
                "scale": scale,
            },
        )

        return complete_result

    def process_decorrelation_only(
        self,
        image: np.ndarray,
        colorspace: str = "YDS",
        scale: float = 15.0,
        selection_mask: np.ndarray | None = None,
    ) -> ProcessingResult:
        """
        Apply only decorrelation stretch without preprocessing.

        Args:
            image: Input RGB image
            colorspace: Colorspace for decorrelation stretch
            scale: Scale factor for decorrelation stretch
            selection_mask: Optional selection mask for analysis

        Returns:
            Decorrelation processing result
        """
        return self.decorrelation.process(image, colorspace, scale, selection_mask)

    def apply_preprocessing_only(
        self, image: np.ndarray, preprocessing_steps: list[dict[str, Any]]
    ) -> tuple[np.ndarray, list[ProcessorResult]]:
        """
        Apply only preprocessing steps without decorrelation.

        Args:
            image: Input RGB image
            preprocessing_steps: List of preprocessing configurations

        Returns:
            Tuple of (preprocessed_image, list_of_processor_results)
        """
        return self._apply_preprocessing(image, preprocessing_steps)

    def _apply_preprocessing(
        self, image: np.ndarray, preprocessing_steps: list[dict[str, Any]]
    ) -> tuple[np.ndarray, list[ProcessorResult]]:
        """Apply preprocessing steps in sequence."""
        current_image = image.copy()
        results = []

        for step_config in preprocessing_steps:
            processor_type = step_config.get("type")
            params = step_config.get("params", {})

            if processor_type == "invert":
                result = self.invert_processor.process(current_image, **params)
                processed_image = result.image
                results.append(result)

            elif processor_type == "auto_contrast":
                result = self.auto_contrast_processor.process(current_image, **params)
                processed_image = result.image
                results.append(result)

            elif processor_type == "color_balance":
                result = self.color_balance_processor.process(current_image, **params)
                processed_image = result.image
                results.append(result)

            elif processor_type == "flatten":
                result = self.flatten_processor.process(current_image, **params)
                processed_image = result.image
                results.append(result)

            else:
                raise ValueError(f"Unknown preprocessing type: {processor_type}")

            current_image = processed_image

        return current_image, results

    def _validate_inputs(self, image: np.ndarray, colorspace: str, scale: float):
        """Validate pipeline inputs."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a numpy array of shape (H, W, 3)")
        if image.dtype != np.uint8:
            raise ValueError("Image must be uint8")
        if not 1.0 <= scale <= 100.0:
            raise ValueError("Scale must be between 1.0 and 100.0")
        if colorspace not in self.decorrelation.colorspaces:
            raise ValueError(f"Unknown colorspace '{colorspace}'")

    # Convenience methods for individual processors

    def invert(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply inversion as independent preprocessing step."""
        return self.invert_processor.process(image, **kwargs).image

    def auto_contrast(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply auto contrast as independent preprocessing step."""
        return self.auto_contrast_processor.process(image, **kwargs).image

    def color_balance(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply color balance as independent preprocessing step."""
        return self.color_balance_processor.process(image, **kwargs).image

    def flatten(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply flatten as independent preprocessing step."""
        return self.flatten_processor.process(image, **kwargs).image

    # Access to processing results

    def get_last_preprocessing_results(self) -> list[ProcessorResult]:
        """Get results from last preprocessing pipeline."""
        return self._preprocessing_results

    def get_last_decorrelation_result(self) -> ProcessingResult | None:
        """Get result from last decorrelation operation."""
        return self._last_decorrelation_result

    def get_available_colorspaces(self) -> list[str]:
        """Get list of available colorspaces for decorrelation."""
        return list(self.decorrelation.colorspaces.keys())

    def get_available_preprocessors(self) -> dict[str, str]:
        """Get list of available preprocessing tools."""
        return {
            "invert": "Image inversion with multiple modes",
            "auto_contrast": "Automatic contrast enhancement",
            "color_balance": "Color balance correction",
            "flatten": "Uneven illumination correction",
        }


class CompletePipelineResult:
    """
    Result container for complete pipeline processing.
    Contains all intermediate results and metadata.
    """

    def __init__(
        self,
        original_image: np.ndarray,
        preprocessed_image: np.ndarray,
        final_image: np.ndarray,
        preprocessing_results: list[ProcessorResult],
        decorrelation_result: ProcessingResult,
        pipeline_config: dict[str, Any],
    ):
        self.original_image = original_image
        self.preprocessed_image = preprocessed_image
        self.final_image = final_image
        self.preprocessing_results = preprocessing_results
        self.decorrelation_result = decorrelation_result
        self.pipeline_config = pipeline_config

    def save_final(self, filepath: str):
        """Save final processed image."""
        import cv2

        cv2.imwrite(filepath, cv2.cvtColor(self.final_image, cv2.COLOR_RGB2BGR))

    def save_preprocessed(self, filepath: str):
        """Save preprocessed image (before decorrelation)."""
        import cv2

        cv2.imwrite(filepath, cv2.cvtColor(self.preprocessed_image, cv2.COLOR_RGB2BGR))

    def get_pipeline_summary(self) -> dict[str, Any]:
        """Get summary of entire pipeline processing."""
        preprocessing_summary = []
        for result in self.preprocessing_results:
            preprocessing_summary.append(
                {
                    "processor": result.processor_type,
                    "parameters": result.parameters,
                    "statistics": result.statistics,
                }
            )

        return {
            "preprocessing_steps": len(self.preprocessing_results),
            "preprocessing_summary": preprocessing_summary,
            "decorrelation_colorspace": self.decorrelation_result.colorspace,
            "decorrelation_scale": self.decorrelation_result.scale,
            "final_transform_matrix": self.decorrelation_result.final_matrix.tolist(),
            "pipeline_config": self.pipeline_config,
        }

    def has_preprocessing(self) -> bool:
        """Check if preprocessing was applied."""
        return len(self.preprocessing_results) > 0

    def get_preprocessing_names(self) -> list[str]:
        """Get names of applied preprocessing steps."""
        return [result.processor_type for result in self.preprocessing_results]


# Convenience functions for common workflows


def create_preprocessing_config(
    invert: bool = False,
    auto_contrast: bool = False,
    color_balance: bool = False,
    flatten: bool = False,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Create preprocessing configuration from boolean flags.

    Args:
        invert: Apply inversion
        auto_contrast: Apply auto contrast
        color_balance: Apply color balance
        flatten: Apply flatten
        **kwargs: Additional parameters for processors

    Returns:
        List of preprocessing step configurations
    """
    steps = []

    if invert:
        steps.append({"type": "invert", "params": kwargs.get("invert_params", {})})

    if auto_contrast:
        steps.append(
            {"type": "auto_contrast", "params": kwargs.get("auto_contrast_params", {})}
        )

    if color_balance:
        steps.append(
            {"type": "color_balance", "params": kwargs.get("color_balance_params", {})}
        )

    if flatten:
        steps.append({"type": "flatten", "params": kwargs.get("flatten_params", {})})

    return steps


def quick_enhance(
    image: np.ndarray,
    enhancement_type: str = "standard",
    colorspace: str = "YDS",
    scale: float = 15.0,
) -> CompletePipelineResult:
    """
    Quick enhancement presets for common archaeological image types.

    Args:
        image: Input RGB image
        enhancement_type: 'standard', 'faint_reds', 'yellows', 'high_contrast'
        colorspace: Colorspace override (optional)
        scale: Scale override (optional)

    Returns:
        Complete pipeline result
    """
    pipeline = DStretchPipeline()

    if enhancement_type == "standard":
        # Basic enhancement for general archaeological images
        preprocessing = create_preprocessing_config(
            auto_contrast=True,
            auto_contrast_params={"clip_percentage": 0.1, "preserve_colors": True},
        )
        colorspace = colorspace or "YDS"

    elif enhancement_type == "faint_reds":
        # Optimized for faint red pictographs
        preprocessing = create_preprocessing_config(
            color_balance=True,
            auto_contrast=True,
            color_balance_params={"method": "gray_world", "strength": 0.8},
            auto_contrast_params={"clip_percentage": 0.2, "preserve_colors": True},
        )
        colorspace = colorspace or "CRGB"

    elif enhancement_type == "yellows":
        # Optimized for yellow pigments
        preprocessing = create_preprocessing_config(
            flatten=True,
            auto_contrast=True,
            flatten_params={"method": "bandpass_filter", "filter_large": 50.0},
            auto_contrast_params={"clip_percentage": 0.1, "preserve_colors": True},
        )
        colorspace = colorspace or "LDS"

    elif enhancement_type == "high_contrast":
        # For high contrast images with good preservation
        preprocessing = create_preprocessing_config(
            flatten=True,
            color_balance=True,
            auto_contrast=True,
            flatten_params={"method": "gaussian_background", "filter_large": 40.0},
            color_balance_params={"method": "white_patch", "strength": 0.6},
            auto_contrast_params={"clip_percentage": 0.05, "preserve_colors": True},
        )
        colorspace = colorspace or "LRE"

    else:
        # Fallback to minimal processing
        preprocessing = []

    return pipeline.process_complete(image, preprocessing, colorspace, scale)


# Legacy compatibility functions


def process_image_legacy(
    image: np.ndarray, colorspace: str = "YDS", scale: float = 15.0
) -> ProcessingResult:
    """
    Legacy function for backward compatibility.
    Applies only decorrelation stretch without preprocessing.
    """
    pipeline = DStretchPipeline()
    return pipeline.process_decorrelation_only(image, colorspace, scale)
