#!/usr/bin/env python3
"""
DStretch Python - Invert Processor
Implements ImageJ Edit > Invert functionality for DStretch ecosystem

Based on ImageJ's simple inversion: pixel_out = 255 - pixel_in
Maintains compatibility with original DStretch workflow.
"""

import logging

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class InvertProcessor:
    """
    Implements ImageJ-compatible image inversion functionality.

    Provides multiple inversion modes:
    - Full inversion (all channels)
    - Luminance-only inversion (preserving chromaticity)
    - Channel-selective inversion
    """

    def __init__(self, invert_mode: str = "full", preserve_hue: bool = False):
        """
        Initialize Invert Processor.

        Args:
            invert_mode: 'full' | 'luminance_only' | 'selective'
            preserve_hue: Whether to preserve hue information during inversion
        """
        self.invert_mode = invert_mode
        self.preserve_hue = preserve_hue

        # Validate parameters
        valid_modes = ["full", "luminance_only", "selective"]
        if invert_mode not in valid_modes:
            raise ValueError(f"Invalid invert_mode. Must be one of: {valid_modes}")

    def process(
        self, image: np.ndarray, selective_channels: list | None = None
    ) -> np.ndarray:
        """
        Apply inversion to image using specified mode.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            selective_channels: For selective mode, list of channels to invert [0, 1, 2]

        Returns:
            numpy.ndarray: Inverted image with same dtype as input

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
            f"Processing inversion - Mode: {self.invert_mode}, "
            f"Shape: {image.shape}, Preserve hue: {self.preserve_hue}"
        )

        # Store original dtype for output
        original_dtype = image.dtype

        # Convert to working format (float64 for precision)
        working_image = image.astype(np.float64)

        # Apply inversion based on mode
        if self.invert_mode == "full":
            result = self._invert_full(working_image)
        elif self.invert_mode == "luminance_only":
            result = self._invert_luminance_only(working_image)
        elif self.invert_mode == "selective":
            result = self._invert_selective(working_image, selective_channels)
        else:
            raise ValueError(f"Unsupported invert mode: {self.invert_mode}")

        # Convert back to original dtype
        if original_dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif original_dtype == np.uint16:
            result = np.clip(result, 0, 65535).astype(np.uint16)
        else:
            result = result.astype(original_dtype)

        logger.info("Inversion completed successfully")
        return result

    def _invert_full(self, image: np.ndarray) -> np.ndarray:
        """
        Full inversion: pixel_out = max_value - pixel_in for all channels.
        This is the standard ImageJ Edit > Invert behavior.
        """
        # Determine max value based on image range
        if image.dtype == np.uint8 or np.max(image) <= 255:
            max_value = 255.0
        elif image.dtype == np.uint16 or np.max(image) <= 65535:
            max_value = 65535.0
        else:
            max_value = np.max(image)

        return max_value - image

    def _invert_luminance_only(self, image: np.ndarray) -> np.ndarray:
        """
        Invert only luminance component, preserving chromaticity.
        Useful for archaeological images where color information is critical.
        """
        if len(image.shape) == 2:
            # Grayscale image - same as full inversion
            return self._invert_full(image)

        # For color images, work in LAB space to separate luminance
        lab_image = self._rgb_to_lab(image)

        # Invert only L* channel (luminance)
        lab_image[:, :, 0] = 100.0 - lab_image[:, :, 0]

        # Convert back to RGB
        return self._lab_to_rgb(lab_image)

    def _invert_selective(
        self, image: np.ndarray, channels: list | None = None
    ) -> np.ndarray:
        """
        Invert only selected channels.
        """
        if len(image.shape) == 2:
            # Grayscale image - full inversion
            return self._invert_full(image)

        if channels is None:
            channels = [0, 1, 2]  # Default to all channels

        result = image.copy()
        max_value = 255.0 if np.max(image) <= 255 else np.max(image)

        for channel in channels:
            if 0 <= channel < image.shape[2]:
                result[:, :, channel] = max_value - image[:, :, channel]

        return result

    def _rgb_to_lab(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB to LAB color space.
        Simplified implementation for luminance inversion.
        """
        # Normalize RGB to [0, 1]
        rgb_norm = rgb_image / 255.0

        # Apply gamma correction (sRGB to linear)
        def gamma_correction(channel):
            # Ensure values are in valid range [0, 1]
            channel_clipped = np.clip(channel, 0.0, 1.0)
            return np.where(
                channel_clipped <= 0.04045,
                channel_clipped / 12.92,
                np.power((channel_clipped + 0.055) / 1.055, 2.4),
            )

        rgb_linear = np.stack(
            [gamma_correction(rgb_norm[:, :, i]) for i in range(3)], axis=2
        )

        # Convert to XYZ (using sRGB matrix)
        xyz_matrix = np.array(
            [
                [0.4124, 0.3576, 0.1805],
                [0.2126, 0.7152, 0.0722],
                [0.0193, 0.1192, 0.9505],
            ]
        )

        xyz = np.dot(rgb_linear, xyz_matrix.T) * 100

        # Convert XYZ to LAB
        def f_transform(t):
            delta = 6.0 / 29.0
            # Clip to avoid negative values in cube root
            t_clipped = np.clip(t, 1e-10, np.inf)
            return np.where(
                t_clipped > delta**3,
                np.power(t_clipped, 1.0 / 3.0),
                t_clipped / (3 * delta**2) + 4.0 / 29.0,
            )

        # D65 illuminant
        xn, yn, zn = 95.047, 100.0, 108.883

        fx = f_transform(xyz[:, :, 0] / xn)
        fy = f_transform(xyz[:, :, 1] / yn)
        fz = f_transform(xyz[:, :, 2] / zn)

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return np.stack([L, a, b], axis=2)

    def _lab_to_rgb(self, lab_image: np.ndarray) -> np.ndarray:
        """
        Convert LAB back to RGB color space.
        """
        L, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]

        # LAB to XYZ
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        def f_inverse(t):
            delta = 6.0 / 29.0
            # Ensure input is within valid range
            t_clipped = np.clip(t, 0.0, 1.0)
            return np.where(
                t_clipped > delta,
                np.power(t_clipped, 3),
                3 * delta**2 * (t_clipped - 4.0 / 29.0),
            )

        # D65 illuminant
        xn, yn, zn = 95.047, 100.0, 108.883

        X = xn * f_inverse(fx)
        Y = yn * f_inverse(fy)
        Z = zn * f_inverse(fz)

        xyz = np.stack([X, Y, Z], axis=2) / 100

        # XYZ to RGB
        rgb_matrix = np.array(
            [
                [3.2406, -1.5372, -0.4986],
                [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570],
            ]
        )

        rgb_linear = np.dot(xyz, rgb_matrix.T)

        # Apply inverse gamma correction
        def inverse_gamma(channel):
            # Clip values to valid range [0, 1] to avoid negative values in power function
            channel_clipped = np.clip(channel, 0.0, 1.0)
            return np.where(
                channel_clipped <= 0.0031308,
                12.92 * channel_clipped,
                1.055 * np.power(channel_clipped, 1.0 / 2.4) - 0.055,
            )

        rgb_corrected = np.stack(
            [inverse_gamma(rgb_linear[:, :, i]) for i in range(3)], axis=2
        )

        return np.clip(rgb_corrected * 255, 0, 255)
