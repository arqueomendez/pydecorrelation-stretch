"""
Pixel Inspector module for DStretch Python.

Provides real-time pixel analysis and color space conversion for archaeological image analysis.
"""

import colorsys
import math
import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np


class ColorSpaceConverter:
    """Handles conversions between different color spaces."""

    @staticmethod
    def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
        """Convert RGB (0-255) to HSV (H: 0-360°, S: 0-100%, V: 0-100%)."""
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360.0, s * 100.0, v * 100.0

    @staticmethod
    def rgb_to_lab(r: float, g: float, b: float) -> tuple[float, float, float]:
        """Convert RGB (0-255) to LAB color space."""
        # First convert to XYZ
        r, g, b = r / 255.0, g / 255.0, b / 255.0

        # Apply gamma correction (sRGB to linear)
        def gamma_correct(c):
            if c <= 0.04045:
                return c / 12.92
            else:
                return math.pow((c + 0.055) / 1.055, 2.4)

        r_linear = gamma_correct(r)
        g_linear = gamma_correct(g)
        b_linear = gamma_correct(b)

        # Convert to XYZ using sRGB matrix
        x = r_linear * 0.4124 + g_linear * 0.3576 + b_linear * 0.1805
        y = r_linear * 0.2126 + g_linear * 0.7152 + b_linear * 0.0722
        z = r_linear * 0.0193 + g_linear * 0.1192 + b_linear * 0.9505

        # Normalize by D65 illuminant
        x /= 0.95047
        y /= 1.00000
        z /= 1.08883

        # Convert to LAB
        def f(t):
            if t > 0.008856:
                return math.pow(t, 1 / 3)
            else:
                return 7.787 * t + 16 / 116

        fx, fy, fz = f(x), f(y), f(z)

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return L, a, b

    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int) -> str:
        """Convert RGB to hexadecimal format."""
        return f"#{r:02X}{g:02X}{b:02X}"


class PixelAnalyzer:
    """Analyzes pixel values and provides statistical information."""

    def __init__(self):
        self.sampling_size = 1  # 1x1, 3x3, or 5x5

    def analyze_pixel(
        self, image: np.ndarray, x: int, y: int, sampling_size: int = 1
    ) -> dict[str, Any]:
        """Analyze pixel(s) at given coordinates with optional sampling area."""
        if image is None:
            return {}

        height, width = image.shape[:2]

        # Validate coordinates
        if not (0 <= x < width and 0 <= y < height):
            return {}

        # Calculate sampling area
        half_size = sampling_size // 2
        y_start = max(0, y - half_size)
        y_end = min(height, y + half_size + 1)
        x_start = max(0, x - half_size)
        x_end = min(width, x + half_size + 1)

        # Extract pixel region
        if sampling_size == 1:
            # Single pixel
            pixel_rgb = image[y, x]
            r, g, b = int(pixel_rgb[0]), int(pixel_rgb[1]), int(pixel_rgb[2])
            intensity = (r + g + b) / 3.0
        else:
            # Average of sampling area
            region = image[y_start:y_end, x_start:x_end]
            avg_rgb = np.mean(region, axis=(0, 1))
            r, g, b = int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2])
            intensity = np.mean(avg_rgb)

        # Convert to other color spaces
        h, s, v = ColorSpaceConverter.rgb_to_hsv(r, g, b)
        L, a, b_lab = ColorSpaceConverter.rgb_to_lab(r, g, b)
        hex_color = ColorSpaceConverter.rgb_to_hex(r, g, b)

        return {
            "coordinates": (x, y),
            "rgb": (r, g, b),
            "hsv": (h, s, v),
            "lab": (L, a, b_lab),
            "hex": hex_color,
            "intensity": intensity,
            "sampling_size": sampling_size,
        }


class PixelInspectorPanel:
    """GUI panel for real-time pixel inspection."""

    def __init__(self, parent, image_canvas):
        self.parent = parent
        self.image_canvas = image_canvas
        self.analyzer = PixelAnalyzer()

        # State
        self.current_image = None
        self.frozen_values = None
        self.is_frozen = False
        self.history = []  # Store last 5 inspected pixels
        self.sampling_size = 1
        self.zoom_controller = None  # Will be set by GUI

        # UI variables
        self.enabled_var = None
        self.sampling_var = None

        self._setup_panel()
        self._bind_events()

    def _setup_panel(self):
        """Setup the inspector panel UI."""
        # Main inspector frame
        self.inspector_frame = ttk.LabelFrame(
            self.parent, text="Pixel Inspector", padding="10"
        )

        # Enable/disable checkbox
        control_frame = ttk.Frame(self.inspector_frame)
        control_frame.grid(row=0, column=0, sticky="we", pady=(0, 10))

        self.enabled_var = tk.BooleanVar(value=True)
        enable_check = ttk.Checkbutton(
            control_frame,
            text="Enable",
            variable=self.enabled_var,
            command=self._toggle_inspector,
        )
        enable_check.grid(row=0, column=0, sticky=tk.W)

        # Sampling size selection
        ttk.Label(control_frame, text="Sample:").grid(
            row=0, column=1, sticky=tk.W, padx=(10, 5)
        )
        self.sampling_var = tk.StringVar(value="1x1")
        sampling_combo = ttk.Combobox(
            control_frame,
            textvariable=self.sampling_var,
            values=["1x1", "3x3", "5x5"],
            state="readonly",
            width=6,
        )
        sampling_combo.grid(row=0, column=2, sticky=tk.W)
        sampling_combo.bind("<<ComboboxSelected>>", self._on_sampling_change)

        # Coordinates display
        coord_frame = ttk.LabelFrame(self.inspector_frame, text="Position", padding="5")
        coord_frame.grid(row=1, column=0, sticky="we", pady=(0, 10))

        self.x_label = ttk.Label(coord_frame, text="X: --", font=("Courier", 10))
        self.x_label.grid(row=0, column=0, sticky=tk.W)

        self.y_label = ttk.Label(coord_frame, text="Y: --", font=("Courier", 10))
        self.y_label.grid(row=1, column=0, sticky=tk.W)

        # RGB values
        rgb_frame = ttk.LabelFrame(self.inspector_frame, text="RGB", padding="5")
        rgb_frame.grid(row=2, column=0, sticky="we", pady=(0, 10))

        self.r_label = ttk.Label(rgb_frame, text="R: --", font=("Courier", 10))
        self.r_label.grid(row=0, column=0, sticky=tk.W)

        self.g_label = ttk.Label(rgb_frame, text="G: --", font=("Courier", 10))
        self.g_label.grid(row=1, column=0, sticky=tk.W)

        self.b_label = ttk.Label(rgb_frame, text="B: --", font=("Courier", 10))
        self.b_label.grid(row=2, column=0, sticky=tk.W)

        # HSV values
        hsv_frame = ttk.LabelFrame(self.inspector_frame, text="HSV", padding="5")
        hsv_frame.grid(row=3, column=0, sticky="we", pady=(0, 10))

        self.h_label = ttk.Label(hsv_frame, text="H: --°", font=("Courier", 10))
        self.h_label.grid(row=0, column=0, sticky=tk.W)

        self.s_label = ttk.Label(hsv_frame, text="S: --%", font=("Courier", 10))
        self.s_label.grid(row=1, column=0, sticky=tk.W)

        self.v_label = ttk.Label(hsv_frame, text="V: --%", font=("Courier", 10))
        self.v_label.grid(row=2, column=0, sticky=tk.W)

        # LAB values
        lab_frame = ttk.LabelFrame(self.inspector_frame, text="LAB", padding="5")
        lab_frame.grid(row=4, column=0, sticky="we", pady=(0, 10))

        self.l_label = ttk.Label(lab_frame, text="L: --", font=("Courier", 10))
        self.l_label.grid(row=0, column=0, sticky=tk.W)

        self.a_label = ttk.Label(lab_frame, text="a: --", font=("Courier", 10))
        self.a_label.grid(row=1, column=0, sticky=tk.W)

        self.b_lab_label = ttk.Label(lab_frame, text="b: --", font=("Courier", 10))
        self.b_lab_label.grid(row=2, column=0, sticky=tk.W)

        # Additional info
        info_frame = ttk.LabelFrame(self.inspector_frame, text="Info", padding="5")
        info_frame.grid(row=5, column=0, sticky="we", pady=(0, 10))

        self.hex_label = ttk.Label(info_frame, text="Hex: --", font=("Courier", 10))
        self.hex_label.grid(row=0, column=0, sticky=tk.W)

        self.intensity_label = ttk.Label(
            info_frame, text="Int: --", font=("Courier", 10)
        )
        self.intensity_label.grid(row=1, column=0, sticky=tk.W)

        # Color preview
        self.color_frame = tk.Frame(
            info_frame, width=40, height=20, relief=tk.SUNKEN, bd=1
        )
        self.color_frame.grid(
            row=0, column=1, rowspan=2, sticky="ens", padx=(10, 0)
        )
        self.color_frame.grid_propagate(False)

        # Control buttons
        button_frame = ttk.Frame(self.inspector_frame)
        button_frame.grid(row=6, column=0, sticky="we", pady=(10, 0))

        self.freeze_button = ttk.Button(
            button_frame, text="Freeze", command=self._toggle_freeze, width=8
        )
        self.freeze_button.grid(row=0, column=0, sticky=tk.W)

        copy_button = ttk.Button(
            button_frame, text="Copy", command=self._copy_values, width=8
        )
        copy_button.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        # Configure grid weights
        self.inspector_frame.columnconfigure(0, weight=1)
        for frame in [coord_frame, rgb_frame, hsv_frame, lab_frame, info_frame]:
            frame.columnconfigure(0, weight=1)

    def _bind_events(self):
        """Bind mouse events to the image canvas."""
        self.image_canvas.bind("<Motion>", self._on_mouse_motion)
        self.image_canvas.bind("<Button-1>", self._on_mouse_click)
        self.image_canvas.bind("<Leave>", self._on_mouse_leave)

    def _toggle_inspector(self):
        """Toggle inspector enable/disable."""
        if self.enabled_var and not self.enabled_var.get():
            self._clear_display()

    def _on_sampling_change(self, event=None):
        """Handle sampling size change."""
        if not self.sampling_var:
            return
        size_map = {"1x1": 1, "3x3": 3, "5x5": 5}
        self.sampling_size = size_map[self.sampling_var.get()]

    def _on_mouse_motion(self, event):
        """Handle mouse motion over image canvas."""
        if (
            not self.enabled_var
            or not self.enabled_var.get()
            or self.is_frozen
            or self.current_image is None
        ):
            return

        # Convert canvas coordinates to image coordinates
        x, y = self._canvas_to_image_coords(event.x, event.y)
        if x is not None and y is not None:
            self._update_display(x, y)

    def _on_mouse_click(self, event):
        """Handle mouse click - toggle freeze."""
        if not self.enabled_var.get() or self.current_image is None:
            return

        x, y = self._canvas_to_image_coords(event.x, event.y)
        if x is not None and y is not None:
            if not self.is_frozen:
                self._freeze_values(x, y)
            else:
                self._unfreeze_values()

    def _on_mouse_leave(self, event):
        """Handle mouse leaving canvas."""
        if not self.enabled_var.get() or self.is_frozen:
            return
        self._clear_display()

    def _canvas_to_image_coords(
        self, canvas_x: int, canvas_y: int
    ) -> tuple[int | None, int | None]:
        """Convert canvas coordinates to image pixel coordinates."""
        if self.current_image is None:
            return None, None

        # Use zoom controller if available for accurate coordinate transformation
        if self.zoom_controller:
            return self.zoom_controller.get_image_coordinates(canvas_x, canvas_y)

        # Fallback to original method if zoom controller not available
        # Get canvas and image dimensions
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        img_height, img_width = self.current_image.shape[:2]

        # Calculate scaling (same logic as in display_image)
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y, 1.0)

        if scale < 1.0:
            display_width = int(img_width * scale)
            display_height = int(img_height * scale)
        else:
            display_width = img_width
            display_height = img_height

        # Calculate offset to center image
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2

        # Convert coordinates
        image_x = int((canvas_x - offset_x) / scale) if scale > 0 else 0
        image_y = int((canvas_y - offset_y) / scale) if scale > 0 else 0

        # Validate coordinates
        if 0 <= image_x < img_width and 0 <= image_y < img_height:
            return image_x, image_y
        else:
            return None, None

    def _update_display(self, x: int, y: int):
        """Update the display with pixel values at given coordinates."""
        if self.current_image is None:
            return
        pixel_data = self.analyzer.analyze_pixel(
            self.current_image, x, y, self.sampling_size
        )

        if pixel_data:
            self._display_pixel_data(pixel_data)
            self._add_to_history(pixel_data)

    def _display_pixel_data(self, pixel_data: dict[str, Any]):
        """Display pixel data in the UI labels."""
        coords = pixel_data["coordinates"]
        rgb = pixel_data["rgb"]
        hsv = pixel_data["hsv"]
        lab = pixel_data["lab"]
        hex_color = pixel_data["hex"]
        intensity = pixel_data["intensity"]

        # Update coordinate labels
        self.x_label.config(text=f"X: {coords[0]}")
        self.y_label.config(text=f"Y: {coords[1]}")

        # Update RGB labels
        self.r_label.config(text=f"R: {rgb[0]}")
        self.g_label.config(text=f"G: {rgb[1]}")
        self.b_label.config(text=f"B: {rgb[2]}")

        # Update HSV labels
        self.h_label.config(text=f"H: {hsv[0]:.0f}°")
        self.s_label.config(text=f"S: {hsv[1]:.0f}%")
        self.v_label.config(text=f"V: {hsv[2]:.0f}%")

        # Update LAB labels
        self.l_label.config(text=f"L: {lab[0]:.1f}")
        self.a_label.config(text=f"a: {lab[1]:.1f}")
        self.b_lab_label.config(text=f"b: {lab[2]:.1f}")

        # Update additional info
        self.hex_label.config(text=f"Hex: {hex_color}")
        self.intensity_label.config(text=f"Int: {intensity:.0f}")

        # Update color preview
        try:
            self.color_frame.config(bg=hex_color)
        except tk.TclError:
            # Invalid color, use default
            self.color_frame.config(bg="gray")

    def _freeze_values(self, x: int, y: int):
        """Freeze the current pixel values."""
        if self.current_image is None:
            return
        self.frozen_values = self.analyzer.analyze_pixel(
            self.current_image, x, y, self.sampling_size
        )
        self.is_frozen = True
        self.freeze_button.config(text="Unfreeze")

        if self.frozen_values:
            self._display_pixel_data(self.frozen_values)

    def _unfreeze_values(self):
        """Unfreeze pixel values."""
        self.is_frozen = False
        self.frozen_values = None
        self.freeze_button.config(text="Freeze")

    def _toggle_freeze(self):
        """Toggle freeze state."""
        if self.is_frozen:
            self._unfreeze_values()
        else:
            # Freeze current values if available
            if self.current_image is not None:
                # Use center of image as default
                height, width = self.current_image.shape[:2]
                self._freeze_values(width // 2, height // 2)

    def _copy_values(self):
        """Copy current pixel values to clipboard."""
        if self.is_frozen and self.frozen_values:
            data = self.frozen_values
        elif self.current_image is not None:
            # Get center pixel
            height, width = self.current_image.shape[:2]
            data = self.analyzer.analyze_pixel(
                self.current_image, width // 2, height // 2, self.sampling_size
            )
        else:
            return

        if data:
            # Format values for clipboard
            coords = data["coordinates"]
            rgb = data["rgb"]
            hsv = data["hsv"]
            lab = data["lab"]
            hex_color = data["hex"]

            clipboard_text = f"""Pixel Analysis:
Position: ({coords[0]}, {coords[1]})
RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}
HSV: {hsv[0]:.0f}°, {hsv[1]:.0f}%, {hsv[2]:.0f}%
LAB: {lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}
Hex: {hex_color}
Sampling: {data["sampling_size"]}x{data["sampling_size"]}"""

            # Copy to clipboard
            try:
                self.parent.clipboard_clear()
                self.parent.clipboard_append(clipboard_text)
            except Exception:
                pass  # Ignore clipboard errors

    def _add_to_history(self, pixel_data: dict[str, Any]):
        """Add pixel data to history (keep last 5)."""
        self.history.append(pixel_data)
        if len(self.history) > 5:
            self.history.pop(0)

    def _clear_display(self):
        """Clear all displayed values."""
        self.x_label.config(text="X: --")
        self.y_label.config(text="Y: --")
        self.r_label.config(text="R: --")
        self.g_label.config(text="G: --")
        self.b_label.config(text="B: --")
        self.h_label.config(text="H: --°")
        self.s_label.config(text="S: --%")
        self.v_label.config(text="V: --%")
        self.l_label.config(text="L: --")
        self.a_label.config(text="a: --")
        self.b_lab_label.config(text="b: --")
        self.hex_label.config(text="Hex: --")
        self.intensity_label.config(text="Int: --")
        self.color_frame.config(bg="lightgray")

    def set_image(self, image: np.ndarray):
        """Set the current image for analysis."""
        self.current_image = image
        if self.enabled_var and not self.enabled_var.get():
            self._clear_display()

    def set_zoom_controller(self, zoom_controller):
        """Set the zoom controller for coordinate transformation."""
        self.zoom_controller = zoom_controller

    def get_frame(self):
        """Get the main inspector frame for layout."""
        return self.inspector_frame

    def grid(self, **kwargs):
        """Grid the inspector frame."""
        self.inspector_frame.grid(**kwargs)
