"""
Zoom & Pan Controller for DStretch Python.

Provides interactive zoom and pan functionality for detailed image analysis.
"""

import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import ttk

from PIL import Image, ImageTk


@dataclass
class ViewState:
    """Represents the current view state of the image."""

    zoom_factor: float = 1.0
    pan_x: float = 0.0  # Pan offset in image coordinates
    pan_y: float = 0.0
    canvas_width: int = 400
    canvas_height: int = 300


class CoordinateTransformer:
    """Handles coordinate transformations between canvas and image space."""

    def __init__(self):
        self.view_state = ViewState()
        self.image_width = 0
        self.image_height = 0

    def set_image_size(self, width: int, height: int):
        """Set the size of the current image."""
        self.image_width = width
        self.image_height = height

    def set_canvas_size(self, width: int, height: int):
        """Set the size of the canvas."""
        self.view_state.canvas_width = width
        self.view_state.canvas_height = height

    def canvas_to_image(
        self, canvas_x: int, canvas_y: int
    ) -> tuple[int | None, int | None]:
        """Convert canvas coordinates to image pixel coordinates."""
        if self.image_width == 0 or self.image_height == 0:
            return None, None

        # Calculate the displayed image size
        display_width = self.image_width * self.view_state.zoom_factor
        display_height = self.image_height * self.view_state.zoom_factor

        # Calculate offset to center image (when smaller than canvas)
        offset_x = 0
        offset_y = 0

        if display_width < self.view_state.canvas_width:
            offset_x = (self.view_state.canvas_width - display_width) / 2
        if display_height < self.view_state.canvas_height:
            offset_y = (self.view_state.canvas_height - display_height) / 2

        # Apply pan offset
        effective_x = canvas_x - offset_x + self.view_state.pan_x
        effective_y = canvas_y - offset_y + self.view_state.pan_y

        # Convert to image coordinates
        image_x = effective_x / self.view_state.zoom_factor
        image_y = effective_y / self.view_state.zoom_factor

        # Validate coordinates
        if 0 <= image_x < self.image_width and 0 <= image_y < self.image_height:
            return int(image_x), int(image_y)
        else:
            return None, None

    def image_to_canvas(self, image_x: int, image_y: int) -> tuple[int, int]:
        """Convert image coordinates to canvas coordinates."""
        # Scale by zoom factor
        canvas_x = image_x * self.view_state.zoom_factor
        canvas_y = image_y * self.view_state.zoom_factor

        # Apply pan offset
        canvas_x -= self.view_state.pan_x
        canvas_y -= self.view_state.pan_y

        # Apply centering offset (when image is smaller than canvas)
        display_width = self.image_width * self.view_state.zoom_factor
        display_height = self.image_height * self.view_state.zoom_factor

        if display_width < self.view_state.canvas_width:
            canvas_x += (self.view_state.canvas_width - display_width) / 2
        if display_height < self.view_state.canvas_height:
            canvas_y += (self.view_state.canvas_height - display_height) / 2

        return int(canvas_x), int(canvas_y)


class ImageRenderer:
    """Handles optimized image rendering at different zoom levels."""

    def __init__(self):
        self.image_cache = {}  # Cache for different zoom levels
        self.max_cache_size = 10

    def render_image(
        self, pil_image: Image.Image, view_state: ViewState
    ) -> tuple[ImageTk.PhotoImage, int, int]:
        """Render image at current zoom and pan level."""
        if pil_image is None:
            return None, 0, 0

        # Calculate target size
        target_width = int(pil_image.width * view_state.zoom_factor)
        target_height = int(pil_image.height * view_state.zoom_factor)

        # Create cache key
        cache_key = (id(pil_image), view_state.zoom_factor, target_width, target_height)

        # Check cache first
        if cache_key in self.image_cache:
            scaled_image = self.image_cache[cache_key]
        else:
            # Choose resampling method based on zoom level
            if view_state.zoom_factor < 1.0:
                # Zooming out - use high quality resampling
                resample = Image.Resampling.LANCZOS
            elif view_state.zoom_factor > 4.0:
                # High zoom - use nearest neighbor to preserve pixel boundaries
                resample = Image.Resampling.NEAREST
            else:
                # Medium zoom - use bicubic
                resample = Image.Resampling.BICUBIC

            # Scale the image
            scaled_image = pil_image.resize((target_width, target_height), resample)

            # Cache the result (limit cache size)
            if len(self.image_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]

            self.image_cache[cache_key] = scaled_image

        # Calculate visible region based on pan
        crop_x = max(0, int(view_state.pan_x))
        crop_y = max(0, int(view_state.pan_y))
        crop_width = min(target_width - crop_x, view_state.canvas_width)
        crop_height = min(target_height - crop_y, view_state.canvas_height)

        # Crop to visible area if image is larger than canvas
        if (
            target_width > view_state.canvas_width
            or target_height > view_state.canvas_height
        ):
            visible_image = scaled_image.crop(
                (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
            )
        else:
            visible_image = scaled_image

        # Convert to PhotoImage
        photo_image = ImageTk.PhotoImage(visible_image)

        return photo_image, visible_image.width, visible_image.height

    def clear_cache(self):
        """Clear the image cache."""
        self.image_cache.clear()


class ZoomPanController:
    """Main controller for zoom and pan functionality."""

    # Predefined zoom levels
    ZOOM_LEVELS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    MIN_ZOOM = 0.1
    MAX_ZOOM = 16.0

    def __init__(self, canvas: tk.Canvas, update_callback: Callable | None = None):
        self.canvas = canvas
        self.update_callback = update_callback

        # Core components
        self.transformer = CoordinateTransformer()
        self.renderer = ImageRenderer()

        # State
        self.current_image = None
        self.current_photo_image = None
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_start_pan_x = 0
        self.drag_start_pan_y = 0

        # Bind events
        self._bind_events()

    def _bind_events(self):
        """Bind mouse and keyboard events."""
        # Mouse wheel for zoom
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)  # Linux
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)  # Linux

        # Right mouse button for pan (to avoid conflict with Inspector Pixels)
        self.canvas.bind("<Button-3>", self._on_drag_start)
        self.canvas.bind("<B3-Motion>", self._on_drag_motion)
        self.canvas.bind("<ButtonRelease-3>", self._on_drag_end)

        # Alternative: Shift + Left mouse button for pan
        self.canvas.bind("<Shift-Button-1>", self._on_drag_start)
        self.canvas.bind("<Shift-B1-Motion>", self._on_drag_motion)
        self.canvas.bind("<Shift-ButtonRelease-1>", self._on_drag_end)

        # Canvas resize
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Focus for keyboard events
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set())

    def set_image(self, pil_image: Image.Image):
        """Set the current image."""
        self.current_image = pil_image
        self.renderer.clear_cache()

        if pil_image:
            self.transformer.set_image_size(pil_image.width, pil_image.height)
            # Reset view state for new image
            self.transformer.view_state = ViewState(
                zoom_factor=1.0,
                pan_x=0.0,
                pan_y=0.0,
                canvas_width=self.canvas.winfo_width(),
                canvas_height=self.canvas.winfo_height(),
            )
            self.fit_to_window()
        else:
            self.transformer.set_image_size(0, 0)

        self._update_display()

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        if event.widget == self.canvas:
            self.transformer.set_canvas_size(event.width, event.height)
            self._update_display()

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel zoom."""
        if self.current_image is None:
            return

        # Determine zoom direction
        if event.delta > 0 or event.num == 4:
            zoom_in = True
        else:
            zoom_in = False

        # Get mouse position for zoom center
        mouse_x = event.x
        mouse_y = event.y

        # Calculate new zoom factor
        current_zoom = self.transformer.view_state.zoom_factor
        if zoom_in:
            new_zoom = min(current_zoom * 1.2, self.MAX_ZOOM)
        else:
            new_zoom = max(current_zoom / 1.2, self.MIN_ZOOM)

        # Zoom centered on mouse position
        self._zoom_to_point(new_zoom, mouse_x, mouse_y)

    def _zoom_to_point(self, new_zoom: float, center_x: int, center_y: int):
        """Zoom to specific factor centered on a point."""
        if self.current_image is None:
            return

        # old_zoom = self.transformer.view_state.zoom_factor

        # Convert center point to image coordinates
        img_x, img_y = self.transformer.canvas_to_image(center_x, center_y)
        if img_x is None or img_y is None:
            # If point is outside image, use image center
            img_x = self.transformer.image_width // 2
            img_y = self.transformer.image_height // 2

        # Update zoom
        self.transformer.view_state.zoom_factor = new_zoom

        # Calculate new pan to keep the point centered
        new_canvas_x, new_canvas_y = self.transformer.image_to_canvas(img_x, img_y)

        # Adjust pan to center the point
        pan_adjust_x = new_canvas_x - center_x
        pan_adjust_y = new_canvas_y - center_y

        self.transformer.view_state.pan_x += pan_adjust_x
        self.transformer.view_state.pan_y += pan_adjust_y

        # Constrain pan to valid bounds
        self._constrain_pan()

        self._update_display()

    def _on_drag_start(self, event):
        """Start pan drag operation."""
        if self.current_image is None:
            return

        self.is_dragging = True
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_start_pan_x = self.transformer.view_state.pan_x
        self.drag_start_pan_y = self.transformer.view_state.pan_y

        # Change cursor to indicate dragging
        self.canvas.configure(cursor="hand2")

    def _on_drag_motion(self, event):
        """Handle pan drag motion."""
        if not self.is_dragging or self.current_image is None:
            return

        # Calculate drag delta
        delta_x = event.x - self.drag_start_x
        delta_y = event.y - self.drag_start_y

        # Update pan (note: drag moves in opposite direction to pan)
        self.transformer.view_state.pan_x = self.drag_start_pan_x - delta_x
        self.transformer.view_state.pan_y = self.drag_start_pan_y - delta_y

        # Constrain pan
        self._constrain_pan()

        self._update_display()

    def _on_drag_end(self, event):
        """End pan drag operation."""
        self.is_dragging = False
        self.canvas.configure(cursor="")

    def _constrain_pan(self):
        """Constrain pan to valid bounds."""
        if self.current_image is None:
            return

        view_state = self.transformer.view_state

        # Calculate image size in canvas coordinates
        display_width = self.transformer.image_width * view_state.zoom_factor
        display_height = self.transformer.image_height * view_state.zoom_factor

        # Constrain pan to prevent losing the image
        max_pan_x = max(0, display_width - view_state.canvas_width)
        max_pan_y = max(0, display_height - view_state.canvas_height)

        view_state.pan_x = max(0, min(view_state.pan_x, max_pan_x))
        view_state.pan_y = max(0, min(view_state.pan_y, max_pan_y))

    def _update_display(self):
        """Update the canvas display."""
        if self.current_image is None:
            self.canvas.delete("all")
            return

        # Render image at current view state
        photo_image, img_width, img_height = self.renderer.render_image(
            self.current_image, self.transformer.view_state
        )

        if photo_image:
            # Clear canvas
            self.canvas.delete("all")

            # Calculate position to display image
            canvas_width = self.transformer.view_state.canvas_width
            canvas_height = self.transformer.view_state.canvas_height

            # Center image if it's smaller than canvas
            if img_width < canvas_width:
                x = (canvas_width - img_width) // 2
            else:
                x = 0

            if img_height < canvas_height:
                y = (canvas_height - img_height) // 2
            else:
                y = 0

            # Display image
            self.canvas.create_image(x, y, anchor=tk.NW, image=photo_image)

            # Keep reference to prevent garbage collection
            self.current_photo_image = photo_image

            # Update scroll region
            scroll_width = max(canvas_width, img_width)
            scroll_height = max(canvas_height, img_height)
            self.canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))

        # Callback for UI updates
        if self.update_callback:
            self.update_callback()

    # Public interface methods
    def zoom_in(self):
        """Zoom in to next level."""
        current_zoom = self.transformer.view_state.zoom_factor
        # Find next zoom level
        for zoom in self.ZOOM_LEVELS:
            if zoom > current_zoom:
                self.set_zoom(zoom)
                return
        # If no next level, zoom by factor
        self.set_zoom(min(current_zoom * 2.0, self.MAX_ZOOM))

    def zoom_out(self):
        """Zoom out to previous level."""
        current_zoom = self.transformer.view_state.zoom_factor
        # Find previous zoom level
        for zoom in reversed(self.ZOOM_LEVELS):
            if zoom < current_zoom:
                self.set_zoom(zoom)
                return
        # If no previous level, zoom by factor
        self.set_zoom(max(current_zoom / 2.0, self.MIN_ZOOM))

    def set_zoom(self, zoom_factor: float):
        """Set specific zoom factor."""
        if self.current_image is None:
            return

        zoom_factor = max(self.MIN_ZOOM, min(zoom_factor, self.MAX_ZOOM))

        # Zoom centered on canvas center
        center_x = self.transformer.view_state.canvas_width // 2
        center_y = self.transformer.view_state.canvas_height // 2

        self._zoom_to_point(zoom_factor, center_x, center_y)

    def fit_to_window(self):
        """Fit image to window."""
        if self.current_image is None:
            return

        # Calculate zoom to fit
        canvas_width = self.transformer.view_state.canvas_width
        canvas_height = self.transformer.view_state.canvas_height

        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready, use default
            canvas_width, canvas_height = 400, 300

        zoom_x = canvas_width / self.transformer.image_width
        zoom_y = canvas_height / self.transformer.image_height
        fit_zoom = min(zoom_x, zoom_y, 1.0)  # Don't zoom in beyond 100%

        # Reset pan and set zoom
        self.transformer.view_state.pan_x = 0
        self.transformer.view_state.pan_y = 0
        self.transformer.view_state.zoom_factor = fit_zoom

        self._update_display()

    def zoom_100(self):
        """Zoom to 100% (actual size)."""
        self.set_zoom(1.0)

    def get_zoom_factor(self) -> float:
        """Get current zoom factor."""
        return self.transformer.view_state.zoom_factor

    def get_image_coordinates(
        self, canvas_x: int, canvas_y: int
    ) -> tuple[int | None, int | None]:
        """Get image coordinates from canvas coordinates."""
        return self.transformer.canvas_to_image(canvas_x, canvas_y)


class ZoomToolbar:
    """Toolbar with zoom controls."""

    def __init__(self, parent, zoom_controller: ZoomPanController):
        self.parent = parent
        self.zoom_controller = zoom_controller

        self.toolbar_frame = None
        self.zoom_label = None
        self.zoom_var = None

        self._setup_toolbar()

    def _setup_toolbar(self):
        """Setup the zoom toolbar."""
        self.toolbar_frame = ttk.Frame(self.parent)

        # Zoom out button
        zoom_out_btn = ttk.Button(
            self.toolbar_frame,
            text="🔍-",
            width=4,
            command=self.zoom_controller.zoom_out,
        )
        zoom_out_btn.grid(row=0, column=0, padx=2)

        # Zoom in button
        zoom_in_btn = ttk.Button(
            self.toolbar_frame,
            text="🔍+",
            width=4,
            command=self.zoom_controller.zoom_in,
        )
        zoom_in_btn.grid(row=0, column=1, padx=2)

        # Fit to window button
        fit_btn = ttk.Button(
            self.toolbar_frame,
            text="Fit",
            width=6,
            command=self.zoom_controller.fit_to_window,
        )
        fit_btn.grid(row=0, column=2, padx=2)

        # 100% button
        hundred_btn = ttk.Button(
            self.toolbar_frame,
            text="100%",
            width=6,
            command=self.zoom_controller.zoom_100,
        )
        hundred_btn.grid(row=0, column=3, padx=2)

        # Zoom percentage display
        self.zoom_var = tk.StringVar(value="100%")
        self.zoom_label = ttk.Label(
            self.toolbar_frame, textvariable=self.zoom_var, font=("Arial", 9, "bold")
        )
        self.zoom_label.grid(row=0, column=4, padx=(10, 2))

        # Zoom selector dropdown
        zoom_options = ["25%", "50%", "100%", "200%", "400%", "800%"]
        zoom_combo = ttk.Combobox(
            self.toolbar_frame, values=zoom_options, width=8, state="readonly"
        )
        zoom_combo.grid(row=0, column=5, padx=2)
        zoom_combo.bind("<<ComboboxSelected>>", self._on_zoom_selected)

        # Instructions label
        instructions = ttk.Label(
            self.toolbar_frame,
            text="Mouse: Wheel=Zoom, Right-drag=Pan (or Shift+drag)",
            font=("Arial", 8),
            foreground="gray",
        )
        instructions.grid(row=0, column=6, padx=(20, 0))

    def _on_zoom_selected(self, event):
        """Handle zoom selection from dropdown."""
        combo = event.widget
        selected = combo.get()

        # Parse percentage
        try:
            percentage = int(selected.replace("%", ""))
            zoom_factor = percentage / 100.0
            self.zoom_controller.set_zoom(zoom_factor)
        except ValueError:
            pass

    def update_zoom_display(self):
        """Update the zoom percentage display."""
        if not self.zoom_var:
            return
        zoom_factor = self.zoom_controller.get_zoom_factor()
        percentage = int(zoom_factor * 100)
        self.zoom_var.set(f"{percentage}%")

    def get_frame(self):
        """Get the toolbar frame."""
        return self.toolbar_frame

    def grid(self, **kwargs):
        """Grid the toolbar frame."""
        if self.toolbar_frame:
            self.toolbar_frame.grid(**kwargs)
