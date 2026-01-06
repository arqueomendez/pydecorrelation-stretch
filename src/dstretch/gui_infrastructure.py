"""
Essential GUI Infrastructure for DStretch Python.

Provides error management, enhanced status bar, tooltips, and performance optimizations.
"""

import logging
import threading
import time
import tkinter as tk
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any


class ErrorManager:
    """Handles application errors with user-friendly messages and logging."""

    def __init__(self, app_name: str = "DStretch Python"):
        self.app_name = app_name
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        log_dir = Path.home() / ".dstretch_logs"
        log_dir.mkdir(exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger("dstretch")
        self.logger.setLevel(logging.INFO)

        # File handler
        log_file = log_dir / f"dstretch_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def handle_error(
        self,
        error: Exception,
        context: str = "",
        user_message: str | None = None,
        show_dialog: bool = True,
    ) -> bool:
        """Handle an error with logging and optional user notification."""

        # Log the error
        error_msg = f"{context}: {str(error)}"
        self.logger.error(error_msg)
        self.logger.error(traceback.format_exc())

        # Show user dialog if requested
        if show_dialog:
            if user_message is None:
                user_message = self._generate_user_message(error, context)

            messagebox.showerror(f"{self.app_name} - Error", user_message)

        return True

    def _generate_user_message(self, error: Exception, context: str) -> str:
        """Generate a user-friendly error message."""
        error_type = type(error).__name__

        # Common error messages
        if "FileNotFoundError" in error_type:
            return "The requested file could not be found. Please check the file path and try again."
        elif "MemoryError" in error_type:
            return "Not enough memory to complete this operation. Try with a smaller image or close other applications."
        elif "PermissionError" in error_type:
            return "Permission denied. Please check file permissions or try running as administrator."
        elif "PIL" in str(error) or "Image" in str(error):
            return "Error processing the image. The file may be corrupted or in an unsupported format."
        elif "numpy" in str(error) or "array" in str(error):
            return "Error in image data processing. Please try with a different image."
        else:
            return f"An unexpected error occurred{' in ' + context if context else ''}.\n\nError: {str(error)}\n\nPlease check the log files for more details."

    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def safe_execute(
        self,
        func: Callable,
        *args,
        context: str = "",
        user_message: str | None = None,
        **kwargs,
    ) -> Any:
        """Safely execute a function with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, context, user_message)
            return None


class AdvancedStatusBar:
    """Enhanced status bar with multiple information panels."""

    def __init__(self, parent):
        self.parent = parent
        self.status_frame = ttk.Frame(parent)

        # Configure grid weights
        self.status_frame.columnconfigure(0, weight=1)

        # Main status label (expandable)
        self.main_status = tk.StringVar(value="Ready")
        self.main_label = ttk.Label(
            self.status_frame,
            textvariable=self.main_status,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2),
        )
        self.main_label.grid(row=0, column=0, sticky="we", padx=(0, 2))

        # Processing info panel
        self.process_info = tk.StringVar(value="")
        self.process_label = ttk.Label(
            self.status_frame,
            textvariable=self.process_info,
            relief=tk.SUNKEN,
            width=15,
            padding=(5, 2),
        )
        self.process_label.grid(row=0, column=1, padx=2)

        # Zoom info panel
        self.zoom_info = tk.StringVar(value="100%")
        self.zoom_label = ttk.Label(
            self.status_frame,
            textvariable=self.zoom_info,
            relief=tk.SUNKEN,
            width=8,
            padding=(5, 2),
        )
        self.zoom_label.grid(row=0, column=2, padx=2)

        # Image info panel
        self.image_info = tk.StringVar(value="No image")
        self.image_label = ttk.Label(
            self.status_frame,
            textvariable=self.image_info,
            relief=tk.SUNKEN,
            width=20,
            padding=(5, 2),
        )
        self.image_label.grid(row=0, column=3, padx=(2, 0))

        # Progress bar (hidden by default)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.status_frame,
            variable=self.progress_var,
            mode="determinate",
            length=100,
        )
        # Progress bar will be shown/hidden as needed

    def set_main_status(self, text: str):
        """Set main status message."""
        self.main_status.set(text)

    def set_image_info(
        self,
        width: int | None = None,
        height: int | None = None,
        filename: str | None = None,
    ):
        """Set image information."""
        if width and height:
            info = f"{width}×{height}"
            if filename:
                info += f" | {Path(filename).name}"
            self.image_info.set(info)
        else:
            self.image_info.set("No image")

    def set_zoom_info(self, zoom_factor: float):
        """Set zoom information."""
        self.zoom_info.set(f"{int(zoom_factor * 100)}%")

    def set_processing_info(
        self, colorspace: str | None = None, scale: float | None = None
    ):
        """Set processing information."""
        if colorspace and scale:
            self.process_info.set(f"{colorspace} | {int(scale)}")
        else:
            self.process_info.set("")

    def show_progress(self, value: float | None = None):
        """Show progress bar with optional value (0-100)."""
        if value is not None:
            self.progress_var.set(value)

        # Show progress bar
        self.progress_bar.grid(row=0, column=4, padx=(5, 0), sticky=tk.E)
        self.status_frame.update_idletasks()

    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.grid_remove()
        self.status_frame.update_idletasks()

    def update_progress(self, value: float):
        """Update progress bar value (0-100)."""
        self.progress_var.set(value)
        self.status_frame.update_idletasks()

    def get_frame(self):
        """Get the status bar frame."""
        return self.status_frame

    def grid(self, **kwargs):
        """Grid the status bar frame."""
        self.status_frame.grid(**kwargs)


class TooltipManager:
    """Simple tooltip manager for better UX."""

    def __init__(self):
        self.tooltips = {}
        self.active_tooltip = None

    def add_tooltip(self, widget, text: str, delay: int = 1000):
        """Add a tooltip to a widget."""
        tooltip = SimpleTooltip(widget, text, delay)
        self.tooltips[widget] = tooltip
        return tooltip

    def remove_tooltip(self, widget):
        """Remove tooltip from a widget."""
        if widget in self.tooltips:
            self.tooltips[widget].destroy()
            del self.tooltips[widget]

    def add_colorspace_tooltip(self, button, colorspace_name: str, description: str):
        """Add specialized tooltip for colorspace buttons."""
        tooltip_text = (
            f"{colorspace_name}\n{description}\n\nClick to select this colorspace"
        )
        self.add_tooltip(button, tooltip_text)


class SimpleTooltip:
    """Lightweight tooltip implementation."""

    def __init__(self, widget, text: str, delay: int = 1000):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.timer_id = None

        # Bind events
        self.widget.bind("<Enter>", self._on_enter)
        self.widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, event):
        """Handle mouse enter."""
        self.timer_id = self.widget.after(self.delay, self._show_tooltip)

    def _on_leave(self, event):
        """Handle mouse leave."""
        if self.timer_id:
            self.widget.after_cancel(self.timer_id)
            self.timer_id = None
        self._hide_tooltip()

    def _show_tooltip(self):
        """Show the tooltip."""
        if self.tooltip_window:
            return

        # Calculate position
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_attributes("-topmost", True)

        # Create content
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            background="#FFFFCC",
            foreground="#000000",
            font=("Arial", 9),
            relief=tk.SOLID,
            borderwidth=1,
            padx=6,
            pady=4,
            justify=tk.LEFT,
        )
        label.pack()

        # Position tooltip
        self.tooltip_window.geometry(f"+{x}+{y}")

    def _hide_tooltip(self):
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def destroy(self):
        """Clean up tooltip."""
        if self.timer_id:
            self.widget.after_cancel(self.timer_id)
        self._hide_tooltip()


class PerformanceManager:
    """Manages performance optimizations and monitoring."""

    def __init__(self):
        self.processing_active = False
        self.start_time = None

    def start_operation(self, operation_name: str = "Processing"):
        """Start timing an operation."""
        self.processing_active = True
        self.start_time = time.time()
        self.operation_name = operation_name

    def end_operation(self) -> float:
        """End timing and return duration."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.processing_active = False
            self.start_time = None
            return duration
        return 0.0

    def execute_with_progress(
        self,
        func: Callable,
        status_bar: AdvancedStatusBar,
        total_steps: int = 100,
        step_callback: Callable | None = None,
    ):
        """Execute a function with progress indication."""

        def progress_wrapper():
            try:
                status_bar.show_progress(0)

                result = None
                # Simulate progress if no step callback provided
                if step_callback is None:
                    for i in range(total_steps + 1):
                        if i == 0:
                            result = func()

                        progress = (i / total_steps) * 100
                        status_bar.update_progress(progress)
                        time.sleep(0.01)  # Small delay for visual feedback
                else:
                    result = func()

                status_bar.hide_progress()
                return result

            except Exception as e:
                status_bar.hide_progress()
                raise e

        return progress_wrapper()


class ThreadManager:
    """Manages background threading for UI responsiveness."""

    def __init__(self):
        self.active_threads = []

    def execute_async(
        self,
        func: Callable,
        callback: Callable | None = None,
        error_callback: Callable | None = None,
        *args,
        **kwargs,
    ):
        """Execute a function asynchronously."""

        def thread_wrapper():
            try:
                result = func(*args, **kwargs)
                if callback:
                    cb = callback
                    # Schedule callback on main thread
                    self_obj = getattr(cb, "__self__", None)
                    if self_obj and hasattr(self_obj, "after"):
                        self_obj.after(0, lambda: cb(result))
                    else:
                        cb(result)
            except Exception as e:
                if error_callback:
                    err_cb = error_callback
                    self_obj = getattr(err_cb, "__self__", None)
                    if self_obj and hasattr(self_obj, "after"):
                        self_obj.after(0, lambda: err_cb(e))
                    else:
                        err_cb(e)

        thread = threading.Thread(target=thread_wrapper, daemon=True)
        thread.start()
        self.active_threads.append(thread)

        # Clean up finished threads
        self.active_threads = [t for t in self.active_threads if t.is_alive()]

        return thread

    def wait_for_all(self, timeout: float | None = None):
        """Wait for all active threads to complete."""
        for thread in self.active_threads:
            thread.join(timeout)


@dataclass
class AppInfo:
    """Application information container."""

    name: str = "DStretch Python"
    version: str = "1.0.0"
    author: str = "Archaeological Image Analysis Team"
    description: str = "Advanced image enhancement for archaeological documentation"
    website: str = "https://github.com/archaeological-image-analysis/dstretch-python"


class GUIInfrastructure:
    """Main infrastructure manager that coordinates all components."""

    def __init__(self, root_widget: tk.Widget):
        self.root = root_widget
        self.app_info = AppInfo()

        # Initialize components
        self.error_manager = ErrorManager(self.app_info.name)
        self.tooltip_manager = TooltipManager()
        self.performance_manager = PerformanceManager()
        self.thread_manager = ThreadManager()

        # Status bar will be created by the main GUI
        self.status_bar = None

    def set_status_bar(self, status_bar: AdvancedStatusBar):
        """Set the status bar reference."""
        self.status_bar = status_bar

    def safe_execute(self, func: Callable, context: str = "", *args, **kwargs):
        """Safely execute a function with full error handling."""
        return self.error_manager.safe_execute(func, *args, context=context, **kwargs)

    def execute_with_progress(self, func: Callable, progress_steps: int = 100):
        """Execute function with progress indication."""
        if self.status_bar:
            return self.performance_manager.execute_with_progress(
                func, self.status_bar, progress_steps
            )
        else:
            return func()

    def execute_async(
        self,
        func: Callable,
        callback: Callable | None = None,
        error_callback: Callable | None = None,
        *args,
        **kwargs,
    ):
        """Execute function asynchronously."""
        return self.thread_manager.execute_async(
            func, callback, error_callback, *args, **kwargs
        )

    def log_info(self, message: str):
        """Log an informational message."""
        self.error_manager.log_info(message)

    def log_warning(self, message: str):
        """Log a warning message."""
        self.error_manager.log_warning(message)

    def cleanup(self):
        """Cleanup infrastructure components."""
        self.thread_manager.wait_for_all(timeout=2.0)
