"""
DStretch Python GUI - Interfaz avanzada de procesamiento y realce de imágenes arqueológicas
==================================================================

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1

Esta interfaz gráfica replica y amplía la funcionalidad de DStretch para el análisis científico y la documentación arqueológica, integrando procesamiento avanzado, zoom/pan y configuración de pipeline.

Versión: 3.5.1 - Indentation Hotfix
Fecha: Septiembre 2025
"""

import gc
import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# Disable Pillow image limit
Image.MAX_IMAGE_PIXELS = None

# Import DStretch components (with error handling)
try:
    from . import list_available_colorspaces
    from .decorrelation import DecorrelationStretch
except ImportError:
    try:
        from dstretch import list_available_colorspaces
        from dstretch.decorrelation import DecorrelationStretch
    except ImportError:
        # Fallback for standalone execution
        print(
            "Warning: Could not import DStretch components. Some features may be limited."
        )

        def list_available_colorspaces():
            return [
                "YDS",
                "CRGB",
                "LAB",
                "LRE",
                "YBR",
                "YBK",
                "YRD",
                "YWE",
                "YBL",
                "YBG",
                "YUV",
                "YYE",
                "LAX",
                "LDS",
                "LRD",
                "LBK",
                "LBL",
                "LWE",
                "LYE",
                "RGB",
                "RGB0",
                "LABI",
            ]

        DecorrelationStretch = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedSettingsWindow(tk.Toplevel):
    """A Toplevel window for advanced preprocessing settings."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Advanced Preprocessing & Custom Pipeline")
        self.app = app

        self.vars = {
            "invert": tk.BooleanVar(value="invert" in self.app.active_processors),
            "auto_contrast": tk.BooleanVar(
                value="auto_contrast" in self.app.active_processors
            ),
            "color_balance": tk.BooleanVar(
                value="color_balance" in self.app.active_processors
            ),
            "flatten": tk.BooleanVar(value="flatten" in self.app.active_processors),
            "contrast_clip": tk.DoubleVar(
                value=self.app.advanced_settings["contrast_clip"]
            ),
            "balance_method": tk.StringVar(
                value=self.app.advanced_settings["balance_method"]
            ),
            "balance_strength": tk.DoubleVar(
                value=self.app.advanced_settings["balance_strength"]
            ),
            "flatten_method": tk.StringVar(
                value=self.app.advanced_settings["flatten_method"]
            ),
            "flatten_large": tk.DoubleVar(
                value=self.app.advanced_settings["flatten_large"]
            ),
            "flatten_small": tk.DoubleVar(
                value=self.app.advanced_settings["flatten_small"]
            ),
        }

        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window(self)

    def _setup_ui(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        basic_frame = ttk.LabelFrame(main_frame, text="Enable Processors", padding=5)
        basic_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(basic_frame, text="Invert", variable=self.vars["invert"]).pack(
            anchor=tk.W
        )
        ttk.Checkbutton(
            basic_frame, text="Auto Contrast", variable=self.vars["auto_contrast"]
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            basic_frame, text="Color Balance", variable=self.vars["color_balance"]
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            basic_frame, text="Flatten Illumination", variable=self.vars["flatten"]
        ).pack(anchor=tk.W)

        adv_frame = ttk.LabelFrame(main_frame, text="Advanced Options", padding=5)
        adv_frame.pack(fill=tk.X, pady=5)

        ttk.Label(adv_frame, text="Contrast Clip %:").pack(anchor=tk.W)
        ttk.Scale(
            adv_frame,
            from_=0.0,
            to=5.0,
            variable=self.vars["contrast_clip"],
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Label(adv_frame, text="Balance Method:").pack(anchor=tk.W)
        ttk.Combobox(
            adv_frame,
            textvariable=self.vars["balance_method"],
            values=["gray_world"],
            state="readonly",
        ).pack(fill=tk.X, padx=5)
        ttk.Label(adv_frame, text="Balance Strength:").pack(anchor=tk.W)
        ttk.Scale(
            adv_frame,
            from_=0.1,
            to=2.0,
            variable=self.vars["balance_strength"],
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Label(adv_frame, text="Flatten Method:").pack(anchor=tk.W)
        ttk.Combobox(
            adv_frame,
            textvariable=self.vars["flatten_method"],
            values=["gaussian"],
            state="readonly",
        ).pack(fill=tk.X, padx=5)
        ttk.Label(adv_frame, text="Large Structures (Blur Size):").pack(anchor=tk.W)
        ttk.Scale(
            adv_frame,
            from_=10.0,
            to=200.0,
            variable=self.vars["flatten_large"],
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, padx=5)
        ttk.Label(adv_frame, text="Small Structures (Detail):").pack(anchor=tk.W)
        ttk.Scale(
            adv_frame,
            from_=1.0,
            to=10.0,
            variable=self.vars["flatten_small"],
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, padx=5, pady=(0, 5))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(button_frame, text="Apply", command=self._apply_changes).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(
            side=tk.RIGHT
        )

    def _apply_changes(self):
        self.app.active_processors.clear()
        if self.vars["invert"].get():
            self.app.active_processors.add("invert")
        if self.vars["auto_contrast"].get():
            self.app.active_processors.add("auto_contrast")
        if self.vars["color_balance"].get():
            self.app.active_processors.add("color_balance")
        if self.vars["flatten"].get():
            self.app.active_processors.add("flatten")

        for key in self.app.advanced_settings:
            if key in self.vars:
                self.app.advanced_settings[key] = self.vars[key].get()

        self.app._rebuild_from_advanced_settings()
        self.destroy()

    def _cancel(self):
        self.destroy()


class MemoryOptimizedImageCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="gray90", relief="sunken", bd=1, **kwargs)
        self.full_res_image = None
        self.display_image_tk = None
        self.imscale = 1.0
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.bind("<Configure>", self._on_canvas_resize)
        self.bind("<ButtonPress-1>", self._on_button_press)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", self._on_mouse_wheel)
        self.bind("<Button-5>", self._on_mouse_wheel)
        self.bind("<Double-1>", self.fit_to_screen)

    def set_image(self, image_np: np.ndarray):
        self.full_res_image = (
            np.clip(image_np, 0, 255).astype(np.uint8)
            if image_np.dtype != np.uint8
            else image_np
        )
        self.fit_to_screen()

    def fit_to_screen(self, event=None):
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._redraw_image()

    def _on_canvas_resize(self, event):
        self.fit_to_screen()

    def _on_button_press(self, event):
        self.delta_x = event.x
        self.delta_y = event.y

    def _on_mouse_drag(self, event):
        self.offset_x += event.x - self.delta_x
        self.offset_y += event.y - self.delta_y
        self.delta_x = event.x
        self.delta_y = event.y
        self._redraw_image()

    def _on_mouse_wheel(self, event):
        factor = 1.1 if (event.num == 4 or event.delta > 0) else 0.9
        self.zoom_level = max(0.1, min(self.zoom_level * factor, 20.0))
        self.offset_x = event.x - (event.x - self.offset_x) * factor
        self.offset_y = event.y - (event.y - self.offset_y) * factor
        self._redraw_image()

    def _redraw_image(self):
        if self.full_res_image is None:
            return
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw <= 1 or ch <= 1:
            self.after(50, self._redraw_image)
            return
        ih, iw = self.full_res_image.shape[:2]
        self.imscale = min(cw / iw, ch / ih)
        total_scale = self.imscale * self.zoom_level
        vx1, vy1 = int(-self.offset_x / total_scale), int(-self.offset_y / total_scale)
        vx2, vy2 = (
            int((cw - self.offset_x) / total_scale),
            int((ch - self.offset_y) / total_scale),
        )
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(iw, vx2), min(ih, vy2)
        if vx2 <= vx1 or vy2 <= vy1:
            self.delete("all")
            return
        cropped = self.full_res_image[vy1:vy2, vx1:vx2]
        dw, dh = int((vx2 - vx1) * total_scale), int((vy2 - vy1) * total_scale)
        if dw <= 0 or dh <= 0:
            self.delete("all")
            return
        resized = cv2.resize(cropped, (dw, dh), interpolation=cv2.INTER_AREA)
        self.display_image_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        self.delete("all")
        self.create_image(
            self.offset_x + vx1 * total_scale,
            self.offset_y + vy1 * total_scale,
            image=self.display_image_tk,
            anchor=tk.NW,
        )


class DStretchGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DStretch Python v3.5 - Advanced UI")
        screen_width = self.root.winfo_screenwidth()
        if screen_width < 1366:
            w, h = 1000, 700
        elif screen_width < 1920:
            w, h = 1200, 800
        else:
            w, h = 1400, 900
        self.root.geometry(f"{w}x{h}")
        self.dstretch = DecorrelationStretch() if DecorrelationStretch else None
        self.original_image, self.preprocessed_image, self.processed_image = (
            None,
            None,
            None,
        )
        self.original_dimensions = None
        self.current_colorspace = "YDS"
        self.active_processors = set()
        self.advanced_settings = self._get_default_advanced_settings()

        self.processors = {
            "invert": lambda img: 255 - img,
            "auto_contrast": lambda img: self._apply_auto_contrast(
                img, clip_percent=self.advanced_settings["contrast_clip"]
            ),
            "color_balance": lambda img: self._apply_color_balance(
                img, strength=self.advanced_settings["balance_strength"]
            ),
            "flatten": lambda img: self._apply_flatten(
                img, ksize_factor=self.advanced_settings["flatten_large"]
            ),
            "hue_shift": self._apply_hue_shift,
        }

        self.status_var = tk.StringVar(value="Ready")
        self._setup_styles()
        self._setup_ui()
        logger.info("DStretch GUI initialized successfully")

    def _get_default_advanced_settings(self):
        return {
            "contrast_clip": 0.1,
            "balance_method": "gray_world",
            "balance_strength": 1.0,
            "flatten_method": "gaussian",
            "flatten_large": 40.0,
            "flatten_small": 3.0,
        }

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.configure(
                "Selected.TButton",
                background="#0078d4",
                foreground="white",
                font=("TkDefaultFont", 9, "bold"),
            )
            style.configure("Active.TButton", background="#28a745", foreground="white")
        except tk.TclError:
            pass

    def _setup_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_frame = ttk.Frame(main_paned)
        main_paned.add(self.image_frame, weight=70)
        self.canvas = MemoryOptimizedImageCanvas(self.image_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.controls_frame = ttk.Frame(main_paned)
        main_paned.add(self.controls_frame, weight=30)
        self._setup_standard_mode()
        self.status_label = ttk.Label(
            self.root, textvariable=self.status_var, relief="sunken", anchor="w"
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_standard_mode(self):
        for widget in self.controls_frame.winfo_children():
            widget.destroy()
        file_frame = ttk.Frame(self.controls_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(file_frame, text="Open image", command=self._open_image).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2)
        )
        ttk.Button(
            file_frame, text="Restore\noriginal", command=self._restore_original
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
        ttk.Button(file_frame, text="Save result", command=self._save_result).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0)
        )

        preprocess_frame = ttk.LabelFrame(
            self.controls_frame, text="Preprocessing", padding=5
        )
        preprocess_frame.pack(fill=tk.X, pady=(0, 10))
        preprocess_row1 = ttk.Frame(preprocess_frame)
        preprocess_row1.pack(fill=tk.X, pady=2)
        self.preprocess_buttons = {}
        buttons1 = {
            "invert": "Invert",
            "auto_contrast": "Auto\ncontrast",
            "color_balance": "Color\nbalance",
            "flatten": "Flatten",
        }
        for name, text in buttons1.items():
            self.preprocess_buttons[name] = ttk.Button(
                preprocess_row1,
                text=text,
                command=lambda n=name: self._toggle_preprocessing(n),
            )
            self.preprocess_buttons[name].pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=2
            )

        preprocess_row2 = ttk.Frame(preprocess_frame)
        preprocess_row2.pack(fill=tk.X, pady=2)
        btn_hue = ttk.Button(
            preprocess_row2,
            text="Hue Shift",
            command=lambda: self._toggle_preprocessing("hue_shift"),
        )
        btn_hue.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        btn_qen = ttk.Button(
            preprocess_row2, text="Quick\nEnhance", command=self._apply_quick_enhance
        )
        btn_qen.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        btn_adv = ttk.Button(
            preprocess_row2, text="ADVANCED", command=self._open_advanced_window
        )
        btn_adv.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.preprocess_buttons["hue_shift"] = btn_hue

        colorspace_frame = ttk.LabelFrame(
            self.controls_frame, text="Color Spaces", padding=5
        )
        colorspace_frame.pack(fill=tk.X, pady=(0, 10))
        self._setup_colorspace_grid(colorspace_frame)

        scale_frame = ttk.LabelFrame(self.controls_frame, text="Scale:", padding=5)
        scale_frame.pack(fill=tk.X, pady=(0, 10))
        scale_control_frame = ttk.Frame(scale_frame)
        scale_control_frame.pack(fill=tk.X)
        self.scale_var = tk.DoubleVar(value=15.0)
        self.scale_slider = ttk.Scale(
            scale_control_frame,
            from_=1.0,
            to=100.0,
            variable=self.scale_var,
            orient=tk.HORIZONTAL,
            command=self._on_scale_changed,
        )
        self.scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.scale_label = ttk.Label(scale_control_frame, text="15", width=4)
        self.scale_label.pack(side=tk.RIGHT, padx=(5, 0))

        footer_frame = ttk.Frame(self.controls_frame)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(footer_frame, text="About", command=self._show_about).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2)
        )
        ttk.Button(
            footer_frame, text="Restore\nSettings", command=self._restore_settings
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
        ttk.Button(footer_frame, text="EXIT", command=self.root.quit).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0)
        )

    def _setup_colorspace_grid(self, parent):
        colorspace_layout = [
            ["YDS", "YBR", "YBK", "YRE"],
            ["YRD", "YWE", "YBL", "YBG"],
            ["YUV", "YYE", "LAX", "LDS"],
            ["LRE", "LRD", "LBK", "LBL"],
            ["LWE", "LYE", "RGB", "LAB"],
            ["CRGB", "RGB0", "LABI", "CUSTOM"],
        ]
        available = (
            list_available_colorspaces() if callable(list_available_colorspaces) else []
        )
        self.colorspace_buttons = {}
        for r, row in enumerate(colorspace_layout):
            for c, name in enumerate(row):
                if name in available or name == "CUSTOM":
                    cmd = (
                        self._open_advanced_window
                        if name == "CUSTOM"
                        else lambda n=name: self._select_colorspace(n)
                    )
                    btn = ttk.Button(parent, text=name, width=8, command=cmd)
                    btn.grid(row=r, column=c, padx=1, pady=1, sticky="ew")
                    self.colorspace_buttons[name] = btn
        for i in range(4):
            parent.columnconfigure(i, weight=1)

    def _open_advanced_window(self):
        if self.original_image is None:
            self._set_status("Load an image to access advanced settings")
            return
        AdvancedSettingsWindow(self.root, self)

    def _apply_auto_contrast(
        self, image: np.ndarray, clip_percent: float
    ) -> np.ndarray:
        result = image.copy()
        for i in range(image.shape[2]):
            channel = result[:, :, i]
            low, high = np.percentile(channel, [clip_percent, 100 - clip_percent])
            result[:, :, i] = np.clip(
                (channel - low) / (high - low + 1e-6) * 255, 0, 255
            ).astype(np.uint8)
        return result

    def _apply_color_balance(self, image: np.ndarray, strength: float) -> np.ndarray:
        if len(image.shape) < 3 or image.shape[2] != 3:
            return image
        avg_r, avg_g, avg_b = (
            np.mean(image[:, :, 0]),
            np.mean(image[:, :, 1]),
            np.mean(image[:, :, 2]),
        )
        avg_gray = (avg_r + avg_g + avg_b) / 3.0
        scale_r, scale_g, scale_b = (
            avg_gray / (avg_r + 1e-6),
            avg_gray / (avg_g + 1e-6),
            avg_gray / (avg_b + 1e-6),
        )
        scale_r, scale_g, scale_b = (
            1 + (scale_r - 1) * strength,
            1 + (scale_g - 1) * strength,
            1 + (scale_b - 1) * strength,
        )
        balanced_image = image.astype(np.float32)
        balanced_image[:, :, 0] *= scale_r
        balanced_image[:, :, 1] *= scale_g
        balanced_image[:, :, 2] *= scale_b
        return np.clip(balanced_image, 0, 255).astype(np.uint8)

    def _apply_flatten(self, image: np.ndarray, ksize_factor: float) -> np.ndarray:
        ksize = int(ksize_factor) | 1
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return np.clip(
            cv2.divide(
                image.astype(np.float32), blurred.astype(np.float32) + 1, scale=255
            ),
            0,
            255,
        ).astype(np.uint8)

    def _apply_hue_shift(self, image: np.ndarray, shift: int = 30) -> np.ndarray:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 0] = ((hsv_image[:, :, 0].astype(int) + shift) % 180).astype(
            np.uint8
        )
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    def _apply_quick_enhance(self):
        if self.original_image is None:
            self._set_status("No image loaded")
            return
        self._set_status("Applying Quick Enhance...")

        def process():
            try:
                source = (
                    self.processed_image
                    if self.processed_image is not None
                    else self.original_image
                )
                if source is None:
                    return
                lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_lab = cv2.merge((clahe.apply(l_channel), a, b))
                contrast_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                hsv = cv2.cvtColor(contrast_enhanced, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                s = np.clip(s.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
                final = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
                self.root.after(0, self._update_quick_enhance_result, final)
            except Exception as e:
                self.root.after(0, self._set_status, f"Quick Enhance error: {str(e)}")

        threading.Thread(target=process, daemon=True).start()

    def _update_quick_enhance_result(self, enhanced_image):
        self.preprocessed_image = enhanced_image
        self.processed_image = enhanced_image
        self.canvas.set_image(self.processed_image)
        self._set_status(
            "Applied Quick Enhance. Active preprocessing has been baked in."
        )
        self.active_processors.clear()
        for name, button in self.preprocess_buttons.items():
            if name in self.processors:
                button.configure(style="TButton")

    def _rebuild_from_advanced_settings(self):
        self._set_status("Applying advanced settings...")
        for name, btn in self.preprocess_buttons.items():
            if name in self.processors:
                btn.configure(
                    style="Active.TButton"
                    if name in self.active_processors
                    else "TButton"
                )
        self._rebuild_current_image()

    def _restore_settings(self):
        self.scale_var.set(15.0)
        self.scale_label.config(text="15")
        self.advanced_settings = self._get_default_advanced_settings()
        self._reset_button_states()
        if self.original_image is not None:
            self.preprocessed_image = self.original_image.copy()
            self.processed_image = self.original_image.copy()
            self.canvas.set_image(self.original_image)
        self._set_status("Settings restored to defaults")
        gc.collect()

    def _show_about(self):
        messagebox.showinfo(
            "About DStretch Python",
            "DStretch Python v3.5 - Advanced Settings UI\n\n"
            "Features:\n"
            "• Advanced settings panel for fine-tuned control\n"
            "• Image Zoom (mousewheel) and Pan (click-drag)\n"
            "• Functional preprocessing tools\n"
            "• 23+ specialized color spaces",
        )

    def _update_rebuilt_result(self, rebuilt_image):
        self.preprocessed_image = rebuilt_image
        self.processed_image = rebuilt_image
        self.canvas.set_image(self.processed_image)
        active = sorted(list(self.active_processors))
        self._set_status(
            f"Rebuilt with: {' + '.join(active)}" if active else "Restored to original"
        )

    def _apply_current_colorspace_safe(self):
        if self.original_image is None or not self.dstretch:
            self._set_status("No image loaded")
            return
        source = (
            self.preprocessed_image
            if self.preprocessed_image is not None
            else self.original_image
        )
        try:
            optimized = self._optimize_image_for_processing(source)
            if optimized is None:
                return
            self._set_status(f"Processing with {self.current_colorspace}...")

            def process():
                try:
                    if self.dstretch:
                        result = self.dstretch.process(
                            optimized, self.current_colorspace, self.scale_var.get()
                        )
                        self.root.after(
                            0, self._update_colorspace_result, result.processed_image
                        )
                except Exception as e:
                    self.root.after(0, self._set_status, f"Error processing: {e}")
                finally:
                    gc.collect()

            threading.Thread(target=process, daemon=True).start()
        except Exception as e:
            self._set_status(f"Failed to start processing: {e}")

    def _update_colorspace_result(self, final_image):
        self.processed_image = final_image
        self.canvas.set_image(self.processed_image)
        active = " + ".join(sorted(list(self.active_processors)))
        status = (
            f"Applied: {active} → {self.current_colorspace}"
            if active
            else f"Applied: {self.current_colorspace}"
        )
        self._set_status(f"{status} (scale {int(self.scale_var.get())})")

    def _open_image(self):
        filename = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            try:
                img = Image.open(filename)
                # Removed thumbnail resizing
                # if max(img.size) > 3000: img.thumbnail((3000, 3000), Image.Resampling.LANCZOS)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                self.original_image = np.array(img, dtype=np.uint8)
                self.original_dimensions = (
                    self.original_image.shape[0],
                    self.original_image.shape[1],
                )
                self.preprocessed_image = self.original_image.copy()
                self.processed_image = self.original_image.copy()
                self.canvas.set_image(self.original_image)
                self._restore_settings()
                self._set_status(
                    f"Loaded: {Path(filename).name} ({self.original_image.shape[1]}x{self.original_image.shape[0]})"
                )
                del img
                gc.collect()
            except Exception as e:
                messagebox.showerror("Open Error", f"Could not open image:\n{e}")

    def _restore_original(self):
        if self.original_image is not None:
            self.preprocessed_image = self.original_image.copy()
            self.processed_image = self.original_image.copy()
            self.canvas.set_image(self.original_image)
            self._reset_button_states()
            self._set_status("Restored to original image")
            gc.collect()

    def _on_closing(self):
        try:
            del self.original_image
            del self.preprocessed_image
            del self.processed_image
            gc.collect()
        except Exception:
            pass
        finally:
            self.root.destroy()

    def _optimize_image_for_processing(self, image):
        if image is None:
            return None
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        # Removed resizing logic to process at full resolution
        # h, w = image.shape[:2]; max_h, max_w = self.max_image_size
        # if h > max_h or w > max_w:
        #     scale = min(max_h / h, max_w / w); nw, nh = int(w * scale), int(h * scale)
        #     self._set_status(f"Resizing for processing: {w}x{h} -> {nw}x{nh}")
        #     return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        return image

    def _select_colorspace(self, name):
        if name == "CUSTOM":
            self._open_advanced_window()
            return
        self.current_colorspace = name
        for n, btn in self.colorspace_buttons.items():
            if n != "CUSTOM":
                btn.configure(style="Selected.TButton" if n == name else "TButton")
        self._apply_current_colorspace_safe()

    # --- CORRECTED TOGGLE FUNCTION ---
    def _toggle_preprocessing(self, name):
        if self.original_image is None:
            self._set_status("No image loaded")
            return
        if name not in self.processors:
            self._set_status(f"Processor '{name}' not implemented")
            return

        btn = self.preprocess_buttons.get(name)
        if not btn:
            return

        if name in self.active_processors:
            self.active_processors.remove(name)
            btn.configure(style="TButton")
            self._set_status(f"Disabled {name}")
            self._rebuild_current_image()
        else:
            self.active_processors.add(name)
            try:
                btn.configure(style="Active.TButton")
            except tk.TclError:
                pass
            self._set_status(f"Enabled {name}")
            self._apply_single_processor(name)

    def _apply_single_processor(self, name):
        def process():
            try:
                source = (
                    self.processed_image
                    if self.processed_image is not None
                    else self.original_image
                )
                enhanced = self.processors[name](source)
                self.root.after(0, self._update_direct_result, enhanced)
            except Exception as e:
                self.root.after(0, self._set_status, f"Error applying {name}: {str(e)}")

        threading.Thread(target=process, daemon=True).start()

    def _rebuild_current_image(self):
        def rebuild():
            try:
                if self.original_image is None:
                    return
                img = self.original_image.copy()
                order = [
                    "invert",
                    "auto_contrast",
                    "color_balance",
                    "flatten",
                    "hue_shift",
                ]
                for name in order:
                    if name in self.active_processors:
                        img = self.processors[name](img)
                self.root.after(0, self._update_rebuilt_result, img)
            except Exception as e:
                self.root.after(0, self._set_status, f"Error rebuilding: {str(e)}")

        threading.Thread(target=rebuild, daemon=True).start()

    def _update_direct_result(self, image):
        self.preprocessed_image = image
        self.processed_image = image
        self.canvas.set_image(self.processed_image)
        active = sorted(list(self.active_processors))
        self._set_status(
            f"Active: {' + '.join(active)}" if active else "No preprocessing active"
        )

    def _on_scale_changed(self, value):
        self.scale_label.config(text=str(int(float(value))))
        if hasattr(self, "_scale_timer"):
            self.root.after_cancel(self._scale_timer)
        self._scale_timer = self.root.after(500, self._apply_current_colorspace_safe)

    def _save_result(self):
        if self.processed_image is None:
            messagebox.showwarning("Save Warning", "No processed image to save")
            return
        default = (
            f"enhanced_{self.current_colorspace}_scale{int(self.scale_var.get())}.jpg"
        )
        filename = filedialog.asksaveasfilename(
            title="Save Enhanced Image",
            initialfile=default,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tiff")],
            defaultextension=".jpg",
        )
        if filename:
            try:
                image_to_save = self.processed_image.copy()

                # Resize back to original dimensions if needed
                if self.original_dimensions is not None:
                    current_h, current_w = image_to_save.shape[:2]
                    orig_h, orig_w = self.original_dimensions
                    if (current_h, current_w) != (orig_h, orig_w):
                        image_to_save = cv2.resize(
                            image_to_save,
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_LANCZOS4,
                        )

                # Configure save options
                save_kwargs = {}
                if filename.lower().endswith((".jpg", ".jpeg")):
                    save_kwargs["quality"] = 95
                    save_kwargs["subsampling"] = 0

                Image.fromarray(image_to_save).save(filename, **save_kwargs)
                self._set_status(f"Saved: {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save image:\n{e}")

    def _reset_button_states(self):
        self.active_processors.clear()
        if hasattr(self, "preprocess_buttons"):
            for btn in self.preprocess_buttons.values():
                btn.configure(style="TButton")
        if hasattr(self, "colorspace_buttons"):
            for n, btn in self.colorspace_buttons.items():
                if n != "CUSTOM":
                    btn.configure(style="TButton")
        self.current_colorspace = "YDS"

    def _set_status(self, message):
        if hasattr(self, "status_var"):
            self.status_var.set(message)
            logger.info(f"Status: {message}")

    def run(self):
        try:
            self.root.minsize(800, 600)
            self.root.resizable(True, True)
            self.root.update_idletasks()
            x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
            y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
            self.root.geometry(f"+{x}+{y}")
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror("Application Error", f"An error occurred:\n{e}")


def main():
    try:
        app = DStretchGUI()
        app.run()
    except Exception as e:
        try:
            import tkinter.messagebox as mb

            mb.showerror("Startup Error", f"Failed to start DStretch GUI:\n{e}")
        except Exception as e:
            print(f"Failed to start DStretch GUI: {e}")


if __name__ == "__main__":
    main()
