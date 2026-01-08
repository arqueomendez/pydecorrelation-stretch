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

import customtkinter as ctk

# Configure CustomTkinter for premium look
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue") 

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
        from pydecorrelation_stretch import list_available_colorspaces
        from pydecorrelation_stretch.decorrelation import DecorrelationStretch
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


class AdvancedSettingsWindow(ctk.CTkToplevel):
    """A Toplevel window for advanced preprocessing settings."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Advanced Preprocessing & Custom Pipeline")
        self.geometry("500x750")
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
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Basic Processors
        ctk.CTkLabel(main_frame, text="Enable Processors", font=("Roboto", 16, "bold")).pack(anchor=tk.W, pady=(10, 5))
        basic_frame = ctk.CTkFrame(main_frame)
        basic_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkCheckBox(basic_frame, text="Invert", variable=self.vars["invert"]).pack(anchor=tk.W, padx=10, pady=5)
        ctk.CTkCheckBox(basic_frame, text="Auto Contrast", variable=self.vars["auto_contrast"]).pack(anchor=tk.W, padx=10, pady=5)
        ctk.CTkCheckBox(basic_frame, text="Color Balance", variable=self.vars["color_balance"]).pack(anchor=tk.W, padx=10, pady=5)
        ctk.CTkCheckBox(basic_frame, text="Flatten Illumination", variable=self.vars["flatten"]).pack(anchor=tk.W, padx=10, pady=5)

        # Advanced Options
        ctk.CTkLabel(main_frame, text="Advanced Options", font=("Roboto", 16, "bold")).pack(anchor=tk.W, pady=(20, 5))
        adv_frame = ctk.CTkFrame(main_frame)
        adv_frame.pack(fill=tk.X, pady=5)

        ctk.CTkLabel(adv_frame, text="Contrast Clip %:").pack(anchor=tk.W, padx=10, pady=(10, 0))
        ctk.CTkSlider(adv_frame, from_=0.0, to=5.0, variable=self.vars["contrast_clip"]).pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(adv_frame, text="Balance Method:").pack(anchor=tk.W, padx=10)
        ctk.CTkOptionMenu(adv_frame, variable=self.vars["balance_method"], values=["gray_world", "white_patch", "manual"]).pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(adv_frame, text="Balance Strength:").pack(anchor=tk.W, padx=10)
        ctk.CTkSlider(adv_frame, from_=0.1, to=2.0, variable=self.vars["balance_strength"]).pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(adv_frame, text="Flatten Method:").pack(anchor=tk.W, padx=10)
        ctk.CTkOptionMenu(adv_frame, variable=self.vars["flatten_method"], values=["gaussian", "rolling_ball"]).pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(adv_frame, text="Large Structures (Blur):").pack(anchor=tk.W, padx=10)
        ctk.CTkSlider(adv_frame, from_=10.0, to=200.0, variable=self.vars["flatten_large"]).pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(adv_frame, text="Small Structures (Detail):").pack(anchor=tk.W, padx=10)
        ctk.CTkSlider(adv_frame, from_=1.0, to=10.0, variable=self.vars["flatten_small"]).pack(fill=tk.X, padx=10, pady=(5, 15))

        # Buttons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ctk.CTkButton(button_frame, text="Apply Changes", command=self._apply_changes).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(button_frame, text="Cancel", fg_color="transparent", border_width=2, command=self._cancel).pack(side=tk.RIGHT, padx=5)

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
        # Dark background for premium look, remove borders
        if "bg" not in kwargs:
            kwargs["bg"] = "#2b2b2b"
            
        super().__init__(parent, relief="flat", bd=0, highlightthickness=0, **kwargs)
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
        # Detect Memmap or Array
        # If memmap, we might want to check size before converting to uint8 full array?
        # But for display, we MUST load it to RAM (downsampled).
        # We can implement a downsample_for_display in generic utils?
        # For now, assuming image_np fits in RAM or this canvas will crash.
        # But wait, we implemented streaming because image doesn't fit in RAM.
        # So we CANNOT just do image_np.astype(np.uint8) if valid image is huge.
        
        # Optimization: if huge, subsample immediately
        if isinstance(image_np, np.memmap) or image_np.size > 100_000_000: # >100MB
             # Subsample
             step = int(max(image_np.shape[0], image_np.shape[1]) / 2000)
             if step < 1: step = 1
             # Slice: image_np[::step, ::step]
             display_data = image_np[::step, ::step].copy()
        else:
             display_data = image_np

        self.full_res_image = (
            np.clip(display_data, 0, 255).astype(np.uint8)
            if display_data.dtype != np.uint8
            else display_data
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
        
        # Basic visual interpolation
        resized = cv2.resize(cropped, (dw, dh), interpolation=cv2.INTER_NEAREST)
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
        self.root = ctk.CTk()
        self.root.title("DStretch Python v4.0 - Premium Edition")
        
        # Maximize or Set Size
        screen_width = self.root.winfo_screenwidth()
        w = min(1600, int(screen_width * 0.9))
        h = int(w * 0.6)
        self.root.geometry(f"{w}x{h}")

        self.dstretch = DecorrelationStretch() if DecorrelationStretch else None
        self.original_image = None
        self.preprocessed_image = None
        self.processed_image = None
        self.original_dimensions = None
        
        self.current_colorspace = "YDS"
        self.active_processors = set()
        self.advanced_settings = self._get_default_advanced_settings()

        # Processor Logic (same as before)
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

        self.status_var = tk.StringVar(value="Ready. Load an image to start.")
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

    # Removed _setup_styles (CTk handles styles/themes)

    def _setup_ui(self):
        # Main layout using Grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        main_container = ctk.CTkFrame(self.root)
        main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        main_container.columnconfigure(0, weight=5) # Image area
        main_container.columnconfigure(1, weight=1) # sidebar
        main_container.rowconfigure(0, weight=1) # full height
        
        # 1. Image Frame (Left)
        self.image_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.canvas = MemoryOptimizedImageCanvas(self.image_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 2. Controls Frame (Right)
        self.controls_frame = ctk.CTkFrame(main_container, corner_radius=10)
        self.controls_frame.grid(row=0, column=1, sticky="nsew")
        
        self._setup_standard_mode()
        
        # 3. Status Bar (Bottom)
        self.status_label = ctk.CTkLabel(self.root, textvariable=self.status_var, anchor="w", text_color="gray70", height=20)
        self.status_label.grid(row=1, column=0, sticky="ew", padx=15, pady=2)

    def _setup_standard_mode(self):
        # Clear
        for widget in self.controls_frame.winfo_children():
            widget.destroy()

        # Title
        ctk.CTkLabel(self.controls_frame, text="Control Panel", font=("Roboto", 18, "bold")).pack(pady=(15, 10))

        # File Actions
        file_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkButton(file_frame, text="Open Image", command=self._open_image, fg_color="#2da43e", hover_color="#2c974b").pack(fill=tk.X, pady=2)
        # Using pack side=left for next two?
        row_svc = ctk.CTkFrame(file_frame, fg_color="transparent")
        row_svc.pack(fill=tk.X, pady=2)
        ctk.CTkButton(row_svc, text="Original", width=80, command=self._restore_original, fg_color="#666666").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ctk.CTkButton(row_svc, text="Save", width=80, command=self._save_result).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

        # Separator
        ttk.Separator(self.controls_frame).pack(fill=tk.X, padx=10, pady=10)

        # Preprocessing
        ctk.CTkLabel(self.controls_frame, text="Preprocessing", font=("Roboto", 14, "bold")).pack(anchor="w", padx=10)
        
        prep_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        prep_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.preprocess_buttons = {}
        # Grid layout for toggles
        toggles = [
            ("invert", "Invert"), ("auto_contrast", "Contrast"),
            ("color_balance", "Balance"), ("flatten", "Flatten")
        ]
        for i, (key, text) in enumerate(toggles):
            btn = ctk.CTkButton(
                prep_frame, 
                text=text, 
                width=60, 
                height=30,
                border_width=1,
                fg_color="transparent", 
                text_color=("gray10", "gray90"),
                command=lambda n=key: self._toggle_preprocessing(n)
            )
            btn.grid(row=i//2, column=i%2, padx=2, pady=2, sticky="ew")
            self.preprocess_buttons[key] = btn
            
        prep_frame.columnconfigure(0, weight=1); prep_frame.columnconfigure(1, weight=1)
        
        # Tools row
        tools_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        tools_frame.pack(fill=tk.X, padx=5, pady=5)
        ctk.CTkButton(tools_frame, text="Hue Shift", width=80, command=lambda: self._toggle_preprocessing("hue_shift"), fg_color="transparent", border_width=1).pack(side=tk.LEFT, expand=True, padx=2)
        ctk.CTkButton(tools_frame, text="Quick Fix", width=80, command=self._apply_quick_enhance, fg_color="#8957e5").pack(side=tk.LEFT, expand=True, padx=2)
        
        ctk.CTkButton(self.controls_frame, text="Advanced Settings", command=self._open_advanced_window, fg_color="transparent", border_width=1).pack(fill=tk.X, padx=10, pady=5)

        # Separator
        ttk.Separator(self.controls_frame).pack(fill=tk.X, padx=10, pady=10)

        # Color Spaces
        ctk.CTkLabel(self.controls_frame, text="Color Spaces", font=("Roboto", 14, "bold")).pack(anchor="w", padx=10)
        cs_frame = ctk.CTkFrame(self.controls_frame)
        cs_frame.pack(fill=tk.X, padx=10, pady=5)
        self._setup_colorspace_grid(cs_frame)

        # Scale
        scale_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        scale_frame.pack(fill=tk.X, padx=10, pady=(15, 5))
        ctk.CTkLabel(scale_frame, text="Scale:").pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=15.0)
        self.scale_label = ctk.CTkLabel(scale_frame, text="15", width=30)
        self.scale_label.pack(side=tk.RIGHT)
        
        self.scale_slider = ctk.CTkSlider(
            scale_frame,
            from_=1.0,
            to=100.0,
            variable=self.scale_var,
            command=self._on_scale_changed
        )
        self.scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Footer
        footer = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        ctk.CTkButton(footer, text="Exit", command=self.root.quit, fg_color="transparent", border_width=1, text_color="red", hover_color="#550000").pack(fill=tk.X, padx=10)

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
                    # CTkButton for Grid
                    btn = ctk.CTkButton(
                        parent, 
                        text=name, 
                        width=40, 
                        height=28,
                        fg_color="transparent",
                        border_width=1,
                        text_color=("gray10", "gray90"),
                        command=cmd
                    )
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
                button.configure(fg_color=["#3B8ED0", "#1F6AA5"])

    def _rebuild_from_advanced_settings(self):
        self._set_status("Applying advanced settings...")
        for name, btn in self.preprocess_buttons.items():
            if name in self.processors:
                btn.configure(
                    fg_color="green" # Active color
                    if name in self.active_processors
                    else ["#3B8ED0", "#1F6AA5"] # Default color
                )
        self._rebuild_current_image()

    def _restore_settings(self):
        self.scale_var.set(15.0)
        self.scale_label.configure(text="15")
        self.advanced_settings = self._get_default_advanced_settings()
        self._reset_button_states()
        if self.original_image is not None:
            self.preprocessed_image = self.original_image
            self.processed_image = self.original_image
            self.canvas.set_image(self.original_image)
        self._set_status("Settings restored to defaults")
        gc.collect()

    def _reset_button_states(self):
        for btn in self.preprocess_buttons.values():
            btn.configure(fg_color="transparent")
        for btn in self.colorspace_buttons.values():
            if btn.cget("text") != "CUSTOM":
                 btn.configure(fg_color="transparent")

    def _show_about(self):
        messagebox.showinfo(
            "About DStretch Python",
            "DStretch Python v4.0 - Premium Edition\n\n"
            "Features:\n"
            "• Massive Image Support (Auto-Memmap Streaming)\n"
            "• Advanced Neural/Algorithmic Enhancement\n"
            "• Premium Dark UI\n"
            "• 23+ Specialized Color Spaces",
        )

    def _update_rebuilt_result(self, rebuilt_image):
        self.preprocessed_image = rebuilt_image
        
        # If streaming/memmap, this might be a handle.
        # But rebuilt_image usually comes from 'apply_single_processor' which 
        # for now assumes RAM processing for pre-processors (Invert/Contrast).
        # TODO: Streaming Preprocessing.
        # For now, pre-processing huge images is skipped or might crash if not handled.
        # Subsampling for display is handled in Canvas.
        
        self.processed_image = rebuilt_image
        # Canvas handles subsampling
        self.canvas.set_image(self.processed_image)
        
        active = sorted(list(self.active_processors))
        self._set_status(
            f"Rebuilt with: {' + '.join(active)}" if active else "Restored to original"
        )

    def _apply_current_colorspace_safe(self):
        if self.original_image is None or not self.dstretch:
            self._set_status("No image loaded")
            return
        
        # Determine source
        source = (
            self.preprocessed_image
            if self.preprocessed_image is not None
            else self.original_image
        )
        
        try:
            # We skip 'optimize_image_for_processing' (resizing/clipping) logic 
            # for huge images to keep full res
            if not isinstance(source, np.memmap):
                 source = self._optimize_image_for_processing(source)
            
            if source is None: return

            self._set_status(f"Processing with {self.current_colorspace}...")

            def process():
                try:
                    if self.dstretch:
                        # process() now handles memmap streaming automatically
                        result = self.dstretch.process(
                            source, self.current_colorspace, self.scale_var.get()
                        )
                        self.root.after(
                            0, self._update_colorspace_result, result.processed_image
                        )
                except Exception as e:
                    self.root.after(0, self._set_status, f"Error processing: {e}")
                    import traceback
                    traceback.print_exc()
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
        from .io_utils import smart_load_image
        
        filename = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            try:
                self._set_status("Loading image...")
                # Use smart loader
                res = smart_load_image(filename)
                self.original_image = res.data
                self.original_dimensions = (
                    self.original_image.shape[0],
                    self.original_image.shape[1],
                )
                
                # If memmap, preprocessed is ref to same memmap
                self.preprocessed_image = self.original_image
                self.processed_image = self.original_image
                
                self.canvas.set_image(self.original_image)
                self._restore_settings()
                
                mem_type = "DISK/STREAMING" if res.is_memmap else "RAM"
                self._set_status(
                    f"Loaded: {Path(filename).name} ({self.original_image.shape[1]}x{self.original_image.shape[0]}) [{mem_type}]"
                )
                gc.collect()
            except Exception as e:
                messagebox.showerror("Open Error", f"Could not open image:\n{e}")
                import traceback
                traceback.print_exc()

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
                # Toggle active/inactive colors
                btn.configure(fg_color="green" if n == name else ["#3B8ED0", "#1F6AA5"])
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
            # Inactive color
            btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
            self._set_status(f"Disabled {name}")
            self._rebuild_current_image()
        else:
            self.active_processors.add(name)
            try:
                # Active color
                btn.configure(fg_color="green")
            except Exception:
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
                # Reset to default theme color
                btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
        if hasattr(self, "colorspace_buttons"):
            for n, btn in self.colorspace_buttons.items():
                if n != "CUSTOM":
                    # Reset to default theme color
                    btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
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
