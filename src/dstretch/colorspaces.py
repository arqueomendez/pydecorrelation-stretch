"""
Color space transformations for DStretch algorithm - FINAL VERSION 5.0

This version applies the definitive correction to the LXXColorspace class,
ensuring its parametric transformation is a precise replica of the original
Java implementation. This resolves the final validation discrepancies.

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1
"""

from abc import ABC, abstractmethod

import numpy as np

from .exact_matrices import (
    BUILTIN_MATRICES,
    D65_ILLUMINANT,
    RGB_TO_XYZ_MATRIX,
    XYZ_TO_RGB_MATRIX,
    build_srgb_to_linear_lut,
    build_xyz_to_lab_function_lut,
)


# --- CLASES BASE (sin cambios) ---
class AbstractColorspace(ABC):
    # ... (sin cambios)

    # Abstract attributes to be defined in subclasses
    # Abstract attributes to be defined in subclasses
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the colorspace."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the colorspace."""
        pass

    @property
    @abstractmethod
    def optimized_for(self) -> list[str]:
        """List of use cases this colorspace is optimized for."""
        pass

    @property
    def scale_adjust_factor(self) -> float:
        return 3.0

    @abstractmethod
    def to_colorspace(self, rgb_image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def from_colorspace(self, color_image: np.ndarray) -> np.ndarray:
        pass


class BuiltinMatrixColorspace(AbstractColorspace):
    # ... (sin cambios)
    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        pass

    # Abstract attribute
    base_colorspace_name: str = ""

    @property
    def scale_adjust_factor(self) -> float:
        return 1.0

    def to_colorspace(self, rgb_image: np.ndarray) -> np.ndarray:
        return rgb_image

    def from_colorspace(self, color_image: np.ndarray) -> np.ndarray:
        return color_image


# --- IMPLEMENTACIONES DE ESPACIOS DE COLOR (con corrección en LXX) ---


class RGBColorspace(AbstractColorspace):
    # ... (sin cambios)
    @property
    def name(self) -> str:
        return "RGB"

    @property
    def description(self) -> str:
        return "Standard RGB. Fast, general purpose."

    @property
    def optimized_for(self) -> list[str]:
        return ["general"]

    def to_colorspace(self, rgb_image: np.ndarray) -> np.ndarray:
        return rgb_image.astype(np.float64)

    def from_colorspace(self, color_image: np.ndarray) -> np.ndarray:
        return np.clip(color_image, 0, 255).astype(np.uint8)


class LABColorspace(AbstractColorspace):
    # ... (sin cambios)
    @property
    def name(self) -> str:
        return "LAB"

    @property
    def description(self) -> str:
        return "CIE LAB. Perceptually uniform."

    @property
    def optimized_for(self) -> list[str]:
        return ["general", "natural_colors", "whites", "blacks"]

    @property
    def scale_adjust_factor(self) -> float:
        return 1.5

    def __init__(self):
        self.D65_WHITE, self.RGB_TO_XYZ, self.XYZ_TO_RGB = (
            D65_ILLUMINANT,
            RGB_TO_XYZ_MATRIX,
            XYZ_TO_RGB_MATRIX,
        )
        self.rgb_to_xyz_lut = build_srgb_to_linear_lut()
        self.xyz_to_lab_lut = build_xyz_to_lab_function_lut()

    def to_colorspace(self, rgb_image: np.ndarray) -> np.ndarray:
        rgb_linear = self.rgb_to_xyz_lut[rgb_image]
        xyz_image = np.einsum("ij,...j->...i", self.RGB_TO_XYZ, rgb_linear)
        xyz_norm = xyz_image / self.D65_WHITE
        xyz_norm_clamped = np.clip(xyz_norm, 0.0, 1.0)
        f_xyz_indices = (xyz_norm_clamped * (len(self.xyz_to_lab_lut) - 1)).astype(int)
        f_xyz = self.xyz_to_lab_lut[f_xyz_indices]
        L = 116.0 * f_xyz[..., 1] - 16.0
        a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
        b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
        return np.stack([L, a, b], axis=-1)

    def from_colorspace(self, color_image: np.ndarray) -> np.ndarray:
        L, a, b = color_image[..., 0], color_image[..., 1], color_image[..., 2]
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        def inv_f(t):
            t_cubed = t**3
            return np.where(t > 0.206893, t_cubed, (t - 16.0 / 116.0) / 7.787)

        xyz_image = (
            np.stack([inv_f(fx), inv_f(fy), inv_f(fz)], axis=-1) * self.D65_WHITE
        )
        rgb_linear = np.einsum("ij,...j->...i", self.XYZ_TO_RGB, xyz_image)
        rgb_linear_norm = np.clip(rgb_linear / 100.0, 0.0, 1.0)
        rgb_srgb = np.where(
            rgb_linear_norm <= 0.0031308,
            rgb_linear_norm * 12.92,
            1.055 * (rgb_linear_norm ** (1.0 / 2.4)) - 0.055,
        )
        return np.clip(rgb_srgb * 255.0, 0, 255).astype(np.uint8)


class YXXColorspace(AbstractColorspace):
    # ... (sin cambios)
    def __init__(self, yxxmuly, yxxmulu, yxxmulv, name, description, optimized_for):
        self._name, self._description, self._optimized_for = (
            name,
            description,
            optimized_for,
        )
        self.yxxmuly, self.yxxmulu, self.yxxmulv = yxxmuly, yxxmulu, yxxmulv

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def optimized_for(self):
        return self._optimized_for

    def to_colorspace(self, rgb_image: np.ndarray) -> np.ndarray:
        rgb_float = rgb_image.astype(np.float64)
        R, G, B = rgb_float[..., 0], rgb_float[..., 1], rgb_float[..., 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = self.yxxmuly * (B - self.yxxmulu * Y)
        V = self.yxxmuly * (R - self.yxxmulv * Y)
        return np.stack([Y, U, V], axis=-1)

    def from_colorspace(self, color_image: np.ndarray) -> np.ndarray:
        Y, U, V = color_image[..., 0], color_image[..., 1], color_image[..., 2]
        R = V / self.yxxmuly + self.yxxmulv * Y
        B = U / self.yxxmuly + self.yxxmulu * Y
        G = (Y - 0.299 * R - 0.114 * B) / 0.587
        rgb_image = np.stack([R, G, B], axis=-1)
        return np.clip(rgb_image, 0, 255).astype(np.uint8)


# *** CLASE LXXColorspace CORREGIDA ***
class LXXColorspace(AbstractColorspace):
    @property
    def scale_adjust_factor(self) -> float:
        return 1.5

    def __init__(
        self, lxxmul1, lxxmul2, lxxmula, lxxmulb, name, description, optimized_for
    ):
        self._name, self._description, self._optimized_for = (
            name,
            description,
            optimized_for,
        )
        self.lxxmul1, self.lxxmul2, self.lxxmula, self.lxxmulb = (
            lxxmul1,
            lxxmul2,
            lxxmula,
            lxxmulb,
        )
        self.lab_processor = LABColorspace()

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def optimized_for(self):
        return self._optimized_for

    def to_colorspace(self, rgb_image: np.ndarray) -> np.ndarray:
        # La lógica de Java no escala los componentes L, a, b directamente.
        # En su lugar, modifica las relaciones f(t) subyacentes.
        # Esto es equivalente a escalar los componentes a y b, pero manteniendo L intacto
        # y usando los parámetros lxxmul1/2 como si fueran para a y b.
        # Esta implementación es una traducción directa de `rgb2lxx`.
        lab_image = self.lab_processor.to_colorspace(rgb_image)
        L, a, b = lab_image[..., 0], lab_image[..., 1], lab_image[..., 2]
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        # El código Java es `250 * (fx - lxxmula * fy) / lxxmul1`
        A_comp = (1.0 / self.lxxmul1) * 250.0 * (fx - self.lxxmula * fy)
        B_comp = (1.0 / self.lxxmul2) * 100.0 * (self.lxxmulb * fy - fz)

        return np.stack([L, A_comp, B_comp], axis=-1)

    def from_colorspace(self, color_image: np.ndarray) -> np.ndarray:
        L, A_comp, B_comp = (
            color_image[..., 0],
            color_image[..., 1],
            color_image[..., 2],
        )
        fy = (L + 16.0) / 116.0

        # Invertir las fórmulas de `to_colorspace` para encontrar fx y fz
        fx = (A_comp * self.lxxmul1 / 250.0) + self.lxxmula * fy
        fz = (self.lxxmulb * fy) - (B_comp * self.lxxmul2 / 100.0)

        # Reconstruir los componentes a y b originales de LAB
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)

        return self.lab_processor.from_colorspace(np.stack([L, a, b], axis=-1))


# (El resto del archivo no necesita cambios)
class YDSColorspace(YXXColorspace):
    def __init__(self):
        super().__init__(1.0, 0.5, 1.0, "YDS", "Yellows", ["yellow", "general"])


class YBRColorspace(YXXColorspace):
    def __init__(self):
        super().__init__(1.0, 0.8, 0.4, "YBR", "Reds", ["red"])


class YBKColorspace(YXXColorspace):
    def __init__(self):
        super().__init__(1.5, 0.2, 1.6, "YBK", "Blacks/blues", ["black", "blue"])


class YREColorspace(YXXColorspace):
    def __init__(self):
        super().__init__(8.0, 1.0, 0.4, "YRE", "Extreme reds", ["red"])


class YRDColorspace(YXXColorspace):
    def __init__(self):
        super().__init__(2.0, 1.0, 0.4, "YRD", "Red pigments", ["red"])


class YWEColorspace(YXXColorspace):
    def __init__(self):
        super().__init__(1.5, 1.6, 0.2, "YWE", "White pigments", ["white"])


class YBLColorspace(YXXColorspace):
    def __init__(self):
        super().__init__(1.5, 0.4, 2.0, "YBL", "Blacks/greens", ["black", "green"])


class YBGColorspace(YXXColorspace):
    def __init__(self):
        super().__init__(2.0, 1.0, 1.7, "YBG", "Green pigments", ["green"])


class YUVColorspace(YXXColorspace):
    def __init__(self):
        super().__init__(0.7, 1.0, 1.0, "YUV", "General purpose", ["general"])


class YYEColorspace(YXXColorspace):
    def __init__(self):
        super().__init__(2.0, 2.0, 1.0, "YYE", "Yellows to brown", ["yellow"])


class LAXColorspace(LXXColorspace):
    def __init__(self):
        super().__init__(1.0, 1.0, 1.0, 1.0, "LAX", "LAB variant", ["general"])


class LDSColorspace(LXXColorspace):
    def __init__(self):
        super().__init__(0.5, 0.5, 0.9, 0.5, "LDS", "Yellows", ["yellow"])


class LREColorspace(LXXColorspace):
    def __init__(self):
        super().__init__(0.5, 0.5, 0.5, 1.0, "LRE", "Reds", ["red", "natural_colors"])

    @property
    def scale_adjust_factor(self) -> float:
        return 0.75


class LRDColorspace(LXXColorspace):
    def __init__(self):
        super().__init__(0.5, 0.5, 0.8, 1.2, "LRD", "Red pigments", ["red"])


class LBKColorspace(LXXColorspace):
    def __init__(self):
        super().__init__(0.5, 0.5, 1.1, 0.6, "LBK", "Black pigments", ["black"])


class LBLColorspace(LXXColorspace):
    def __init__(self):
        super().__init__(0.5, 0.5, 1.2, 1.0, "LBL", "Black pigments", ["black"])


class LWEColorspace(LXXColorspace):
    def __init__(self):
        super().__init__(0.5, 0.5, 1.0, 1.4, "LWE", "White pigments", ["white"])


class LYEColorspace(LXXColorspace):
    def __init__(self):
        super().__init__(0.2, 0.2, 1.0, 2.0, "LYE", "Yellows to brown", ["yellow"])


class CRGBColorspace(BuiltinMatrixColorspace):
    @property
    def name(self) -> str:
        return "CRGB"

    @property
    def description(self) -> str:
        return "Pre-calculated matrix for faint reds"

    @property
    def optimized_for(self) -> list[str]:
        return ["red", "faint_pigments"]
    base_colorspace_name = "RGB"

    @property
    def matrix(self):
        return BUILTIN_MATRICES["CRGB"]


# Nota: BUILTIN_MATRICES ahora debe ser importado si se usa aquí.
# O, mejor aún, que las clases que lo usan lo importen directamente.


class RGB0Colorspace(BuiltinMatrixColorspace):
    @property
    def name(self) -> str:
        return "RGB0"

    @property
    def description(self) -> str:
        return "Built-in matrix for enhancing reds"

    @property
    def optimized_for(self) -> list[str]:
        return ["red"]
    base_colorspace_name = "RGB"

    @property
    def matrix(self):
        return BUILTIN_MATRICES["RGB0"]


class LABIColorspace(BuiltinMatrixColorspace):
    @property
    def name(self) -> str:
        return "LABI"

    @property
    def description(self) -> str:
        return "Built-in matrix for inverted colors (applied in LAB space)"

    @property
    def optimized_for(self) -> list[str]:
        return ["special_effect"]
    base_colorspace_name = "LAB"

    @property
    def matrix(self):
        return BUILTIN_MATRICES["LABI"]


COLORSPACES: dict[str, AbstractColorspace] = {
    cs.name: cs
    for cs in [
        RGBColorspace(),
        LABColorspace(),
        YDSColorspace(),
        YBRColorspace(),
        YBKColorspace(),
        YREColorspace(),
        YRDColorspace(),
        YWEColorspace(),
        YBLColorspace(),
        YBGColorspace(),
        YUVColorspace(),
        YYEColorspace(),
        LAXColorspace(),
        LDSColorspace(),
        LREColorspace(),
        LRDColorspace(),
        LBKColorspace(),
        LBLColorspace(),
        LWEColorspace(),
        LYEColorspace(),
        CRGBColorspace(),
        RGB0Colorspace(),
        LABIColorspace(),
    ]
}


class ColorspaceManager:
    """Manager for available colorspaces."""

    def list_available(self):
        """List all available colorspace names."""
        return list(COLORSPACES.keys())

    def get_colorspace(self, name):
        """Get a colorspace instance by name."""
        if name not in COLORSPACES:
            raise ValueError(f"Unknown colorspace '{name}'")
        return COLORSPACES[name]

    def is_available(self, name):
        """Check if a colorspace is available."""
        return name in COLORSPACES
