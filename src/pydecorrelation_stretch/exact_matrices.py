"""
Datos de Matriz y Constantes para el Algoritmo DStretch - VERSIÓN DEFINITIVA.

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1

Este módulo contiene todas las matrices de transformación, constantes numéricas
y funciones para construir las Tablas de Búsqueda (LUTs) necesarias para replicar
fielmente el plugin DStretch de ImageJ.

Los valores han sido extraídos y verificados a partir del análisis del código
fuente Java original (`DStretch_.java`).
"""

import numpy as np

# =============================================================================
# CONSTANTES DE ESPACIOS DE COLOR ESTÁNDAR
# =============================================================================

# Punto blanco de referencia D65 estándar, utilizado en la conversión a CIELAB.
D65_ILLUMINANT = np.array([95.047, 100.0, 108.883], dtype=np.float64)

# Matriz de transformación estándar de sRGB (linealizado) a XYZ.
RGB_TO_XYZ_MATRIX = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]],
    dtype=np.float64,
)

# Matriz de transformación estándar de XYZ a sRGB (linealizado).
# Es la inversa de la anterior, con valores exactos del código Java.
XYZ_TO_RGB_MATRIX = np.array(
    [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]],
    dtype=np.float64,
)

# Matriz de transformación de RGB a YCbCr (parte lineal).
YCBCR_MATRIX = np.array(
    [
        [0.25679, 0.50413, 0.09790],
        [-0.148223, -0.291, 0.43922],
        [0.43922, -0.367789, -0.071427],
    ],
    dtype=np.float64,
)

# Matriz de transformación de RGB a YUV.
YUV_MATRIX = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.147, -0.289, 0.436],  # Corresponde a 0.492 * (B - Y)
        [0.615, -0.515, -0.100],  # Corresponde a 0.877 * (R - Y)
    ],
    dtype=np.float64,
)

# Matriz inversa de YUV a RGB, con valores exactos del código.
YUV_INVERSE_MATRIX = np.array(
    [[1.000, 0.000, 1.140], [1.000, -0.395, -0.581], [1.000, 2.032, 0.000]],
    dtype=np.float64,
)


# =============================================================================
# MATRICES "BUILT-IN" (PREDEFINIDAS) PARA MEJORAS ESPECÍFICAS
# =============================================================================

# Estas matrices se aplican directamente en un espacio de color específico
# para lograr un efecto de realce particular sin necesidad de análisis estadístico.

BUILTIN_MATRICES = {
    "CRGB": np.array(
        [[0.37, 0.34, 0.30], [-3.80, 7.70, -4.00], [-1.80, 0.22, 2.00]],
        dtype=np.float64,
    ),
    "RGB0": np.array(
        [[0.38, 0.32, 0.33], [-2.30, 3.20, -0.42], [-0.47, -0.76, 2.43]],
        dtype=np.float64,
    ),
    # Esta matriz se aplica en el espacio de color LAB.
    "LABI": np.array(
        [[0.21, 4.64, -0.64], [-0.85, 0.05, 0.09], [0.34, 0.42, 3.13]], dtype=np.float64
    ),
    # Las siguientes matrices se aplican en el espacio YXX con parámetros específicos,
    # pero el código Java también las tiene como matrices precalculadas.
    # Aquí las incluimos para una replicación completa.
    "YBG": np.array(
        [[5.68, 0.44, 2.29], [0.44, 1.86, 0.96], [2.29, 0.96, 1.81]], dtype=np.float64
    ),
    "YBL": np.array(
        [[8.86, -0.92, 3.00], [-0.92, 5.69, 2.18], [3.00, 2.18, 2.76]], dtype=np.float64
    ),
}

# =============================================================================
# FUNCIONES PARA CONSTRUIR TABLAS DE BÚSQUEDA (LUTs)
# =============================================================================


def build_srgb_to_linear_lut() -> np.ndarray:
    """
    Construye la LUT para la corrección gamma inversa (sRGB a lineal).

    Replica la lógica exacta de `setrgb2xyzlut` del código Java.
    El resultado se escala por 100 como en el original.
    """
    srgb_normalized = np.arange(256) / 255.0
    linear = np.where(
        srgb_normalized <= 0.04045,
        srgb_normalized / 12.92,
        ((srgb_normalized + 0.055) / 1.055) ** 2.4,
    )
    return linear * 100.0


def build_xyz_to_lab_function_lut() -> np.ndarray:
    """
    Construye la LUT para la función no lineal f(t) usada en la conversión a CIELAB.

    Replica la lógica de `setxyz2lablut` del código Java.
    """
    t = np.linspace(0, 1, 1001)  # 1001 puntos como en el original
    f_t = np.where(t > 0.008856, t ** (1.0 / 3.0), 7.787 * t + (16.0 / 116.0))
    return f_t


# =============================================================================
# EJEMPLO DE USO (se puede comentar o eliminar)
# =============================================================================
if __name__ == "__main__":
    print("--- DStretch Matrix Data ---")
    print("\nMatriz CRGB:")
    print(BUILTIN_MATRICES["CRGB"])

    print("\nMatriz RGB a XYZ:")
    print(RGB_TO_XYZ_MATRIX)

    # Construir y probar las LUTs
    srgb_lut = build_srgb_to_linear_lut()
    lab_lut = build_xyz_to_lab_function_lut()

    print(f"\nTamaño de la LUT sRGB->Lineal: {len(srgb_lut)}")
    print(f"Valor para sRGB=128: {srgb_lut[128]:.4f}")

    print(f"\nTamaño de la LUT f(t) para LAB: {len(lab_lut)}")
    print(f"Valor para t=0.5: {lab_lut[500]:.4f}")
