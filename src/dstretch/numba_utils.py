"""
Numba-optimized functions for heavy pixel-wise operations.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def lab_to_rgb_fast(L, a, b, D65_WHITE, XYZ_TO_RGB, output_shape):
    """
    Convert LAB colorspace to RGB using Numba JIT acceleration.
    """
    height, width = output_shape
    result = np.empty((height, width, 3), dtype=np.uint8)
    
    # Constants from matrix for faster access
    xr, xg, xb = XYZ_TO_RGB[0, 0], XYZ_TO_RGB[0, 1], XYZ_TO_RGB[0, 2]
    yr, yg, yb = XYZ_TO_RGB[1, 0], XYZ_TO_RGB[1, 1], XYZ_TO_RGB[1, 2]
    zr, zg, zb = XYZ_TO_RGB[2, 0], XYZ_TO_RGB[2, 1], XYZ_TO_RGB[2, 2]
    
    white_x, white_y, white_z = D65_WHITE[0], D65_WHITE[1], D65_WHITE[2]

    for i in prange(height):
        for j in range(width):
            l_val = L[i, j]
            a_val = a[i, j]
            b_val = b[i, j]
            
            fy = (l_val + 16.0) / 116.0
            fx = a_val / 500.0 + fy
            fz = fy - b_val / 200.0
            
            # inv_f logic inline
            if fx > 0.206893:
                x_normalized = fx * fx * fx
            else:
                x_normalized = (fx - 16.0 / 116.0) / 7.787
                
            if fy > 0.206893:
                y_normalized = fy * fy * fy
            else:
                y_normalized = (fy - 16.0 / 116.0) / 7.787
                
            if fz > 0.206893:
                z_normalized = fz * fz * fz
            else:
                z_normalized = (fz - 16.0 / 116.0) / 7.787
            
            # XYZ scaling
            X = x_normalized * white_x
            Y = y_normalized * white_y
            Z = z_normalized * white_z
            
            # Matrix multiplication (XYZ -> Linear RGB)
            r_lin = (X * xr + Y * xg + Z * xb) / 100.0
            g_lin = (X * yr + Y * yg + Z * yb) / 100.0
            b_lin = (X * zr + Y * zg + Z * zb) / 100.0
            
            # Gamma correction (Linear RGB -> sRGB)
            if r_lin <= 0.0031308:
                r_srgb = r_lin * 12.92
            else:
                r_srgb = 1.055 * (r_lin ** (1.0 / 2.4)) - 0.055

            if g_lin <= 0.0031308:
                g_srgb = g_lin * 12.92
            else:
                g_srgb = 1.055 * (g_lin ** (1.0 / 2.4)) - 0.055
                
            if b_lin <= 0.0031308:
                b_srgb = b_lin * 12.92
            else:
                b_srgb = 1.055 * (b_lin ** (1.0 / 2.4)) - 0.055
                
            # Clip and cast
            result[i, j, 0] = min(max(int(r_srgb * 255.0), 0), 255)
            result[i, j, 1] = min(max(int(g_srgb * 255.0), 0), 255)
            result[i, j, 2] = min(max(int(b_srgb * 255.0), 0), 255)
            
    return result

@njit(parallel=True, fastmath=True, cache=True)
def rgb_to_lab_fast(rgb_image, rgb_to_xyz_lut, RGB_TO_XYZ, D65_WHITE, xyz_to_lab_lut):
    """
    Convert RGB to LAB using Numba optimization.
    NOTE: Using LUTs inside parallel loop might be tricky with cache coherency,
    relying on read-only access.
    """
    height, width, _ = rgb_image.shape
    result = np.empty((height, width, 3), dtype=np.float64)
    
    # Unpack constants
    xr, xg, xb = RGB_TO_XYZ[0, 0], RGB_TO_XYZ[0, 1], RGB_TO_XYZ[0, 2]
    yr, yg, yb = RGB_TO_XYZ[1, 0], RGB_TO_XYZ[1, 1], RGB_TO_XYZ[1, 2]
    zr, zg, zb = RGB_TO_XYZ[2, 0], RGB_TO_XYZ[2, 1], RGB_TO_XYZ[2, 2]
    
    wx, wy, wz = D65_WHITE[0], D65_WHITE[1], D65_WHITE[2]
    

    
    # We can't easily use the large LUT in cache-friendly way per-thread 
    # unless it fits L1/L2. sRGB->Linear LUT is small (256 floats).
    # XYZ->LAB LUT is big (4096+). 
    # For now, let's replicate the logic or just use the passed LUTs efficiently.
    
    lab_lut_len = len(xyz_to_lab_lut) - 1
    
    for i in prange(height):
        for j in range(width):
            r = rgb_image[i, j, 0]
            g = rgb_image[i, j, 1]
            b = rgb_image[i, j, 2]
            
            # Linearize via LUT (fast)
            # Assuming rgb_to_xyz_lut is array of float32
            r_lin = rgb_to_xyz_lut[r]
            g_lin = rgb_to_xyz_lut[g]
            b_lin = rgb_to_xyz_lut[b]
            
            # RGB -> XYZ Matrix
            X = r_lin * xr + g_lin * xg + b_lin * xb
            Y = r_lin * yr + g_lin * yg + b_lin * yb
            Z = r_lin * zr + g_lin * zg + b_lin * zb
            
            # Normalize XYZ by White Point
            X_norm = X / wx
            Y_norm = Y / wy
            Z_norm = Z / wz
            
            # XYZ -> LAB via LUT interpolation or direct index
            # Replicating logic: f_xyz_indices = (xyz_norm_clamped * (len - 1)).astype(int)
            
            # Clamp
            if X_norm < 0.0: X_norm = 0.0
            if X_norm > 1.0: X_norm = 1.0
            
            if Y_norm < 0.0: Y_norm = 0.0
            if Y_norm > 1.0: Y_norm = 1.0
            
            if Z_norm < 0.0: Z_norm = 0.0
            if Z_norm > 1.0: Z_norm = 1.0
            
            idx_x = int(X_norm * lab_lut_len)
            idx_y = int(Y_norm * lab_lut_len)
            idx_z = int(Z_norm * lab_lut_len)
            
            fx = xyz_to_lab_lut[idx_x]
            fy = xyz_to_lab_lut[idx_y]
            fz = xyz_to_lab_lut[idx_z]
            
            L_val = 116.0 * fy - 16.0
            a_val = 500.0 * (fx - fy)
            b_val = 200.0 * (fy - fz)
            
            result[i, j, 0] = L_val
            result[i, j, 1] = a_val
            result[i, j, 2] = b_val
            
    return result
