"""
Validación automática con reconocimiento de patrones de nombres y logging detallado.
Formato esperado: [nombre]_[colorspace]_scale[número].jpg
Ejemplo: 13-_ybk_scale15.jpg

MODIFICADO: Todos los resultados de la validación se guardan en una
carpeta única con marca de tiempo para una fácil revisión.
"""

import datetime
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pydecorrelation_stretch.colorspaces import COLORSPACES  # Asumiendo que ahora está centralizado

# Asegúrate de que pydecorrelation_stretch y sus componentes estén accesibles
# por ejemplo, instalados o en PYTHONPATH.
from pydecorrelation_stretch.decorrelation import DecorrelationStretch

# --- Funciones de Utilidad (sin cambios) ---


def calculate_simple_ssim(img1, img2):
    """Calcula SSIM simplificado usando solo NumPy."""
    # (El código de esta función permanece sin cambios)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return numerator / denominator


# --- Clase Principal Modificada ---


class SmartValidator:
    """Validador que reconoce automáticamente patrones de nombres con logging."""

    def __init__(self, output_base_dir="validation_results"):
        self.dstretch = DecorrelationStretch()
        self.results = []

        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.results_dir = Path(output_base_dir) / f"review_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Llama a la nueva función de setup
        self.logger = self._setup_logging()

        # Genera el mapeo de colores dinámicamente para evitar desincronización.
        # Crea un diccionario de {nombre_en_minusculas: nombre_oficial}
        self.colorspace_mapping = {name.lower(): name for name in COLORSPACES.keys()}

        self.logger.info(
            f"Validator initialized. All results will be saved to: {self.results_dir}"
        )
        self.logger.info(
            f"Initialized with {len(self.colorspace_mapping)} colorspace mappings"
        )

    def _setup_logging(self):
        """
        CORREGIDO: Configura el sistema de logging añadiendo manejadores explícitamente.
        """
        # Crear el logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Evitar que se añadan manejadores duplicados si la función se llama varias veces
        if logger.hasHandlers():
            logger.handlers.clear()

        # Crear el formateador
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        # 1. Crear el manejador para el archivo de log
        log_filepath = self.results_dir / f"validation_log_{self.timestamp}.log"
        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # 2. Crear el manejador para la consola
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        # 3. Añadir ambos manejadores al logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        logger.info("=" * 80)
        logger.info("DSTRETCH PYTHON VALIDATION LOG STARTED")
        logger.info(f"Results Directory: {self.results_dir.resolve()}")
        logger.info("=" * 80)

        return logger

    # --- Los métodos `parse_filename`, `find_original_image`, `discover_images` permanecen sin cambios ---

    def parse_filename(self, filename):
        """Parsea el nombre del archivo para extraer información."""
        # Esta expresión regular ahora es más flexible con el nombre base
        pattern = r"^(.+?)_([a-zA-Z0-9]+)_scale(\d+)\.jpg$"
        match = re.match(pattern, filename)

        if match:
            original_name = match.group(1)
            colorspace_raw = match.group(2).lower()
            scale = int(match.group(3))
            colorspace = self.colorspace_mapping.get(
                colorspace_raw, colorspace_raw.upper()
            )

            self.logger.debug(
                f"Parsed {filename}: {original_name} | {colorspace} | {scale}"
            )

            return {
                "original_name": original_name,
                "colorspace": colorspace,
                "scale": scale,
                "valid": True,
            }

        self.logger.debug(f"Could not parse filename: {filename}")
        return {"valid": False}

    def find_original_image(self, original_name, directory="validation_images"):
        """Busca la imagen original basada en el nombre base."""
        # (El código de esta función permanece sin cambios)
        directory = Path(directory)
        extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
        variations = [
            original_name,
            original_name.rstrip("-"),
            original_name.rstrip("_"),
        ]

        for ext in extensions:
            for variation in variations:
                # Búsqueda más robusta en subdirectorios
                for file_path in directory.rglob(f"{variation}{ext}"):
                    # Asegurarse de no confundir un archivo procesado con el original
                    if "_scale" not in file_path.name:
                        self.logger.info(
                            f"Original found: {file_path.name} for {original_name}"
                        )
                        return str(file_path)

        self.logger.warning(f"Original image not found for: {original_name}")
        return None

    def discover_images(self, directory="validation_images"):
        """Descubre automáticamente todas las imágenes procesadas."""
        # (El código de esta función permanece sin cambios)
        directory = Path(directory)
        processed_files = []
        image_groups = defaultdict(list)

        self.logger.info(f"Discovering images in: {directory}")

        for file_path in directory.rglob("*.jpg"):
            parsed = self.parse_filename(file_path.name)
            if parsed["valid"]:
                processed_files.append(
                    {"file_path": str(file_path), "filename": file_path.name, **parsed}
                )
                image_groups[parsed["original_name"]].append(parsed)

        self.logger.info(
            f"Discovery complete: {len(processed_files)} processed images found"
        )
        self.logger.info(f"Grouped into {len(image_groups)} original images")

        for original_name, group in image_groups.items():
            colorspaces = [item["colorspace"] for item in group]
            self.logger.info(
                f"  {original_name}: {len(group)} variants - {colorspaces}"
            )

        return processed_files, image_groups

    def validate_all_discovered(self, directory="validation_images"):
        """Valida todas las imágenes descubiertas automáticamente."""

        self.logger.info("Starting automatic validation process")

        processed_files, _ = self.discover_images(directory)

        if not processed_files:
            self.logger.error(
                "No processed images found with expected pattern in the "
                "specified directory."
            )
            self.logger.info("Expected pattern: [name]_[colorspace]_scale[number].jpg")
            return

        # ... (resto del método sin cambios hasta el final)
        validated_count = 0
        failed_count = 0

        for file_info in processed_files:
            original_name = file_info["original_name"]
            colorspace = file_info["colorspace"]
            scale = file_info["scale"]
            processed_path = file_info["file_path"]

            self.logger.info(f"--- Processing: {file_info['filename']} ---")

            original_path = self.find_original_image(original_name, directory)
            if not original_path:
                self.logger.error(
                    f"SKIPPING: Original image not found for base name: "
                    f"'{original_name}'"
                )
                failed_count += 1
                continue

            result = self.validate_single_image(
                original_path, processed_path, colorspace, scale, original_name
            )

            if result:
                validated_count += 1
                self.logger.info(
                    f"Validation successful: {colorspace} | MSE={result['mse']:.1f} | "
                    f"SSIM={result['ssim']:.3f} | STATUS: {result['status']}"
                )
            else:
                failed_count += 1
                self.logger.error(f"Validation failed for: {file_info['filename']}")

        self.logger.info("=" * 50)
        self.logger.info("VALIDATION COMPLETED")
        self.logger.info(f"Successfully validated: {validated_count}")
        self.logger.info(f"Failed/Skipped validations: {failed_count}")
        self.logger.info(f"Total processed: {len(processed_files)}")

        self.generate_comprehensive_report()
        self.export_analysis_logs()

    def validate_single_image(
        self, original_path, imagej_path, colorspace, scale, image_name
    ):
        """Valida una imagen individual con logging detallado."""
        try:
            self.logger.debug(
                f"Loading images: {Path(original_path).name} | {Path(imagej_path).name}"
            )

            original_rgb = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
            imagej_rgb = cv2.cvtColor(cv2.imread(imagej_path), cv2.COLOR_BGR2RGB)

            self.logger.debug(
                f"Processing with DStretch Python: {colorspace} scale {scale}"
            )
            result = self.dstretch.process(original_rgb, colorspace, float(scale))
            our_result = result.processed_image

            # ... (resto del método sin cambios hasta guardar los resultados)
            mse = np.mean((imagej_rgb.astype(float) - our_result.astype(float)) ** 2)
            ssim_scores = [
                calculate_simple_ssim(imagej_rgb[:, :, c], our_result[:, :, c])
                for c in range(3)
            ]
            ssim_avg = np.mean(ssim_scores)

            if mse < 25 and ssim_avg > 0.95:
                status = "EXCELLENT"
            elif mse < 100 and ssim_avg > 0.90:
                status = "GOOD"
            elif mse < 300 and ssim_avg > 0.80:
                status = "ACCEPTABLE"
            else:
                status = "NEEDS_ADJUSTMENT"

            result_data = {
                "image_name": image_name,
                "colorspace": colorspace,
                "scale": scale,
                "mse": mse,
                "ssim": ssim_avg,
                "ssim_per_channel": ssim_scores,
                "max_difference": np.max(
                    np.abs(imagej_rgb.astype(float) - our_result.astype(float))
                ),
                "mean_difference": np.mean(
                    np.abs(imagej_rgb.astype(float) - our_result.astype(float))
                ),
                "significant_diff_pct": np.sum(
                    np.abs(imagej_rgb.astype(float) - our_result.astype(float)) > 10
                )
                / imagej_rgb.size
                * 100,
                "status": status,
                "original_file": Path(original_path).name,
                "imagej_file": Path(imagej_path).name,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            # ** CAMBIO: Guardar en el directorio de resultados **
            our_filename = (
                self.results_dir / f"python_{image_name}_{colorspace}_s{scale}.jpg"
            )
            cv2.imwrite(str(our_filename), cv2.cvtColor(our_result, cv2.COLOR_RGB2BGR))
            result_data["python_file"] = our_filename.name

            self.logger.debug(f"Python result saved: {our_filename}")

            # ** CAMBIO: Guardar comparación en el directorio de resultados **
            self.create_detailed_comparison(
                original_rgb, imagej_rgb, our_result, result_data
            )

            self.results.append(result_data)
            return result_data

        except Exception as e:
            self.logger.exception(
                f"CRITICAL VALIDATION ERROR for {image_name} {colorspace}: {e}"
            )
            return None

    def create_detailed_comparison(self, original, imagej_result, our_result, metrics):
        """Crea comparación visual detallada y la guarda en la carpeta de resultados."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # ... (código de ploteo sin cambios)
        axes[0, 0].imshow(original)
        axes[0, 0].set_title(
            f"Original\n{metrics['image_name']}", fontsize=12, fontweight="bold"
        )
        axes[0, 0].axis("off")
        axes[0, 1].imshow(imagej_result)
        axes[0, 1].set_title(
            f"ImageJ DStretch\n{metrics['colorspace']} (scale {metrics['scale']})",
            fontsize=12,
            fontweight="bold",
        )
        axes[0, 1].axis("off")
        axes[0, 2].imshow(our_result)
        axes[0, 2].set_title(
            f"DStretch Python\n{metrics['colorspace']} (scale {metrics['scale']})",
            fontsize=12,
            fontweight="bold",
        )
        axes[0, 2].axis("off")
        diff_image = np.abs(imagej_result.astype(float) - our_result.astype(float))
        diff_normalized = (
            diff_image / np.max(diff_image) if np.max(diff_image) > 0 else diff_image
        )
        axes[1, 0].imshow(diff_normalized, cmap="hot")
        axes[1, 0].set_title("Difference Map\n(Normalized)", fontsize=12)
        axes[1, 0].axis("off")
        axes[1, 1].hist(diff_image.flatten(), bins=50, color="red")
        axes[1, 1].set_title("Difference Distribution", fontsize=12)
        axes[1, 1].set_xlabel("Pixel Difference")
        axes[1, 1].set_ylabel("Frequency")
        status_colors = {
            "EXCELLENT": "darkgreen",
            "GOOD": "blue",
            "ACCEPTABLE": "orange",
            "NEEDS_ADJUSTMENT": "red",
        }
        metrics_text = (
            f"VALIDATION METRICS\n\n"
            f"MSE: {metrics['mse']:.2f}\n"
            f"SSIM (avg): {metrics['ssim']:.4f}\n\n"
            f"Max Diff: {metrics['max_difference']:.1f}\n"
            f"Mean Diff: {metrics['mean_difference']:.1f}\n\n"
            f"STATUS: {metrics['status']}"
        )
        axes[1, 2].text(
            0.05,
            0.95,
            metrics_text,
            fontsize=12,
            verticalalignment="top",
            color=status_colors.get(metrics["status"], "black"),
            family="monospace",
        )
        axes[1, 2].axis("off")

        plt.suptitle(
            f"Validation Report: {metrics['image_name']} - {metrics['colorspace']}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # ** CAMBIO: Guardar en el directorio de resultados **
        comparison_filename = (
            self.results_dir
            / f"comparison_{metrics['image_name']}_{metrics['colorspace']}_"
            f"s{metrics['scale']}.png"
        )
        plt.savefig(str(comparison_filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.logger.debug(f"Comparison saved: {comparison_filename}")

    def export_analysis_logs(self):
        """Exporta logs de análisis en el directorio de resultados."""
        if not self.results:
            self.logger.warning("No results to export.")
            return

        # ** CAMBIO: Todos los archivos se guardan en self.results_dir **

        # 1. CSV
        csv_filename = self.results_dir / f"validation_summary_{self.timestamp}.csv"
        # ... (código para escribir CSV sin cambios)
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            headers = [
                "timestamp",
                "image_name",
                "colorspace",
                "scale",
                "mse",
                "ssim",
                "status",
                "original_file",
                "imagej_file",
                "python_file",
            ]
            f.write(",".join(headers) + "\n")
            for result in self.results:
                row = [str(result.get(h, "")) for h in headers]
                f.write(",".join(row) + "\n")

        # 2. Reporte JSON detallado
        json_filename = self.results_dir / f"validation_report_{self.timestamp}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)

        self.logger.info("Analysis logs exported to results directory:")
        self.logger.info(f"  - CSV Summary: {csv_filename.name}")
        self.logger.info(f"  - Detailed JSON: {json_filename.name}")
        self.logger.info(f"  - Main Log File: {self.logger.handlers[0].baseFilename}")

    # --- El método `generate_comprehensive_report` permanece sin cambios ---
    def generate_comprehensive_report(self):
        """Genera reporte completo con estadísticas detalladas."""
        # (El código de esta función permanece sin cambios)
        if not self.results:
            self.logger.warning("No results available for comprehensive report")
            return
        self.logger.info("=" * 80)
        self.logger.info("COMPREHENSIVE VALIDATION REPORT")
        self.logger.info("=" * 80)
        colorspace_stats = defaultdict(
            lambda: {
                "count": 0,
                "excellent": 0,
                "good": 0,
                "acceptable": 0,
                "needs_fix": 0,
                "mse_sum": 0,
                "ssim_sum": 0,
            }
        )
        for result in self.results:
            cs, status = result["colorspace"], result["status"].lower()
            status_key = "needs_fix" if status == "needs_adjustment" else status
            colorspace_stats[cs]["count"] += 1
            colorspace_stats[cs][status_key] += 1
            colorspace_stats[cs]["mse_sum"] += result["mse"]
            colorspace_stats[cs]["ssim_sum"] += result["ssim"]
        self.logger.info("RESULTS BY COLORSPACE:")
        self.logger.info("-" * 80)
        header = (
            f"{'Colorspace':<10} {'Tests':<6} {'Excellent':<10} {'Good':<6} "
            f"{'Accept':<7} {'Needs Fix':<10} {'Avg MSE':<8} {'Avg SSIM':<9}"
        )
        self.logger.info(header)
        self.logger.info("-" * 80)
        for cs in sorted(colorspace_stats.keys()):
            stats = colorspace_stats[cs]
            avg_mse = stats["mse_sum"] / stats["count"]
            avg_ssim = stats["ssim_sum"] / stats["count"]
            line = (
                f"{cs:<10} {stats['count']:<6} {stats['excellent']:<10} "
                f"{stats['good']:<6} {stats['acceptable']:<7} {stats['needs_fix']:<10} "
                f"{avg_mse:<8.1f} {avg_ssim:<9.3f}"
            )
            self.logger.info(line)
        total = len(self.results)
        excellent = sum(1 for r in self.results if r["status"] == "EXCELLENT")
        good = sum(1 for r in self.results if r["status"] == "GOOD")
        acceptable = sum(1 for r in self.results if r["status"] == "ACCEPTABLE")
        needs_fix = sum(1 for r in self.results if r["status"] == "NEEDS_ADJUSTMENT")
        success_rate = (excellent + good) / total * 100 if total > 0 else 0
        self.logger.info("=" * 80)
        self.logger.info("FINAL SUMMARY:")
        self.logger.info(f"  Total validations: {total}")
        self.logger.info(
            f"  Excellent: {excellent} "
            f"({excellent / total * 100 if total > 0 else 0:.1f}%)"
        )
        self.logger.info(
            f"  Good: {good} ({good / total * 100 if total > 0 else 0:.1f}%)"
        )
        self.logger.info(
            f"  Acceptable: {acceptable} "
            f"({acceptable / total * 100 if total > 0 else 0:.1f}%)"
        )
        self.logger.info(
            f"  Needs Adjustment: {needs_fix} "
            f"({needs_fix / total * 100 if total > 0 else 0:.1f}%)"
        )
        self.logger.info(f"  SUCCESS RATE (Excellent + Good): {success_rate:.1f}%")
        if success_rate >= 80:
            self.logger.info("  🎉 VALIDATION SUCCESSFUL")
        elif success_rate >= 60:
            self.logger.info("  ⚠️ PARTIAL VALIDATION")
        else:
            self.logger.info("  ❌ REQUIRES SIGNIFICANT ADJUSTMENTS")


if __name__ == "__main__":
    # El directorio con las imágenes de validación se pasa aquí.
    # Puede ser una ruta relativa o absoluta.
    validation_directory = "validation_images"

    validator = SmartValidator(output_base_dir="validation_results")
    validator.validate_all_discovered(directory=validation_directory)
