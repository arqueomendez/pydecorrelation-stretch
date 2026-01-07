# DStretch Python

[![DOI](https://zenodo.org/badge/1047210737.svg)](https://doi.org/10.5281/zenodo.17172811)

**DStretch Python** es una implementación avanzada y validada del algoritmo decorrelation stretch para el realce de imágenes arqueológicas, especialmente arte rupestre, en Python. Replica matemáticamente el plugin DStretch de ImageJ (Jon Harman), con arquitectura moderna, interfaces CLI/GUI y 23 espacios de color validados.

## Motivación

DStretch Python permite revelar pigmentos y detalles invisibles en fotografías de arte rupestre y otros contextos arqueológicos. Utiliza análisis estadístico de color (PCA decorrelation stretch) y matrices predefinidas para separar y realzar colores de interés, facilitando la documentación, análisis y preservación digital.

## Características

- **Replicación exacta** del algoritmo DStretch ImageJ (validación >99.97% SSIM)
- **23 espacios de color** (YDS, CRGB, LRE, LAB, RGB, etc.)
- **Soporte para Imágenes Masivas**: Procesamiento "Auto-Memmap" inteligente.
- **Nueva Interfaz "Premium"**: GUI moderna con tema oscuro (CustomTkinter).
- **Interfaces CLI y GUI** multiplataforma (Windows, macOS, Linux)
- **API de Python** para scripting y procesamiento por lotes.
- **Optimización para imágenes grandes** con uso eficiente de memoria.
- **Arquitectura extensible** para fácil integración en pipelines científicos.
- **Documentación completa** técnica y de uso.

## Formatos de Imagen Soportados

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- TIFF (`.tif`, `.tiff`)
- WebP (`.webp`)

## Instalación

Instalar ambiente [uv](https://docs.astral.sh/uv/getting-started/installation/) según sistema operativo: 

- Windows:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- macOS y linux
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Descargar repositorio:

```bash
git clone https://github.com/arqueomendez/dstretch-python.git
```

Instalar Decorrelation Stretch:
Con UV:
```bash
cd dstretch-python
uv sync
```

O usando pip:
```bash
uv pip install -r requirements
```

## Descarga de Ejecutable (Windows)

Para usuarios de **Windows** que prefieren no usar la línea de comandos ni instalar Python, ofrecemos un **ejecutable autot contenido (.exe)**.

1.  Ve a la sección de **[Releases](https://github.com/arqueomendez/dstretch-python/releases)** del repositorio.
2.  Descarga el archivo `DStretch-GUI-vX.Y.Z.exe` de la última versión.
3.  Ejecuta el archivo directamente para iniciar la interfaz gráfica (GUI).

Este ejecutable contiene todo lo necesario para funcionar y ofrece la misma experiencia que la versión instalada manualmente.

## Tutorial en Video 📹

Para una explicación completa y visual del proyecto DStretch Python, incluyendo instalación, uso y ejemplos prácticos, consulta este video tutorial en español:

[![DStretch Python - Tutorial Completo](https://img.youtube.com/vi/0bSW_uju6TI/0.jpg)](https://www.youtube.com/watch?v=0bSW_uju6TI)

**🎥 [Ver tutorial en YouTube: DStretch Python - Explicación completa en español](https://www.youtube.com/watch?v=0bSW_uju6TI)**

## Uso Rápido

El proyecto se puede utilizar de dos maneras: a través de una Interfaz de Línea de Comandos (CLI) para scripting y procesamiento por lotes, o mediante una Interfaz Gráfica de Usuario (GUI) para análisis visual e interactivo.

Para asegurar que los comandos se ejecuten correctamente dentro del entorno virtual gestionado por `uv`, se recomienda usar `uv run`. Esto evita conflictos con otras instalaciones de Python o paquetes en el sistema.

### CLI
La CLI es ideal para procesar imágenes de forma automática o integrar DStretch en flujos de trabajo existentes.

```bash
# Procesamiento básico (espacio YDS por defecto)
dstretch input.jpg
# Especificar espacio de color e intensidad
dstretch input.jpg --colorspace CRGB --scale 25
# Guardar en archivo específico
dstretch input.jpg --colorspace LRE --scale 30 --output enhanced.jpg
# Procesamiento básico
uv run dstretch input.jpg

# Especificar espacio de color, intensidad y archivo de salida
uv run dstretch input.jpg --colorspace CRGB --scale 25 --output enhanced.jpg

# Listar espacios disponibles
dstretch --list-colorspaces
uv run dstretch --list-colorspaces
```

### GUI
```bash
uv run dstretch-gui
# Interfaz gráfica similar a ImageJ DStretch
```

### Python API
```python
from dstretch import DecorrelationStretch
import cv2
image = cv2.imread("input.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
dstretch = DecorrelationStretch()
result = dstretch.process(image, colorspace="YDS", scale=15.0)
enhanced = result.processed_image
cv2.imwrite("output.jpg", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
```

## Espacios de Color Disponibles

- **Estándar:** RGB, LAB
- **Serie Y (YUV):** YDS, YBR, YBK, YRE, YRD, YWE, YBL, YBG, YUV, YYE
- **Serie L (LAB):** LAX, LDS, LRE, LRD, LBK, LBL, LWE, LYE
- **Predefinidos:** CRGB, RGB0, LABI

## Ejemplos de Aplicación

- **Documentación de pictografías:** realce de pigmentos rojos, amarillos, negros y blancos
- **Análisis de pigmentos:** separación de minerales y composición
- **Registro de sitios:** mejora de fotografías para informes y publicaciones
- **Investigación:** revelado de arte rupestre invisible

### Recomendaciones por Pigmento

- Rojo ocre/hematita: CRGB, LRE, YBR
- Amarillo ocre: YDS, LDS, YYE
- Carbón negro: YBK, LBK
- Caolín blanco: YWE, LWE
- Realce general: YDS, LDS, LAB

## Validación y Precisión

- Validación pixel a pixel contra DStretch ImageJ (SSIM promedio >0.9997)
- 40/40 tests EXCELLENT en suite automática
- 23/23 espacios de color implementados y validados
- Métricas: SSIM, MSE, diferencias por canal

## Arquitectura y Detalles Técnicos

- Núcleo matemático desacoplado de interfaces
- Procesadores independientes (preprocesamiento: flatten, auto-contraste, balance de color, etc.)
- Pipeline configurable y extensible
- Optimización: LUTs precomputadas, procesamiento por chunks, threading en GUI
- Documentación inline y en `docs/`

## Estructura del Proyecto

```
dstretch_python/
├── src/dstretch/           # Paquete principal
│   ├── decorrelation.py    # Algoritmo principal
│   ├── colorspaces.py      # Transformaciones de color
│   ├── cli.py              # Interfaz CLI
│   ├── gui.py              # Interfaz gráfica
│   └── ...
├── tests/                 # Pruebas automáticas
├── examples/              # Ejemplos de uso
├── docs/                  # Documentación técnica
└── validation_results/    # Resultados de validación
```

## Contribución

¡Contribuciones bienvenidas! Puedes ayudar en:
- Nuevos espacios de color o procesadores
- Optimización de rendimiento
- Mejoras en la GUI
- Documentación y tutoriales
- Casos de estudio arqueológicos

## Licencia

Este proyecto está licenciado bajo **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

- Puedes usar, modificar y compartir el software libremente **solo para fines no comerciales**.
- Es obligatorio citar el proyecto y a los autores originales.
- Para uso comercial, contacta a los autores.

Ver el archivo LICENSE para detalles completos.

## Cita y Autores

Si utilizas DStretch Python en trabajos académicos, por favor cita el software utilizando su DOI de Zenodo, que garantiza una referencia permanente y rastreable.

Formato de cita sugerido (APA 7 para software):

> Méndez, V. (2025). *DStretch Python* (Version 0.5.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17172811

### Autores y Agradecimientos

- **Autor principal:** Víctor Méndez
- **Asistido por:** Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1
- **Agradecimientos:** A Jon Harman por crear el plugin DStretch original para ImageJ y a la comunidad arqueológica por su retroalimentación.

> Harmand, J. (2008). Using Decorrelation Stretch to Enhance Rock Art Images. American Rock Art Research Association Annual Meeting. American Rock Art Research Association Annual Meeting. 15-12-2024. https://www.dstretch.com/AlgorithmDescription.html

## Soporte y Comunidad

- Issues: Reporta bugs y solicita mejoras en GitHub Issues
- Documentación: Consulta la documentación técnica en `docs/`
- Comunidad: Únete a foros y discusiones científicas

---