"""
DStretch Python - Decorrelation Stretch para análisis de arte rupestre y arqueología

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1

Versión 0.0.2: Arquitectura de pipeline independiente
"""

__version__ = "0.0.2"
__author__ = (
    "Víctor Méndez, asistido por Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1"
)

# Core decorrelation algorithm (legacy)
# Core decorrelation algorithm (legacy)
from .decorrelation import DecorrelationStretch as DecorrelationStretch
from .decorrelation import ProcessingResult as ProcessingResult

# Independent processors (new architecture)
from .independent_processors import (
    AutoContrastProcessor as AutoContrastProcessor,
)
from .independent_processors import (
    BaseProcessor as BaseProcessor,
)
from .independent_processors import (
    ColorBalanceProcessor as ColorBalanceProcessor,
)
from .independent_processors import (
    FlattenProcessor as FlattenProcessor,
)
from .independent_processors import (
    HueShiftProcessor as HueShiftProcessor,
)
from .independent_processors import (
    InvertProcessor as InvertProcessor,
)
from .independent_processors import (
    PreprocessingPipeline as PreprocessingPipeline,
)
from .independent_processors import (
    ProcessingResult as ProcessorResult,
)
from .independent_processors import (
    ProcessorFactory as ProcessorFactory,
)
from .independent_processors import (
    ProcessorType as ProcessorType,
)
from .independent_processors import (
    create_preprocessing_config as create_preprocessing_config,
)
from .independent_processors import (
    quick_enhance as quick_enhance,
)

# New pipeline architecture
try:
    from .pipeline import (
        CompletePipelineResult as CompletePipelineResult,
    )
    from .pipeline import (
        DStretchPipeline as DStretchPipeline,
    )
    from .pipeline import (
        process_image_legacy as process_image_legacy,
    )
    from .pipeline import (
        quick_enhance as quick_enhance_pipeline,
    )

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    DStretchPipeline = None
    quick_enhance_pipeline = None

# Colorspaces
# Colorspaces
from .colorspaces import COLORSPACES as COLORSPACES
from .colorspaces import ColorspaceManager as ColorspaceManager

# GUI components (if available)
try:
    from .gui_infrastructure import (
        AdvancedStatusBar as AdvancedStatusBar,
    )
    from .gui_infrastructure import (
        ErrorManager as ErrorManager,
    )
    from .gui_infrastructure import (
        GUIInfrastructure as GUIInfrastructure,
    )
    from .gui_infrastructure import (
        PerformanceManager as PerformanceManager,
    )
    from .gui_infrastructure import (
        ThreadManager as ThreadManager,
    )
    from .gui_infrastructure import (
        TooltipManager as TooltipManager,
    )
    from .pixel_inspector import (
        ColorSpaceConverter as ColorSpaceConverter,
    )
    from .pixel_inspector import (
        PixelAnalyzer as PixelAnalyzer,
    )
    from .pixel_inspector import (
        PixelInspectorPanel as PixelInspectorPanel,
    )
    from .zoom_pan_controller import (
        ZoomPanController as ZoomPanController,
    )
    from .zoom_pan_controller import (
        ZoomToolbar as ZoomToolbar,
    )

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


# Helper functions
def list_available_colorspaces():
    """Returns a list of available colorspace names."""
    return list(COLORSPACES.keys())


def get_available_processors():
    """Get dictionary of available processors and their descriptions."""
    pipeline = PreprocessingPipeline()
    info = pipeline.get_all_processors_info()
    return {v["name"]: v["description"] for k, v in info.items()}


def get_pipeline_info():
    """Get information about the pipeline architecture."""
    return {
        "version": __version__,
        "architecture": "Independent Pipeline",
        "available_colorspaces": len(COLORSPACES),
        "available_processors": len(get_available_processors()),
        "gui_available": GUI_AVAILABLE,
        "pipeline_available": PIPELINE_AVAILABLE,
    }


# Main factory function for new architecture
def create_preprocessing_pipeline():
    """
    Create a new preprocessing pipeline instance.

    Returns:
        PreprocessingPipeline: New pipeline instance with corrected architecture
    """
    return PreprocessingPipeline()


def create_dstretch_pipeline():
    """
    Create a new DStretch pipeline instance (if available).

    Returns:
        DStretchPipeline: New pipeline instance with corrected architecture
    """
    if PIPELINE_AVAILABLE:
        from .pipeline import DStretchPipeline

        return DStretchPipeline()
    else:
        raise ImportError(
            "DStretchPipeline not available. Use create_preprocessing_pipeline() instead."
        )


# Legacy compatibility function
def create_decorrelation_stretch():
    """
    Create legacy DecorrelationStretch instance for backward compatibility.

    Returns:
        DecorrelationStretch: Legacy instance (deprecated)
    """
    return DecorrelationStretch()


# Quick processing functions
def process_image(image, preprocessing_steps=None, colorspace="YDS", scale=15.0):
    """
    Quick image processing using new pipeline architecture.

    Args:
        image: Input RGB image (numpy array)
        preprocessing_steps: List of preprocessing configurations (optional)
        colorspace: Colorspace for decorrelation stretch
        scale: Scale factor for decorrelation stretch

    Returns:
        Processed image or tuple with results
    """
    if PIPELINE_AVAILABLE and DStretchPipeline:
        pipeline = DStretchPipeline()
        return pipeline.process_complete(image, preprocessing_steps, colorspace, scale)
    else:
        # Fallback to basic processing
        if preprocessing_steps:
            preprocessing_pipeline = PreprocessingPipeline()
            preprocessed_image, _ = preprocessing_pipeline.process(
                image, preprocessing_steps
            )
        else:
            preprocessed_image = image

        # Apply decorrelation stretch
        dstretch = DecorrelationStretch()
        result = dstretch.process(preprocessed_image, colorspace, scale)
        return result.processed_image


def process_with_preset(image, preset="standard", colorspace=None, scale=None):
    """
    Process image with enhancement preset.

    Args:
        image: Input RGB image
        preset: Enhancement preset ('standard', 'faint_reds', 'yellows', 'high_contrast')
        colorspace: Override colorspace (optional)
        scale: Override scale (optional)

    Returns:
        Processed image
    """
    if PIPELINE_AVAILABLE and quick_enhance_pipeline:
        return quick_enhance_pipeline(image, preset, colorspace or "YDS", scale or 15.0)
    else:
        # Fallback to simple enhancement
        return quick_enhance(image, preset)


# Export main classes and functions
__all__ = [
    # Core classes
    "DecorrelationStretch",
    "ProcessingResult",
    # Processors
    "BaseProcessor",
    "InvertProcessor",
    "AutoContrastProcessor",
    "ColorBalanceProcessor",
    "FlattenProcessor",
    "HueShiftProcessor",
    "ProcessorFactory",
    "PreprocessingPipeline",
    "ProcessorType",
    "ProcessorResult",
    # Pipeline functions
    "create_preprocessing_config",
    "quick_enhance",
    "create_preprocessing_pipeline",
    "create_dstretch_pipeline",
    "process_image",
    "process_with_preset",
    # Utilities
    "list_available_colorspaces",
    "get_available_processors",
    "get_pipeline_info",
    "COLORSPACES",
    "ColorspaceManager",
    # Legacy compatibility
    "create_decorrelation_stretch",
]
