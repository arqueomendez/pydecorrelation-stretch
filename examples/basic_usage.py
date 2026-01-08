"""
Basic usage examples for DStretch Python.

Demonstrates the main functionality and typical workflows.

Inspirado y basado en el plugin DStretch original de Jon Harman (ImageJ).

Autor principal: Víctor Méndez
Asistido por: Claude Sonnet 4, Gemini 2.5 Pro, Copilot con GPT-4.1
"""

import numpy as np
from PIL import Image

from pydecorrelation_stretch import DecorrelationStretch, get_available_colorspaces
from pydecorrelation_stretch.decorrelation import process_image


def example_basic_processing():
    """Basic image processing example."""
    print("=== Basic DStretch Processing ===")

    # Create or load a test image
    # For this example, we'll create a synthetic image
    test_image = create_test_image()

    # Initialize DStretch
    dstretch = DecorrelationStretch()

    # Process with default parameters
    result = dstretch.process(test_image, colorspace="YDS", scale=15.0)

    print(f"Processed image shape: {result.processed_image.shape}")
    print(f"Colorspace used: {result.colorspace}")
    print(f"Scale factor: {result.scale}")

    # Save result
    result_pil = Image.fromarray(result.processed_image)
    result_pil.save("example_output_yds.jpg")
    print("Saved: example_output_yds.jpg")

    return result


def example_colorspace_comparison():
    """Compare different colorspaces on the same image."""
    print("\n=== Colorspace Comparison ===")

    test_image = create_test_image()
    dstretch = DecorrelationStretch()

    # Get available colorspaces
    available = get_available_colorspaces()
    print(f"Available colorspaces: {list(available.keys())}")

    # Process with different colorspaces
    colorspaces_to_test = ["RGB", "LAB", "YDS", "CRGB", "LDS", "LRE"]

    for cs in colorspaces_to_test:
        if cs in available:
            result = dstretch.process(test_image, colorspace=cs, scale=20.0)

            # Save result
            result_pil = Image.fromarray(result.processed_image)
            result_pil.save(f"example_output_{cs.lower()}.jpg")

            print(f"Processed with {cs}: saved example_output_{cs.lower()}.jpg")


def example_scale_variation():
    """Demonstrate effect of different scale values."""
    print("\n=== Scale Variation Example ===")

    test_image = create_test_image()
    dstretch = DecorrelationStretch()

    scale_values = [5, 15, 30, 50, 80]

    for scale in scale_values:
        result = dstretch.process(test_image, colorspace="YDS", scale=float(scale))

        # Save result
        result_pil = Image.fromarray(result.processed_image)
        result_pil.save(f"example_scale_{scale}.jpg")

        print(f"Scale {scale}: saved example_scale_{scale}.jpg")


def example_file_processing():
    """Example of processing an actual image file."""
    print("\n=== File Processing Example ===")

    # This assumes you have a test image file
    input_file = "test_image.jpg"  # Replace with actual file

    try:
        # Using the convenience function
        result = process_image(
            input_file,
            colorspace="CRGB",  # Good for red pigments
            scale=25.0,
            output_path="enhanced_rock_art.jpg",
        )

        print(f"Successfully processed {input_file}")
        print("Output saved as: enhanced_rock_art.jpg")
        print(f"Original size: {result.original_image.shape}")
        print(f"Processed size: {result.processed_image.shape}")

    except Exception as e:
        print(f"Could not process file (this is expected if file doesn't exist): {e}")


def create_test_image():
    """Create a synthetic test image with some color variation."""
    # Create an image with different colored regions
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Background
    image[:, :] = [120, 100, 80]  # Brownish background (like rock)

    # Add some "pigment" areas
    # Red area (simulating red pigment)
    image[50:100, 50:100] = [150, 80, 70]

    # Yellow area (simulating yellow pigment)
    image[120:170, 50:100] = [140, 130, 80]

    # Black area (simulating black pigment)
    image[50:100, 120:170] = [60, 50, 40]

    # Add some noise to make it more realistic
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

    # Save test image
    test_pil = Image.fromarray(image)
    test_pil.save("synthetic_test_image.jpg")
    print("Created synthetic test image: synthetic_test_image.jpg")

    return image


def example_advanced_usage():
    """Advanced usage with selection masks and custom parameters."""
    print("\n=== Advanced Usage Example ===")

    test_image = create_test_image()
    dstretch = DecorrelationStretch()

    # Create a selection mask (only analyze part of the image)
    mask = np.zeros((200, 200), dtype=bool)
    mask[60:90, 60:90] = True  # Only analyze the red pigment area

    # Process using selection mask
    result = dstretch.process(
        test_image,
        colorspace="LRE",  # Good for reds
        scale=35.0,
        selection_mask=mask,
    )

    # Save result
    result_pil = Image.fromarray(result.processed_image)
    result_pil.save("example_advanced_selection.jpg")

    print("Advanced processing with selection mask completed")
    print("Saved: example_advanced_selection.jpg")

    # Reset to original
    original = dstretch.reset_to_original()
    if original is not None:
        original_pil = Image.fromarray(original)
        original_pil.save("example_reset_original.jpg")
        print("Reset to original: example_reset_original.jpg")


if __name__ == "__main__":
    print("DStretch Python - Usage Examples")
    print("=" * 40)

    # Run all examples
    example_basic_processing()
    example_colorspace_comparison()
    example_scale_variation()
    example_file_processing()
    example_advanced_usage()

    print("\n" + "=" * 40)
    print("All examples completed!")
    print("Check the generated image files to see the results.")
