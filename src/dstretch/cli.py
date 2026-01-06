"""
Command line interface for DStretch Python - Version 2.0
Independent Pipeline Architecture

Provides a CLI that supports the new preprocessing pipeline where tools
operate on RGB images BEFORE decorrelation stretch.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

from . import (
    DStretchPipeline,
    create_preprocessing_config,
    get_available_processors,
    get_pipeline_info,
    list_available_colorspaces,
    get_pipeline_info,
    list_available_colorspaces,
    process_with_preset,
)


def _convert_config_to_list(config: dict) -> list:
    """Convert config dict to list for pipeline compatibility."""
    steps = []
    # Order matters!
    order = ["flatten", "color_balance", "auto_contrast", "hue_shift", "invert"]
    for name in order:
        if name in config and config[name].get("enabled", False):
            step = {"type": name, "params": config[name].copy()}
            step["params"].pop("enabled", None)
            steps.append(step)
    return steps


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DStretch Python v2.0 - Independent Pipeline Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing with default YDS colorspace
  dstretch input.jpg
  
  # Specify colorspace and intensity
  dstretch input.jpg --colorspace CRGB --scale 25
  
  # Apply preprocessing before decorrelation
  dstretch input.jpg --auto-contrast --color-balance --colorspace LRE
  
  # Use enhancement preset
  dstretch input.jpg --preset faint_reds
  
  # Full preprocessing pipeline
  dstretch input.jpg --invert --flatten --auto-contrast --color-balance --colorspace YDS
  
  # Save to specific output file
  dstretch input.jpg --colorspace LRE --output enhanced.jpg
  
  # List available colorspaces and processors
  dstretch --list-colorspaces
  dstretch --list-processors
        """,
    )

    # Main arguments
    parser.add_argument("input", nargs="?", help="Input image file")

    parser.add_argument(
        "-c",
        "--colorspace",
        default="YDS",
        help="Color space for decorrelation stretch (default: YDS)",
    )

    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=15.0,
        help="Enhancement intensity (1-100, default: 15)",
    )

    parser.add_argument(
        "-o", "--output", help="Output file path (default: auto-generated)"
    )

    # Enhancement presets
    parser.add_argument(
        "--preset",
        choices=["standard", "faint_reds", "yellows", "high_contrast"],
        help="Enhancement preset (overrides individual preprocessing flags)",
    )

    # Information commands
    parser.add_argument(
        "--list-colorspaces",
        action="store_true",
        help="List all available colorspaces and exit",
    )

    parser.add_argument(
        "--list-processors",
        action="store_true",
        help="List all available processors and exit",
    )

    parser.add_argument(
        "--pipeline-info",
        action="store_true",
        help="Show pipeline architecture information and exit",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="DStretch Python 2.0.0 - Independent Pipeline Architecture",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Preprocessing flags (applied BEFORE decorrelation stretch)
    preprocessing_group = parser.add_argument_group(
        "Preprocessing Tools (Applied Before Decorrelation)"
    )

    preprocessing_group.add_argument(
        "--invert",
        action="store_true",
        help="Apply inversion BEFORE decorrelation stretch",
    )

    preprocessing_group.add_argument(
        "--invert-mode",
        choices=["full", "luminance_only", "selective"],
        default="full",
        help="Inversion mode (default: full)",
    )

    preprocessing_group.add_argument(
        "--auto-contrast",
        action="store_true",
        help="Apply auto contrast enhancement BEFORE decorrelation stretch",
    )

    preprocessing_group.add_argument(
        "--contrast-clip",
        type=float,
        default=0.1,
        help="Auto contrast clip percentage (0.0-5.0, default: 0.1)",
    )

    preprocessing_group.add_argument(
        "--color-balance",
        action="store_true",
        help="Apply color balance correction BEFORE decorrelation stretch",
    )

    preprocessing_group.add_argument(
        "--balance-method",
        choices=["gray_world", "white_patch", "manual"],
        default="gray_world",
        help="Color balance method (default: gray_world)",
    )

    preprocessing_group.add_argument(
        "--balance-strength",
        type=float,
        default=1.0,
        help="Color balance effect strength (0.0-2.0, default: 1.0)",
    )

    preprocessing_group.add_argument(
        "--temperature-offset",
        type=float,
        default=0.0,
        help="Manual temperature adjustment (-100 to +100, default: 0)",
    )

    preprocessing_group.add_argument(
        "--tint-offset",
        type=float,
        default=0.0,
        help="Manual tint adjustment (-100 to +100, default: 0)",
    )

    preprocessing_group.add_argument(
        "--flatten",
        action="store_true",
        help="Apply flatten (illumination correction) BEFORE decorrelation stretch",
    )

    preprocessing_group.add_argument(
        "--flatten-method",
        choices=[
            "bandpass_filter",
            "gaussian_background",
            "sliding_paraboloid",
            "rolling_ball",
        ],
        default="bandpass_filter",
        help="Flatten method (default: bandpass_filter)",
    )

    preprocessing_group.add_argument(
        "--filter-large",
        type=float,
        default=40.0,
        help="Large structures to remove for flatten (pixels, default: 40)",
    )

    preprocessing_group.add_argument(
        "--filter-small",
        type=float,
        default=3.0,
        help="Small structures to preserve for flatten (pixels, default: 3)",
    )

    # Advanced options
    advanced_group = parser.add_argument_group("Advanced Options")

    advanced_group.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Also save preprocessed image (before decorrelation)",
    )

    advanced_group.add_argument(
        "--preprocessing-only",
        action="store_true",
        help="Apply only preprocessing, skip decorrelation stretch",
    )

    advanced_group.add_argument(
        "--decorrelation-only",
        action="store_true",
        help="Apply only decorrelation stretch, skip preprocessing",
    )

    args = parser.parse_args()

    # Handle information commands
    if args.list_colorspaces:
        print("Available colorspaces:")
        colorspaces = list_available_colorspaces()
        for name in sorted(colorspaces):
            print(f"  {name}")
        sys.exit(0)

    if args.list_processors:
        print("Available processors:")
        processors = get_available_processors()
        for name, description in processors.items():
            print(f"  {name:<15} - {description}")
        sys.exit(0)

    if args.pipeline_info:
        info = get_pipeline_info()
        print("DStretch Pipeline Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        sys.exit(0)

    # Validate input file is provided
    if not args.input:
        parser.error("Input image file is required")

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
        sys.exit(1)

    # Validate colorspace
    available_colorspaces = list_available_colorspaces()
    if args.colorspace not in available_colorspaces:
        print(f"Error: Unknown colorspace '{args.colorspace}'", file=sys.stderr)
        print(
            f"Available colorspaces: {sorted(available_colorspaces)}", file=sys.stderr
        )
        sys.exit(1)

    # Validate scale
    if not 1.0 <= args.scale <= 100.0:
        print(
            f"Error: Scale must be between 1 and 100, got {args.scale}", file=sys.stderr
        )
        sys.exit(1)

    # Validate mutually exclusive options
    if args.preprocessing_only and args.decorrelation_only:
        parser.error(
            "--preprocessing-only and --decorrelation-only are mutually exclusive"
        )

    if args.preset and (
        args.invert or args.auto_contrast or args.color_balance or args.flatten
    ):
        parser.error("--preset cannot be used with individual preprocessing flags")

    # Generate output path if not provided
    if not args.output:
        stem = input_path.stem
        suffix = input_path.suffix if input_path.suffix else ".jpg"

        if args.preprocessing_only:
            args.output = f"{stem}_preprocessed{suffix}"
        elif args.preset:
            args.output = (
                f"{stem}_{args.preset}_{args.colorspace}_s{int(args.scale)}{suffix}"
            )
        else:
            args.output = f"{stem}_{args.colorspace}_s{int(args.scale)}{suffix}"

    output_path = Path(args.output)

    # Verbose output
    if args.verbose:
        print("DStretch Python v2.0 - Independent Pipeline Architecture")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Colorspace: {args.colorspace}")
        print(f"Scale: {args.scale}")
        if args.preset:
            print(f"Preset: {args.preset}")
        else:
            print(
                f"Preprocessing: invert={args.invert}, auto_contrast={args.auto_contrast}, color_balance={args.color_balance}, flatten={args.flatten}"
            )

    try:
        # Load image
        if args.verbose:
            print("Loading image...")

        import cv2

        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not load image from {input_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if args.verbose:
            print(f"Image dimensions: {image.shape}")

        # Create pipeline
        pipeline = DStretchPipeline()

        if args.preset:
            # Use enhancement preset
            if args.verbose:
                print(f"Applying preset: {args.preset}")
            result = process_with_preset(image, args.preset, args.colorspace, args.scale)

        elif args.preprocessing_only:
            # Apply only preprocessing
            if args.verbose:
                print("Applying preprocessing only...")

            preprocessing_steps = create_preprocessing_config(
                invert=args.invert,
                auto_contrast=args.auto_contrast,
                color_balance=args.color_balance,
                flatten=args.flatten,
                invert_params={"invert_mode": args.invert_mode},
                auto_contrast_params={
                    "clip_percentage": args.contrast_clip,
                    "preserve_colors": True,
                },
                color_balance_params={
                    "method": args.balance_method,
                    "strength": args.balance_strength,
                    "temperature_offset": args.temperature_offset,
                    "tint_offset": args.tint_offset,
                    "preserve_luminance": True,
                },
                flatten_params={
                    "method": args.flatten_method,
                    "filter_large": args.filter_large,
                    "filter_small": args.filter_small,
                },
            )

            processed_image, processor_results = pipeline.apply_preprocessing_only(
                image, _convert_config_to_list(preprocessing_steps)
            )

            # Save preprocessed image
            cv2.imwrite(
                str(output_path), cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            )

            print(
                f"Successfully applied preprocessing to '{input_path}' -> '{output_path}'"
            )

            if args.verbose:
                print(f"Applied {len(processor_results)} preprocessing steps:")
                for result in processor_results:
                    print(f"  - {result.processor_type}: {result.parameters}")

            sys.exit(0)

        elif args.decorrelation_only:
            # Apply only decorrelation stretch
            if args.verbose:
                print("Applying decorrelation stretch only...")

            decorrelation_result = pipeline.process_decorrelation_only(
                image, args.colorspace, args.scale
            )
            decorrelation_result.save(str(output_path))

            print(
                f"Successfully applied decorrelation stretch to '{input_path}' -> '{output_path}'"
            )

            if args.verbose:
                print(f"Colorspace: {decorrelation_result.colorspace}")
                print(f"Scale: {decorrelation_result.scale}")

            sys.exit(0)

        else:
            # Full pipeline processing
            if args.verbose:
                print("Applying full pipeline...")

            preprocessing_steps = create_preprocessing_config(
                invert=args.invert,
                auto_contrast=args.auto_contrast,
                color_balance=args.color_balance,
                flatten=args.flatten,
                invert_params={"invert_mode": args.invert_mode},
                auto_contrast_params={
                    "clip_percentage": args.contrast_clip,
                    "preserve_colors": True,
                },
                color_balance_params={
                    "method": args.balance_method,
                    "strength": args.balance_strength,
                    "temperature_offset": args.temperature_offset,
                    "tint_offset": args.tint_offset,
                    "preserve_luminance": True,
                },
                flatten_params={
                    "method": args.flatten_method,
                    "filter_large": args.filter_large,
                    "filter_small": args.filter_small,
                },
            )

            result = pipeline.process_complete(
                image,
                _convert_config_to_list(preprocessing_steps),
                args.colorspace,
                args.scale,
            )

        # Save result
        if isinstance(result, np.ndarray):
            # Array result (legacy/fallback)
            cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            if args.verbose:
                 print(f"Final dimensions: {result.shape}")
        else:
            # CompletePipelineResult
            result.save_final(str(output_path))

            # Save preprocessed image if requested
            if args.save_preprocessed and result.has_preprocessing():
                preprocessed_path = output_path.with_stem(
                    f"{output_path.stem}_preprocessed"
                )
                result.save_preprocessed(str(preprocessed_path))
                if args.verbose:
                    print(f"Saved preprocessed image: {preprocessed_path}")

            if args.verbose:
                if result.has_preprocessing():
                    print(
                        f"Applied preprocessing steps: {result.get_preprocessing_names()}"
                    )
                print(f"Decorrelation colorspace: {result.decorrelation_result.colorspace}")
                print(f"Decorrelation scale: {result.decorrelation_result.scale}")
                print(f"Final dimensions: {result.final_image.shape}")
        
        print(f"Successfully processed '{input_path}' -> '{output_path}'")

    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
