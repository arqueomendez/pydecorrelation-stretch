import numpy as np

from pydecorrelation_stretch.pipeline import DStretchPipeline


class TestDStretchPipeline:
    def setup_method(self):
        self.image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        self.pipeline = DStretchPipeline()

    def test_initialization(self):
        assert self.pipeline is not None
        assert self.pipeline.decorrelation is not None

    def test_apply_preprocessing_only(self):
        # Config as a list of dictionaries (correct format)
        config = [
            {"type": "invert", "params": {}},
            {"type": "auto_contrast", "params": {"clip_percent": 1.0}},
        ]

        processed_image, results = self.pipeline.apply_preprocessing_only(
            self.image, config
        )

        assert processed_image.shape == self.image.shape
        assert len(results) == 2
        assert results[0].processor_type == "Invert"
        assert results[1].processor_type == "Auto Contrast"

        # Verify changes
        # Invert -> Auto Contrast
        # Should be different from original
        assert not np.array_equal(processed_image, self.image)

    def test_process_complete(self):
        config = [{"type": "invert", "params": {}}]

        result = self.pipeline.process_complete(
            self.image, preprocessing_steps=config, colorspace="YDS", scale=10.0
        )

        assert result.final_image is not None
        assert result.decorrelation_result.colorspace == "YDS"
        assert len(result.preprocessing_results) == 1
        assert result.preprocessing_results[0].processor_type == "Invert"

    def test_convenience_methods(self):
        # Test direct access methods
        inverted = self.pipeline.invert(self.image)
        contrast = self.pipeline.auto_contrast(self.image)
        balanced = self.pipeline.color_balance(self.image)
        flattened = self.pipeline.flatten(self.image)

        assert inverted.shape == self.image.shape
        assert contrast.shape == self.image.shape
        assert balanced.shape == self.image.shape
        assert flattened.shape == self.image.shape
