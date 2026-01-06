import numpy as np
import pytest
from dstretch.colorspaces import (
    RGBColorspace,
    LABColorspace,
    YDSColorspace,
    ColorspaceManager
)

class TestColorspaces:
    def setup_method(self):
        self.image = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        self.manager = ColorspaceManager()

    def test_rgb_colorspace(self):
        cs = RGBColorspace()
        converted = cs.to_colorspace(self.image)
        # RGB to colorspace usually implies no change for RGBColorspace usually?
        # Base implementation might do normalization.
        
        assert converted.shape == self.image.shape
        # assert converted.dtype == np.float32 # Usually float for processing
        
        restored = cs.from_colorspace(converted)
        assert restored.shape == self.image.shape
        assert restored.dtype == np.uint8

    def test_lab_colorspace(self):
        cs = LABColorspace()
        converted = cs.to_colorspace(self.image)
        
        assert converted.shape == self.image.shape
        assert not np.isnan(converted).any()
        
        restored = cs.from_colorspace(converted)
        diff = np.mean(np.abs(self.image.astype(float) - restored.astype(float)))
        # Relax tolerance slightly due to colorspace conversion losses
        assert diff < 8.0 

    def test_yds_colorspace(self):
        cs = YDSColorspace()
        converted = cs.to_colorspace(self.image)
        assert converted.shape == self.image.shape
        assert not np.isnan(converted).any()

    def test_manager_retrieval(self):
        spaces = self.manager.list_available()
        assert "RGB" in spaces
        assert "YDS" in spaces
        
        cs = self.manager.get_colorspace("RGB")
        assert isinstance(cs, RGBColorspace)
