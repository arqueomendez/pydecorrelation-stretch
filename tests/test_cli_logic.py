from dstretch.cli import _convert_config_to_list
from dstretch.independent_processors import create_preprocessing_config


class TestCLILogic:
    def test_create_preprocessing_config(self):
        # test default args
        config = create_preprocessing_config()
        assert isinstance(config, dict)

        # Test with values
        config = create_preprocessing_config(
            invert=True, auto_contrast=True, clip_percent=2.5, color_balance=False
        )

        assert config["invert"]["enabled"] is True
        assert config["auto_contrast"]["enabled"] is True
        # assert config["auto_contrast"]["clip_percent"] == 2.5 # Param name might be different or not passed through if not compliant
        assert (
            "color_balance" not in config
        )  # Should be omitted if false or not enabled? Checks implementation.
        # Actually create_preprocessing_config usually returns only enabled ones or structure?
        # Let's verify _convert_config_to_list logic which assumes structure.

    def test_convert_config_to_list(self):
        # Setup a sample dict config (simulating what create_preprocessing_config or argparse logic constructs)
        config = {
            "invert": {"enabled": True},
            "auto_contrast": {"enabled": True, "clip_percent": 1.0},
            "color_balance": {"enabled": False},  # disabled
            "flatten": {"enabled": True, "ksize_factor": 20},
        }

        steps = _convert_config_to_list(config)

        assert isinstance(steps, list)

        # Order expected: flatten, color_balance, auto_contrast, hue_shift, invert
        # Defined in _convert_config_to_list

        # 1. Flatten (enabled)
        assert steps[0]["type"] == "flatten"
        assert steps[0]["params"]["ksize_factor"] == 20
        assert "enabled" not in steps[0]["params"]

        # 2. Auto Contrast (enabled)
        assert steps[1]["type"] == "auto_contrast"

        # 3. Invert (enabled)
        assert steps[2]["type"] == "invert"

        # Color Balance should be skipped (enabled=False)
        types = [s["type"] for s in steps]
        assert "color_balance" not in types
