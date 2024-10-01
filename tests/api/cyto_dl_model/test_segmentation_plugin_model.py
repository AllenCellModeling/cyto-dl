from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cyto_dl.api.cyto_dl_model import SegmentationPluginModel


# Taken together, these tests verify that the SegmentationPluginModel aligns with the
# default segmentation_plugin config file. It does this by a) verifying that methods which
# attempt to change config items on a bad (fake) config throw exceptions because those config
# items do not exist and b) verifying that those methods do not throw exceptions when called on
# a model that uses the default config file
#
# Note that these tests do not test whether or not the config variables set by SegmentationPluginModel
# are actually used how we expect by Cyto-DL, just that they exist in the default config.
@pytest.fixture
def model_with_default_config() -> SegmentationPluginModel:
    return SegmentationPluginModel.from_default_config(3)


@pytest.fixture
def model_with_bad_config() -> SegmentationPluginModel:
    return SegmentationPluginModel.from_existing_config(Path(__file__).parent / "bad_config.yaml")


class TestDefaultConfig:
    @patch("cyto_dl.api.cyto_dl_model.cyto_dl_base_model.train_model")
    def test_train_no_ckpt(
        self, train_model_mock: Mock, model_with_default_config: SegmentationPluginModel
    ):
        model_with_default_config.train(1, "manifest", "output_dir")
        train_model_mock.assert_called_once()

    @patch("cyto_dl.api.cyto_dl_model.cyto_dl_base_model.train_model")
    def test_train_with_ckpt(
        self, train_model_mock: Mock, model_with_default_config: SegmentationPluginModel
    ):
        model_with_default_config.train(1, "manifest", "output_dir", checkpoint=Path("ckpt"))
        train_model_mock.assert_called_once()

    @patch("cyto_dl.api.cyto_dl_model.cyto_dl_base_model.evaluate_model")
    def test_predict(
        self, evaluate_model_mock: Mock, model_with_default_config: SegmentationPluginModel
    ):
        model_with_default_config.predict("manifest", "output_dir", Path("ckpt"))
        evaluate_model_mock.assert_called_once()

    def test_input_channel(self, model_with_default_config: SegmentationPluginModel):
        assert model_with_default_config.get_input_channel() is not None
        model_with_default_config.set_input_channel(785)
        assert model_with_default_config.get_input_channel() == 785

    def test_raw_image_channels(self, model_with_default_config: SegmentationPluginModel):
        assert model_with_default_config.get_raw_image_channels() is not None
        model_with_default_config.set_raw_image_channels(35)
        assert model_with_default_config.get_raw_image_channels() == 35

    def test_manifest_column_names(self, model_with_default_config: SegmentationPluginModel):
        assert len(model_with_default_config.get_manifest_column_names()) == 6
        model_with_default_config.set_manifest_column_names("a", "b", "c", "d", "e", "f")
        assert model_with_default_config.get_manifest_column_names() == (
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
        )

    def test_split_column(self, model_with_default_config: SegmentationPluginModel):
        assert model_with_default_config.get_split_column() is None
        model_with_default_config.set_split_column("foo")
        assert model_with_default_config.get_split_column() == "foo"
        model_with_default_config.remove_split_column()
        assert model_with_default_config.get_split_column() is None


class TestBadConfig:
    @patch("cyto_dl.api.cyto_dl_model.cyto_dl_base_model.train_model")
    def test_train_no_ckpt(
        self, train_model_mock: Mock, model_with_bad_config: SegmentationPluginModel
    ):
        with pytest.raises(KeyError):
            model_with_bad_config.train(1, "manifest", "output_dir")

    @patch("cyto_dl.api.cyto_dl_model.cyto_dl_base_model.train_model")
    def test_train_with_ckpt(
        self, train_model_mock: Mock, model_with_bad_config: SegmentationPluginModel
    ):
        with pytest.raises(KeyError):
            model_with_bad_config.train(1, "manifest", "output_dir", checkpoint=Path("ckpt"))

    @patch("cyto_dl.api.cyto_dl_model.cyto_dl_base_model.evaluate_model")
    def test_predict(
        self, evaluate_model_mock: Mock, model_with_bad_config: SegmentationPluginModel
    ):
        with pytest.raises(KeyError):
            model_with_bad_config.predict("manifest", "output_dir", Path("ckpt"))

    def test_input_channel(self, model_with_bad_config: SegmentationPluginModel):
        with pytest.raises(KeyError):
            model_with_bad_config.get_input_channel()
        with pytest.raises(KeyError):
            model_with_bad_config.set_input_channel(785)

    def test_raw_image_channels(self, model_with_bad_config: SegmentationPluginModel):
        with pytest.raises(KeyError):
            model_with_bad_config.get_raw_image_channels()
        with pytest.raises(KeyError):
            model_with_bad_config.set_raw_image_channels(35)

    def test_manifest_column_names(self, model_with_bad_config: SegmentationPluginModel):
        with pytest.raises(KeyError):
            model_with_bad_config.get_manifest_column_names()
        with pytest.raises(KeyError):
            model_with_bad_config.set_manifest_column_names("a", "b", "c", "d", "e", "f")

    def test_split_column(self, model_with_bad_config: SegmentationPluginModel):
        with pytest.raises(KeyError):
            model_with_bad_config.get_split_column()
        with pytest.raises(KeyError):
            model_with_bad_config.set_split_column("foo")
