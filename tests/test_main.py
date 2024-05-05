import os
import tempfile
from unittest import mock

import pytest
from dotenv import load_dotenv
from PIL import Image


@pytest.fixture(scope="session", autouse=True)
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), "mock.env")
    load_dotenv(dotenv_path=env_path)


from deepaipng.api import (
    is_valid_uuid_v4,
    raise_for_apikey,
    remove_background,
    resize_and_center_crop,
)


def test_api_key_from_env(mocker):
    mocker.patch("os.getenv", return_value="16b6c570-91f8-474d-8afd-756d3c48148d")
    api_key = raise_for_apikey()
    assert api_key == "16b6c570-91f8-474d-8afd-756d3c48148d"
    assert is_valid_uuid_v4(api_key) is True


def test_missing_api_key_from_env(mocker):
    mocker.patch("os.getenv", return_value=None)
    with pytest.raises(ValueError) as excinfo:
        raise_for_apikey()
    assert "The DeepAI API key was not found" in str(excinfo.value)


def test_invalid_api_key_from_env(mocker):
    mocker.patch("os.getenv", return_value="invalid-api-key")
    with pytest.raises(ValueError) as excinfo:
        raise_for_apikey()
    assert "not a valid UUID version 4" in str(excinfo.value)


def test_raise_for_apikey_with_valid_key():
    api_key = raise_for_apikey()
    assert api_key == "16b6c570-91f8-474d-8afd-756d3c48148d"


def test_is_valid_uuid_v4():
    valid_uuid = "16b6c570-91f8-474d-8afd-756d3c48148d"
    invalid_uuid = "invalid-uuid-426b-8960-example"
    assert is_valid_uuid_v4(valid_uuid) is True
    assert is_valid_uuid_v4(invalid_uuid) is False


def test_raise_for_apikey_with_invalid_key():
    with pytest.raises(ValueError):
        raise_for_apikey("invalid-uuid-d34d-b33f-0badc0ffee")


def test_resize_and_center_crop():
    img = Image.new("RGB", (200, 200), color="red")
    temp_file = tempfile.mktemp(suffix=".png")
    img.save(temp_file, format="PNG")
    output = resize_and_center_crop(temp_file, 100)
    assert output.size == (100, 100)
    os.remove(temp_file)


@mock.patch("builtins.open", mock.mock_open())
@mock.patch("httpx.post")
def test_remove_background(mock_post):
    mock_response = mock.Mock()
    api_key = "16b6c570-91f8-474d-8afd-756d3c48148d"
    mock_response.json.return_value = {"output_url": "http://deepai.org/null.png"}
    mock_response.raise_for_status = mock.Mock()
    mock_post.return_value = mock_response

    with mock.patch("httpx.get") as mock_get:
        mock_get.return_value = mock.Mock()
        mock_get.return_value.content = b"image data"
        result = remove_background("path/to/image.png", api_key)
        assert result == b"image data"
