# DeepAI PNG

This command-line utility processes images by resizing, cropping to fit specified dimensions, making the background transparent, and can be extended to perform other tasks. It leverages the DeepAI Background Remover API and Python's Pillow library for image manipulations.


[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/djstompzone/deepaipng/python-package-cicd.yml?label=Tests&link=https%3A%2F%2Fpypi.org%2Fproject%2Fdeepaipng%2F&logo=pytest&labelColor=090909)](https://pypi.org/project/deepaipng/) [![Python 3.10](https://img.shields.io/badge/Python-3.10-111111?link=https%3A%2F%2Fpypi.org%2Fproject%2Fdeepaipng%2F&logo=python&color=bb6633&labelColor=090909)](https://pypi.org/project/deepaipng/) [![Python 3.11](https://img.shields.io/badge/Python-3.11-111111?link=https%3A%2F%2Fpypi.org%2Fproject%2Fdeepaipng%2F&logo=python&color=dd9933&labelColor=090909)](https://pypi.org/project/deepaipng/) [![Python 3.10](https://img.shields.io/badge/Python-3.12-111111?link=https%3A%2F%2Fpypi.org%2Fproject%2Fdeepaipng%2F&logo=python&color=229933&labelColor=090909)](https://pypi.org/project/deepaipng/)

## Features

- **Resize and Crop**: Adjusts the size of the image to maintain aspect ratio and ensures it fits within a specified square dimension.
- **Background Removal**: Uses DeepAI's Background Remover API to make the image background transparent.
- **Command-Line Interface**: Offers a straightforward command-line interface for easy usage in various environments.

## Requirements

- Python 3.10 or higher

## Installation

DeepAI PNG can be installed easily with pip:

```bash
pip install deepaipng
```

You also need to obtain an API key from [DeepAI](https://deepai.org/). Set this API key in your environment as `DEEPAI_API_KEY` or pass it directly to the script if modified to accept it as an argument.

## Usage

To process an image, specify the input and output file paths, and optionally the target size. Run DeepAI PNG from the command line:

```bash
python -m deepaipng --input_path 'path/to/input.jpg' --output_path 'path/to/output.png' --size 120
```

### Arguments

- `input_path`: Path to the input image file.
- `output_path`: Path to save the output image file.
- `-s`, `--size`: Optional. Target size for the output image's width and height in pixels. Defaults to 120.

## Configuration

Set your DeepAI API key in your environment to avoid passing it directly with your scripts. 

Use the `.env` file for local development:

```plaintext
DEEPAI_API_KEY=your_api_key_here
```

Always keep your API keys secure and avoid committing them to version control. You are responsible for any unauthorized usage of your DeepAI API key.

Load this environment variable with `dotenv` when the script runs.

## Author

- dj@deepai.org

## License

This project is licensed under the ISC License - see the LICENSE file for details.
