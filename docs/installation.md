# Installation Guide for ComfyUI-FlashFace

This guide provides detailed instructions for installing and setting up the ComfyUI-FlashFace extension, including the recent improvements for model management and automatic downloads.

## Basic Installation

1. Navigate to the `custom_nodes` directory in your ComfyUI installation.
2. Clone the repository by running:

   ```bash
   git clone https://github.com/cold-hand/ComfyUI-FlashFace.git
   ```

3. Change to the extension directory:

   ```bash
   cd ComfyUI-FlashFace
   ```

4. Run the setup script for your operating system:
   - For Windows: `setup.bat`
   - For Linux/macOS: `./setup.sh`
   
   These scripts will install the required Python dependencies, including `pydash`, and download all necessary model files.
   
   Alternatively, you can manually install dependencies:

   ```bash
   pip install -r requirements-comfy.txt
   pip install pydash==7.0.7
   ```

5. Restart ComfyUI and refresh your browser.

## Dependencies

The extension requires the following Python packages:

- accelerate
- huggingface-hub
- numpy
- omegaconf
- easydict
- ftfy
- pydash (version 7.0.7 or newer)

If you encounter any import errors related to missing packages, you can run:

```bash
python install_dependencies.py
```

## New Features

### Automatic Model Downloads

The extension now supports automatic downloading of model files when they are not found locally. When you first run a node that requires a model, it will:

1. Check if the model exists in the correct location
2. If not found, download it from HuggingFace automatically
3. Verify the download with MD5 checksum to ensure file integrity

This makes the installation process much simpler as you don't need to manually download model files.

### Improved Model Path Resolution

The extension now has a robust model path resolution system that checks multiple possible locations for models:

1. Standard ComfyUI location: `ComfyUI/models/flashface/`
2. Custom node location: `ComfyUI/custom_nodes/ComfyUI-FlashFace/models/flashface/`
3. Extension directory: `ComfyUI/custom_nodes/ComfyUI-FlashFace/models/`
4. Fallback cache directory: `ComfyUI/custom_nodes/ComfyUI-FlashFace/cache/`

This ensures the extension works across different ComfyUI installation configurations.

### SafeTensors Support

All model loading now supports the SafeTensors format, which provides:

1. Better security (protection against malicious model files)
2. Faster loading times for large models
3. Improved memory efficiency

## Troubleshooting

If you encounter any issues with automatic downloads:

1. Check your internet connection
2. Ensure you have write permissions for the models directory
3. Try manually downloading the model files from:
   - FlashFace model: [https://huggingface.co/mmeendez/ComfyUI-FlashFace/resolve/main/model.safetensors](https://huggingface.co/mmeendez/ComfyUI-FlashFace/resolve/main/model.safetensors)
   - RetinaFace model: [https://huggingface.co/mmeendez/ComfyUI-FlashFace/resolve/main/detection_Resnet50_Final.pth](https://huggingface.co/mmeendez/ComfyUI-FlashFace/resolve/main/detection_Resnet50_Final.pth)

Place them in the `ComfyUI/models/flashface/` directory with the filenames:

- `flashface.safetensors` for the FlashFace model
- `retinaface_resnet50.pth` for the RetinaFace detection model

If you encounter "ImportError: No module named 'pydash'" or similar:

1. Run `python install_dependencies.py` from the extension directory
2. Or manually install pydash: `pip install pydash==7.0.7`
