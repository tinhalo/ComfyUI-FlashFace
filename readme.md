# ComfyUI-FlashFace
ComfyUI implementation of FlashFace: Human Image Personalization with High-fidelity Identity Preservation </center>
> Officially implemented here: https://github.com/ali-vilab/FlashFace

## Installation </center>
1. Navigate to the `custom_nodes` directory in your ComfyUI installation.
2. Install the custom node by running `git clone https://github.com/cold-hand/ComfyUI-FlashFace.git` in the `custom_nodes` directory.
3. If you are using a virtual environment, activate it.
4. cd into the `ComfyUI-FlashFace` directory and run setup.sh or setup.bat depending on your OS.
   - This will install all required dependencies including `pydash`
   - Alternatively, you can run `pip install pydash==7.0.7` manually
5. Restart ComfyUI and refresh your browser and you should see the FlashFace node in the node list.
6. Load the provided example-workflow.json in file in the examples/comfyui folder of this repo to see how the nodes are used.

## Dependencies

This extension requires the following key dependencies:

- Python 3.8+
- PyTorch (included in ComfyUI)
- pydash 7.0.7 or newer
- See `requirements.txt` for the full list

If you encounter any import errors, run `python install_dependencies.py` from the extension directory.

## Documentation

- [Installation Guide](docs/installation.md) - Detailed installation instructions
- [Technical Documentation](docs/technical.md) - Architecture and developer information
- [Improvements](docs/improvements.md) - Summary of recent refactoring and improvements

## Results </center>
The FlashFace node takes in an image and outputs a personalized image with the desired attributes. In comparison with InstantId
I feel that it yields more realistic results. The original repo had a limit of 4 reference images, that limitation has been removed in this implementation.
I've found that generally the more reference images you provide the better the results. as seen below:

Example Reference Image: 
<br /><img src="examples/comfyui/Nicki/source.jpeg" width=25% height=auto>
<br />Nicki Minaj
<table>
  <tr>
    <td>Output with 1 Reference Image</td>
     <td>Output with 4 Reference Images</td>
     <td>Output with 8 Reference Images</td>
     <td>Output with 16 Reference Images</td>
  </tr>
  <tr>
    <td><img src="examples/comfyui/Nicki/1.png" width=100% height=auto></td>
    <td><img src="examples/comfyui/Nicki/4.png" width=100% height=auto></td>
    <td><img src="examples/comfyui/Nicki/8.png" width=100% height=auto></td>
    <td><img src="examples/comfyui/Nicki/16.png" width=100% height=auto></td>
  </tr>
</table>



## Change Log
- [5/22/2024]: Initial release
- [Current Date]: Major refactoring and improvements:
  - Added robust model loading with SafeTensors support
  - Implemented automatic model downloading from HuggingFace
  - Improved model path resolution for different ComfyUI installation configurations
  - Enhanced error handling and reporting
  - Added pydash dependency to requirements
  - Added model registry system for easy management of model files


