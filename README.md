# Image Augmentation Script for Deep Learning

This repository contains a Python script designed to augment image datasets, particularly in preparation for training deep learning models. The script uses TensorFlow to apply a variety of random transformations to images, increasing the diversity of the dataset and improving the robustness of machine learning models.

## Features
- **Random Transformations:** Includes horizontal flipping, random rotation, brightness, contrast, saturation, and hue adjustments.
- **Noise Addition:** Introduces Gaussian noise to the images for further variability.
- **Random Translation:** Slight shifts in the image position to simulate different perspectives.
- **High-Quality Output:** Augmented images are saved in the same format and quality as the original.
- **Batch Augmentation:** Processes entire directories of images, creating multiple augmented versions of each image.

## Use Cases
- **Data Preparation for GANs:** Ideal for projects like WGAN-GP, where diverse training data is crucial for generating high-quality images.
- **Image Classification:** Enhances datasets for training more robust and generalized models in image recognition tasks.
- **Research and Development:** Useful for researchers and developers experimenting with data augmentation techniques.

## How to Use
1. Clone the repository.
2. Update the `dataset_path` variable in the script to point to your image dataset directory.
3. Run the script to generate augmented images.

## Requirements
- TensorFlow
- Python 3.x

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
