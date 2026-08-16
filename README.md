# Multiview-to-CAD AutoCraft

An AI-powered pipeline for generating 3D vehicle CAD geometry from multiple images of a vehicle.

This project is being developed as part of the AutoCraft project by Delta-Cross Dynamics.

## Current Status

The current prototype focuses on vehicle image segmentation using **Microsoft Florence-2**.

The pipeline currently:

1. Takes a vehicle image as input.
2. Uses Florence-2 referring-expression segmentation to identify the car.
3. Extracts the segmentation polygon.
4. Generates a binary mask of the vehicle.
5. Provides the segmented vehicle as input for subsequent 3D reconstruction stages.

The complete multiview-to-CAD pipeline is still under development.

## Project Goals

The eventual system is intended to:

- Accept multiple photographs of a vehicle.
- Detect and segment the vehicle in each image.
- Extract useful visual information from different views.
- Reconstruct the vehicle's 3D shape.
- Generate a usable 3D/CAD representation.
- Integrate the resulting workflow into AutoCraft.

## Current Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Microsoft Florence-2
- Computer Vision
- Image Segmentation
- 3D Reconstruction

## Project Structure

```text
multiview-to-cad-autocraft/
├── input/              # Local input images (not tracked by Git)
├── output/             # Generated results (not tracked by Git)
├── test_florence.py    # Florence-2 segmentation testing
├── test_node.py        # Node/pipeline testing
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md