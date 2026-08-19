# Image-to-3D Vehicle Body / AutoCraft

Clean baseline project for reconstructing exterior vehicle panels from photographs.

## Baseline experiment

**Comprehensive Exterior Junction Detection**

Input: one vehicle photograph.

VLM: `gpt-5.6-sol`

Outputs:
- detected visible exterior components
- visible sides
- junction/node list
- normalized and pixel coordinates
- component associations
- junction category
- visibility
- confidence
- short reasoning
- annotated visualization with numbered nodes
- JSON result

## Exterior-only topology

The system models only exterior surfaces. The B-pillar is not an exterior CAD panel.
Visible window panes/glass are treated as exterior surfaces.

## Run

Create a fresh Windows virtual environment and install:

```text
pip install openai opencv-python numpy pillow
```

Set the API key:

PowerShell:
```powershell
$env:OPENAI_API_KEY="your-key"
```

Put a vehicle image in `input/`, then:

```powershell
python experiments/junction_detection.py input/car01.jpg
```

Results are written to `output/`.
