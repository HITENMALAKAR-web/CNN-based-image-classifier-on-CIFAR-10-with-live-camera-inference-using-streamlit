# CIFAR-10 Image Classifier (Streamlit + Live Camera Inference)

A CNN-based image classifier trained on the CIFAR-10 dataset, deployed as an interactive Streamlit web app. Upload an image (or use live camera input) and get real-time predictions across 10 object classes.

## Classes
plane, car, bird, cat, deer, dog, frog, horse, ship, truck

## Features
- Upload an image (jpg, jpeg, png) for instant classification
- Custom CNN architecture (`Hnet`) built with PyTorch
- Simple, responsive Streamlit UI

## Project Structure
├── app.py # Streamlit app entry point
├── src/
│ ├── init.py
│ ├── pedict.py # Model loading & inference logic
│ ├── CNN_Architecture.py # Hnet model definition
│ └── data_loader.py # Class name definitions
├── model/
│ └── net.pth # Trained model weights
├── requirements.txt
└── README.md

## Model Architecture — Hnet

Hnet follows a VGG-style design: four convolutional blocks of increasing depth, each doubling the channel width and halving the spatial resolution via max pooling.

| Block | Channels | Layers | Output Size |
|-------|----------|--------|--------------|
| 1 | 3 → 64 → 64 | 2× Conv2d(3×3) + BatchNorm + ReLU, then MaxPool | 16×16 |
| 2 | 64 → 128 → 128 | 2× Conv2d(3×3) + BatchNorm + ReLU, then MaxPool | 8×8 |
| 3 | 128 → 256 → 256 | 2× Conv2d(3×3) + BatchNorm + ReLU, then MaxPool | 4×4 |
| 4 | 256 → 512 → 512 | 2× Conv2d(3×3) + BatchNorm + ReLU, then MaxPool | 2×2 |

Each convolution uses `padding=1` to preserve spatial dimensions before pooling, and BatchNorm after every conv layer stabilizes training and speeds convergence.

The resulting 512×2×2 feature map (2,048 values) is flattened and passed through a classifier head:
- `Linear(2048 → 128)`
- `Dropout(0.4)` for regularization
- `Linear(128 → 64)`
- `Linear(64 → 10)` — final class scores

## Setup

1. Clone the repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

## Model
The classifier uses a custom CNN (`Hnet`) trained on the CIFAR-10 dataset. Input images are resized to 32x32 and normalized before inference.

## License
MIT