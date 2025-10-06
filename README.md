# Woman Face Blur – PyTorch Pipeline

A deep learning pipeline to **detect, segment, and blur women's faces** in images or real-time video streams, built with PyTorch.  
The goal is to provide an **automated, scalable, and privacy-focused solution** for anonymizing women’s identities in photos and videos.



## 📌 Features

- **Multi-stage or end-to-end** architecture:
  - **Face Detection** – Detect human faces in images or frames.
  - **Gender Classification** – Identify if the detected face is female.
  - **Face Segmentation** – Pixel-level mask generation for precise blurring.
  - **Blurring** – Apply Gaussian blur to masked face areas.
- Works on:
  - Static images
  - Real-time webcam feeds
  - Pre-recorded videos
- **Modular design** – Swap models or components without rewriting the entire pipeline.
- **Scalable training** – Config-driven approach for custom datasets.
- **Colab-ready** – Run on free GPU resources without local hardware limitations.


## 🏗 Project Structure

```bash
woman_face_blur/
│
├── configs/ # YAML/JSON config files
├── data/ # Dataset instructions
├── datasets/ # Data loading & preprocessing
├── models/ # Model architectures (detector, classifier, segmenter)
├── utils/ # Helper functions
├── scripts/ # Training & inference scripts
└── tests/ # Unit tests
```
## Pipeline Flow
```mathematica
Image/Video Frame
     │
YOLOv8 (person detection)
     │
For each person:
     ├─> Crop person box
     ├─> Detect face in crop
     ├─> Classify gender from face
     ├─> If female → Blur person box
```
## Datasets:

- **Detection**: 
  - WIDER FACE (very varied, but no gender label)
- **Gender Classification**: 
  - CelebA (202k images, has gender attribute)
- **Segmentation**: 
  - CelebAMask-HQ (segmentation masks for CelebA)
---
**Combined approach will be used**: Train detection on WIDER FACE, gender classifier on CelebA, segmentation on CelebAMask-HQ.


