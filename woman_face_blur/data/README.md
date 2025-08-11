# Data instructions

This project uses two types of data during development:

1. **Detection** (person) - we use a pre-trained YOLO (COCO) model for person detection, so you DO NOT need to download a detection dataset to try inference.

2. **Gender classifier (optional)** - to train your own gender classifier, download CelebA:
   - Download images and `list_attr_celeba.txt` from the CelebA project page.
   - Place images under `data/celeba/img_align_celeba`.
   - The dataset class in `datasets/celeba_dataset.py` expects the `list_attr_celeba.txt` in the same `data/celeba/` folder.

Place test images or short videos for quick experiments in `data/inputs/`.