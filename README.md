# Woman Face Blur – PyTorch Pipeline
## Overview
Woman Face Blur Pipeline is an end-to-end computer vision solution for automatically detecting and blurring female faces in images and videos. The project leverages deep learning models for person detection, face detection, and gender classification, making it suitable for privacy-preserving applications such as media anonymization, compliance with privacy laws, and ethical AI deployments.


## 📌 Features

- **Person Detection:**
 Locates people in images/videos using YOLOv8 models.
- **Face Detection:** Identifies faces within detected person regions.
- **Gender Classification:** Classifies detected faces as male or female using a trained MobileNetV3-based classifier.
- **Selective Blurring:**
 Blurs the entire person region if the detected face is classified as female.
- **Batch Processing:**
 Supports both images and videos
- **ML Engineering Practices:**
Modular code, reproducible and scalable training, and clear experiment tracking.
- **Kaggle-ready** – Run on kaggle free GPU resources without local hardware limitations.


## 🏗 Project Structure

```bash
woman_face_blur/
├── main.py
├── pipeline.py
├── README.md
├── requirements.txt
├── configs/
├── data/
├── datasets/
├── models/
├── outputs/
├── scripts/
├── tests/
└── utils/
```
## Getting Started
### 1. Clone the repository:
```bash
git clone https://github.com/AbdoEldgwy/Woman-face-blur.git
cd woman_face_blur
pip install -r requirements.txt
```

### 2. Dataset Preparation
- Download the CelebA dataset from [Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset) and place it in `data/celeba/`.
- Ensure `list_attr_celeba.csv` and images are in the correct folders.

### 3. Running the Pipeline
```python
from pipeline import WomanBlurPipeline

pipeline = WomanBlurPipeline()
pipeline.process_image(
    img_path="data/test_images/test_image.jpg",
    output_path="outputs/output_blurred.jpg",
    display=True
)
```
## Model Training & Accuracy Optimization

#### Gender Classifier Development
- **Initial Approach:** Started with a simple CNN (`GenderCNN`) trained on CelebA. Achieved moderate accuracy but struggled with generalization.
- **Challenge:** The dataset was imbalanced and the model overfit to the majority class (male).
- **Solution:** Implemented dataset balancing and label normalization. Used data augmentation (random flips, crops) to improve robustness.
- **Advanced Approach:** Fine-tuned a pretrained MobileNetV3-Small using transfer learning. This significantly improved accuracy and inference speed.
- **Experiment Tracking:** All experiments and results are documented in [`gender-classifier.ipynb`](https://github.com/AbdoEldgwy/Woman-face-blur/tree/main/models/gender-classifier.ipynb)

#### Training Highlights
- **Balanced Dataset:** Ensured equal representation of male and female samples.
- **Transforms:** Used normalization and augmentation for better generalization.
- **Early Stopping:** Prevented overfitting by monitoring validation accuracy.
- **Best Model:** Achieved **>97% validation accuracy** with MobileNetV3-Small. Model weights are saved in `outputs/best_mobilenetv3_small_gender.pth`.

## Example Results
original | blurred
:-------------------------:|:----------:
![original](https://github.com/AbdoEldgwy/Woman-face-blur/blob/main/data/test_images/test_image.jpg?raw=true)  | ![blurred](https://github.com/AbdoEldgwy/Woman-face-blur/blob/main/outputs/output_blurred.jpg?raw=true)


## Challenges & Solutions
- **Accuracy Optimization:** Faced difficulty with gender misclassification due to dataset imbalance and subtle facial features. Solved by balancing the dataset and using a more powerful backbone (MobileNetV3).
- **Speed vs. Accuracy:** Needed real-time inference for video. MobileNetV3-Small provided a good trade-off.
- **Deployment:** Exported model to ONNX for easy integration in production systems.

## References
- [CelebA Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [PyTorch](https://pytorch.org/)

## Contact
- [LinkedIn](https://www.linkedin.com/in/abdelrahman-eldgwy%E2%80%AC%E2%80%8F-7119171b9/)
- For questions or collaboration, please open an issue or contact abdo.eldgwy@gmail.com