# Leukemia Classification with CNN and SVM

This project aims to classify leukemia images into two categories: `hem` (healthy) and `all` (acute lymphoblastic leukemia) using a combination of Convolutional Neural Networks (CNN) and Support Vector Machines (SVM).

## Project Overview

- **Dataset**: The dataset consists of grayscale images of blood samples, divided into training, validation, and testing sets.
- **Model**: A CNN is used for feature extraction, and the extracted features are further classified using PCA and SVM.
- **Goal**: To achieve high accuracy in classifying leukemia images while addressing challenges like underfitting and overfitting.

## Key Features

1. **Data Augmentation**: Techniques like rotation, zoom, and horizontal flipping are applied to improve generalization.
2. **CNN Architecture**: A custom CNN model with dropout layers to prevent overfitting.
3. **PCA + SVM**: Principal Component Analysis (PCA) is used to reduce dimensionality, and SVM is used for final classification.

## Challenges

- **Underfitting**: The model may underfit due to limited dataset size or insufficient complexity in the architecture.
- **Overfitting**: Dropout layers and early stopping are used to mitigate overfitting.

## Results

- **Train Accuracy**: Achieved during CNN training.
- **Validation Accuracy**: Monitored to prevent overfitting.
- **Test Accuracy**: Evaluated using SVM on PCA-transformed features.

## How to Use

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python jackfruitcode_1.py
   ```

## Limitations

- The model may not generalize well to unseen data due to dataset limitations.
- Further hyperparameter tuning and data preprocessing could improve results.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Dataset: [C-NMC_Leukemia Dataset](https://www.kaggle.com/competitions/c-nmc-leukemia/overview)
- Libraries: TensorFlow, scikit-learn, Matplotlib, NumPy
