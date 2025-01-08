# White-blood-cell_classification_using_multihead_CNN


## Overview
This project focuses on classifying white blood cells into five categories: Basophil, Eosinophil, Lymphocyte, Monocyte, and Neutrophil using a Convolutional Neural Network (CNN). The model is evaluated based on its performance, as visualized in the confusion matrix provided.

## Features
- **Deep Learning Model:** A custom CNN architecture is implemented with attention mechanisms to improve performance.
- **Data Preprocessing:** Includes normalization and augmentation using `ImageDataGenerator`.
- **Visualization:** Confusion matrix provides insights into classification performance and misclassification patterns.
- **Performance Metrics:** Evaluation based on accuracy, precision, recall, and F1-score.

## Model Architecture
The CNN model includes the following layers:
1. **Convolutional Layers:** Extract spatial features using filters of varying sizes (e.g., 8x8, 5x5).
2. **Batch Normalization:** Normalizes the activations to stabilize and accelerate training.
3. **Pooling Layers:** Reduces spatial dimensions to focus on salient features.
4. **Attention Mechanism:** Enhances feature focus to improve classification accuracy.
5. **Dense Layers:** Fully connected layers for final classification into 5 categories.
6. **Dropout Layers:** Regularization to prevent overfitting.

## Dataset
The dataset contains labeled images of white blood cells, divided into:
- **Training Set:** For model training.
- **Validation Set:** For hyperparameter tuning.
- **Test Set:** For final evaluation.

## Confusion Matrix
The confusion matrix visualizes the classification results:
- **Rows:** True labels.
- **Columns:** Predicted labels.
- **Diagonal Values:** Correct classifications.
- **Off-Diagonal Values:** Misclassifications.

### Observations:
- The model performs well overall, with high diagonal values.
- Common misclassifications:
  - Eosinophils often misclassified as Neutrophils (488 instances).
  - Lymphocytes occasionally misclassified as Eosinophils (34 instances).

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```
5. View the confusion matrix:
   ```bash
   python plot_confusion_matrix.py
   ```

## Requirements
- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Potential Improvements
1. **Data Augmentation:** Enhance underrepresented classes to reduce misclassification.
2. **Hyperparameter Tuning:** Experiment with different optimizers, learning rates, and architectures.
3. **Advanced Models:** Implement Vision Transformers or ResNet for better feature extraction.
4. **Error Analysis:** Analyze misclassified samples to refine preprocessing and model design.

## Results
The final model achieves:
- **High Accuracy:** Correctly classifies most test samples.
- **Misclassification Patterns:** Errors primarily occur between visually similar cell types (e.g., Eosinophils vs. Neutrophils).

## Acknowledgments
This project is inspired by advancements in medical imaging and the potential of deep learning to aid in diagnostics. Special thanks to the open-source community for providing datasets and tools.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
```

Contact kuantian_procai@hotmail.com for more information
