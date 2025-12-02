# Machine Learning Homework Analysis

# Murad Mirzayev 

## 1. Project Overview
- Task: Image classification using a neural network.
- Dataset: Imbalanced dataset with **Class 0 = 303 samples** and **Class 1 = 647 samples**.
- Goal: Train a model to classify images accurately while handling class imbalance.
-------------------------------------------------------------------
## 2. Data Preprocessing
- Images were loaded and normalized.
- Data augmentation was considered to improve generalization.
- Class weights were calculated and applied in the loss function to handle imbalance:
  - Class 0 weight: 1.5677
  - Class 1 weight: 0.7342
-------------------------------------------------------------------
## 3. Model Training
- Training used standard techniques for supervised classification.
- Loss function: Cross-entropy with class weights.
- Optimizer: [Your optimizer, e.g., Adam] (replace if different).
- Epochs: [Number of epochs you used]
- Batch size: [Batch size you used]
-------------------------------------------------------------------
## 4. Results
- Test accuracy: **70.63%**
- Observations:
  - The model performs reasonably well despite class imbalance.
  - Some misclassification occurs in the minority class (Class 0) due to fewer samples.
  - Training loss decreased steadily, but validation performance plateaued, indicating slight overfitting.
-------------------------------------------------------------------
## 5. Analysis & Improvements
- **Strengths**:
  - Class weighting improved minority class performance.
  - Model achieved acceptable accuracy on the test set.
- **Weaknesses**:
  - Overfitting observed after several epochs.
  - Limited dataset size affects generalization.
- **Future Improvements**:
  - Data augmentation to artificially increase dataset size.
  - Hyperparameter tuning (learning rate, batch size, optimizer choice).
  - Experiment with more complex architectures if needed.
-------------------------------------------------------------------