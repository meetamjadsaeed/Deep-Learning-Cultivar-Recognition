# 📑 Deep Learning: Cultivar Recognition

This project focuses on classifying tomato cultivars from the dataset [Tomato Cultivars](https://www.kaggle.com/olgabelitskaya/tomato-cultivars). The dataset contains around 700-800 images of size 160x160x3, categorized into 15 different tomato cultivars. The images are stored in Hierarchical Data Format (HDF5).

## Project Overview

- **Dataset**: Images in PNG format, labeled by file prefixes.
- **Tools and Libraries**:
  - [scikit-learn](https://scikit-learn.org/stable/): Machine Learning in Python
  - [Keras](https://keras.io/): Python Deep Learning Library
  - [PyTorch](https://pytorch.org/): Open Source Machine Learning Framework
  - [Neural Structured Learning](https://github.com/tensorflow/neural-structured-learning): Library for adversarial learning

## Installation

First, ensure you have the necessary packages:

```bash
pip install --upgrade pip
pip install neural_structured_learning
```

## Data Loading and Preprocessing

The data is downloaded from the provided URL and loaded using the HDF5 format. Images are resized and split into training, validation, and test sets. The dataset is then converted into PyTorch `Dataset` and `DataLoader` objects.

## Model Training and Evaluation

### Scikit-Learn Algorithms

- **Random Forest Classifier**:
  - Trained and evaluated on the reshaped image data.
  - Performance metrics: accuracy, hamming loss.

### CNN-Based Models with Adversarial Regularization

- **Custom CNN**:
  - Built using TensorFlow/Keras.
  - Utilizes adversarial regularization for improved performance.

### TensorFlow Hub Models

- **MobileNetV2**:
  - Pretrained model used for feature extraction and fine-tuning.
- **InceptionV3**:
  - Another pre-trained model used for feature extraction and classification.

### TorchVision Models

- **VGG16**:
  - Pretrained VGG16 model modified for the task.
  - Fine-tuned and evaluated using PyTorch.

## Code Overview

### Importing Libraries and Defining Functions

Imports various libraries for data handling, model training, and evaluation.

### Data Loading

Defines functions for downloading, loading, and preprocessing the dataset.

### Model Training

Defines and trains models using different libraries and techniques, including sci-kit-learn, TensorFlow/Keras, and PyTorch.

## Example Usage

1. **Run the Script**:

    ```bash
    python script_name.py
    ```

2. **Interactive Visualization**:
   - Displays sample images from the dataset.

3. **Training Models**:
   - Runs training for various models and displays performance metrics.

## Results

The models are evaluated based on accuracy and loss. Performance reports and plots for different classifiers and neural networks are generated.

## License

This project is open-source and available under the MIT License.

## Acknowledgements

- Dataset provided by [Olga Belitskaya](https://www.kaggle.com/olgabelitskaya).
