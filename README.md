Dataset: GTSRB (assumed to be placed in a directory named gtsrb with subfolders for categories 0-42).
Preprocessing: Images are resized to 30x30, flattened, and standardized using StandardScaler.
Model: Sequential neural network with one hidden layer (varied units and activation), L2 regularization, and softmax output for 43 classes.
Hyperparameter Tuning: Grid search over:
Hidden units: 16, 32, 64, 128
L2 regularization: 0, 0.001, 0.01
Hidden activation: ReLU, Tanh
Epochs: 25, 50, 75, 100

Evaluation: Uses cross-validation set for tuning, test set for final metrics (accuracy, precision, recall, F1, confusion matrix).
Output: Saves the best model, test data, and generates plots for accuracy and confusion matrix.

The notebook performs a full pipeline from data import to model saving, with the best model selected based on validation loss.

Prerequisites:
Python 3.10+ (tested with 3.13.9 in the notebook metadata).
Jupyter Notebook or JupyterLab.
Dataset: Download the GTSRB dataset from INIs website and extract it into a folder named gtsrb in the same directory as the notebook. The structure should be gtsrb/<category_number>/*.jpg (or .ppm).

Installation

Clone or download this repository.
Install libraries manually based on imports

Dependencies
The notebook imports the following libraries:
os, cv2 (OpenCV), numpy, matplotlib, tensorflow, sklearn (preprocessing, model_selection, metrics), seaborn, pandas, datetime.

Key versions (inferred from notebook):
TensorFlow: 2.x
scikit-learn: 1.x
Others: Standard Python libraries.

To install:
textpip install opencv-python numpy matplotlib tensorflow scikit-learn seaborn pandas

Usage
Place the GTSRB dataset in the gtsrb directory.
Open the notebook in Jupyter:textjupyter notebook part2.ipynb
Run all cells sequentially.
Data loading may take time due to ~26,640 images.
Hyperparameter tuning performs 128 trainings (4 units × 3 L2 × 2 activations × 4 epochs).

Outputs:
Printed metrics (accuracy, loss, precision, recall, F1).
Confusion matrix heatmap (via Seaborn).
Accuracy plot (train vs. validation).
Best model saved in a timestamped folder (e.g., 2023-10-01_12-00-00/<loss>_<acc>.keras).
Test data saved as CSV and NPY files.


Results
The notebook tracks the best configuration based on validation loss.
Example output (from notebook): Best config might be something like units=64, L2=0.0, activation='relu', epochs=50, with val_loss ~0.2587 and val_acc ~0.9538.
Final test evaluation includes precision, recall, F1 per class, and overall accuracy.

Notes
The notebook assumes 43 classes (0-42) and skips .DS_Store files (macOS artifact).
No GPU acceleration is explicitly configured; TensorFlow will use GPU if available.
The document provided is truncated (~64,650 characters), but the code appears complete up to model saving.
For improvements: Consider adding convolutional layers (CNN) for better image classification performance, as this is a flat MLP.

License
This project is unlicensed (public domain). Feel free to use or modify.
