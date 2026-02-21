🧠 MNIST Digit Classification with Deep Neural Networks

An end-to-end implementation of handwritten digit recognition using a fully connected neural network trained on the MNIST dataset.

This project demonstrates model design, training, checkpointing, performance visualization, and evaluation using TensorFlow/Keras.

📌 Project Overview

The goal of this project is to build a neural network capable of classifying handwritten digits (0–9) from grayscale images.

Unlike image denoising projects (which use Autoencoders to reconstruct clean images), this project focuses purely on supervised classification.

The MNIST dataset contains:

60,000 training images

10,000 test images

Image size: 28×28 pixels

10 output classes (digits 0–9)

All images were normalized to the range [0, 1] before training.

🧠 Model Architecture

The model consists of:

Flatten layer (28×28 → 784)

Fully connected Dense layer(s) with ReLU activation

Output layer with Softmax activation (10 neurons)

Loss Function: Sparse Categorical Crossentropy
Optimizer: Adam
Evaluation Metric: Accuracy

🚀 Training Configuration

Epochs: 10

Model checkpoint saved after each epoch

Training and validation metrics logged

Visualization supported via TensorBoard

Although the model was trained for only 10 epochs (as this is a practice project), the performance converged effectively within this range.

📊 Performance Results
Accuracy

Training Accuracy: ~99%

Validation Accuracy: ~99%

Minimal gap between training and validation accuracy

No significant overfitting observed

Loss

Training Loss decreased from ~0.30 to ~0.02

Validation Loss decreased from ~0.08 to ~0.025

Stable convergence across epochs

Minor fluctuation around epoch 7, without instability

Overall, the model demonstrates strong generalization performance given its simple fully connected architecture.

📈 Performance Analysis

The training curves show consistent improvement in both accuracy and loss.

The close alignment between training and validation metrics suggests that the model effectively learned meaningful patterns without memorizing the dataset.

Given the simplicity of MNIST, a fully connected network can achieve near-optimal performance.
Further improvements would likely require architectural changes (e.g., Convolutional Neural Networks) rather than simply increasing the number of epochs.


📂 Project Structure
mnist-digit-classification/
│
├── main.py
├── model.py
├── checkpoints/
├── logs/
├── requirements.txt
└── README.md


🛠 How to Run

Clone the repository:

git clone https://github.com/c-ehsan/MNIST_Neural_Network_Classifier
cd mnist-digit-classification

Install dependencies:

pip install -r requirements.txt

Run training:

python main.py

To visualize training logs:
tensorboard --logdir=logs


🔮 Future Improvements

Implement a Convolutional Neural Network (CNN)

Add Dropout for stronger regularization

Perform hyperparameter tuning

Add confusion matrix visualization

Deploy as a simple web application

📜 License

This project is released under the MIT License.
