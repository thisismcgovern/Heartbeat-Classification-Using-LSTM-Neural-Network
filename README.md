# 🫀 Heartbeat Classification Using LSTM Neural Network

This project builds and evaluates a **Long Short-Term Memory (LSTM)** neural network to classify heartbeat signals from the **MIT-BIH Arrhythmia dataset**, available on Kaggle.
The model learns to distinguish between different heartbeat types based on ECG waveform data, demonstrating how deep learning can assist in automated cardiac diagnostics.

---

## 📊 Dataset

**Source:** [MIT-BIH Arrhythmia Dataset (via Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

The dataset contains segmented heartbeat data derived from ECG recordings, split into:

* `mitbih_train.csv`: Training samples
* `mitbih_test.csv`: Test samples

Each sample represents 187 signal points followed by a label (0–4), corresponding to heartbeat classes:

| Label | Heartbeat Type                   |
| ----- | -------------------------------- |
| 0     | Normal                           |
| 1     | Supraventricular Premature       |
| 2     | Ventricular Premature            |
| 3     | Fusion of Ventricular and Normal |
| 4     | Unclassifiable                   |

---

## ⚙️ Project Workflow

### 1. Environment Setup

We configure access to Kaggle’s API, authenticate with a `kaggle.json` key, and download the heartbeat dataset.

### 2. Data Loading & Preprocessing

* Load CSV data using **pandas**.
* Split data into features (`X`) and labels (`y`).
* Reshape inputs to fit LSTM expectations:
  Each sample → (timesteps, features) = (187, 1).
* One-hot encode labels using `to_categorical()` for multi-class classification.

### 3. Model Architecture

A sequential **LSTM-based neural network** was built using **TensorFlow/Keras**:

```python
model = Sequential([
    LSTM(64, input_shape=(187, 1)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')
])
```

* **LSTM layer**: Captures temporal dependencies in ECG signals.
* **Dropout**: Reduces overfitting by randomly disabling neurons during training.
* **Dense layers**: Perform nonlinear transformation and classification.
* **Softmax**: Outputs class probabilities across 5 heartbeat types.

### 4. Model Compilation & Training

Compiled using:

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Trained for **5 epochs** with a **batch size of 64**, tracking both training and validation performance.

### 5. Evaluation

* Visualized training/validation **accuracy and loss** curves.
* Computed **confusion matrix** and **classification report** for detailed performance analysis.

---

## 📈 Results

* Achieved **≈0.82 test accuracy** after 5 epochs.
* The model performs well in detecting normal beats but struggles slightly with rarer heartbeat types due to class imbalance.

---

## 🔍 Visualizations

* **Training Curves:** Accuracy and loss over epochs.
* **Confusion Matrix:** Displays correct vs. incorrect predictions across all classes.
* **Classification Report:** Includes precision, recall, and F1-score for each class.

---

## 🧠 Future Improvements

* Apply **data balancing** (e.g., SMOTE or class weighting).
* Use **1D CNN-LSTM hybrid models** for enhanced feature extraction.
* Increase epochs or fine-tune hyperparameters.
* Deploy model in a real-time ECG monitoring system.

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy / Pandas**
* **Seaborn / Matplotlib**
* **Scikit-learn**
* **Google Colab**

---

## 📚 Reference

* Shayan Fazeli and M. Sarrafzadeh, *“Heartbeat Classification Using Deep Learning Techniques”*, Kaggle, MIT-BIH Dataset.
* [Original Paper Link (if applicable)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

---

## 👤 Author

**McGovern Twumasi Owusu-Bekoe**
Biomedical Engineering | AI & Healthcare Research
📧 [Add your email or LinkedIn if you wish]
