# 🌿 Plant Disease Detection

### Deep Learning-Based Plant Disease Classification Using Computer Vision

Plant Disease Detection is a deep learning project that identifies plant diseases from leaf images using **Convolutional Neural Networks (CNNs)**.

The system uses the **PlantVillage dataset** along with image preprocessing and data augmentation to train a computer vision model capable of classifying plant leaves into different disease categories.

The project also includes a prediction interface, prediction logging, and training-performance visualizations to analyze the behavior of the trained model.

---

## 🚀 Features

### 🌱 Disease Detection

Upload an image of a plant leaf and use the trained deep learning model to predict the corresponding disease class.

* Image-based disease classification
* CNN-based deep learning model
* Automated image preprocessing
* Model-based prediction
* Prediction confidence analysis

### 🧠 Deep Learning Model

The project uses **TensorFlow and Keras** to build and train the image classification model.

The training pipeline includes:

* Image preprocessing
* Image resizing
* Pixel normalization
* Data augmentation
* CNN feature extraction
* Multi-class classification
* Model evaluation

### 🔄 Data Augmentation

Data augmentation is applied during training to improve the model's ability to generalize to different leaf images.

Possible transformations include:

* Image rotation
* Horizontal/vertical transformations
* Zooming
* Shifting
* Flipping

This helps reduce overfitting and exposes the model to greater variation during training.

### 📊 Training Analysis

Training performance is recorded and visualized using:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Training history

The repository includes training-curve visualizations for evaluating model performance.

### 📝 Prediction Logging

Predictions can be recorded in a CSV file for later analysis.

```text
prediction_log.csv
```

This allows predictions to be tracked and reviewed after model inference.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │     Leaf Image       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Preprocessing  │
                    │ Resize / Normalize   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   CNN Deep Learning  │
                    │       Model          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Extraction   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Disease Classifier   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Predicted Disease    │
                    │ + Confidence Score   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Prediction Log     │
                    └──────────────────────┘
```

---

## 🔬 Machine Learning Pipeline

```text
PlantVillage Dataset
        │
        ▼
Data Collection
        │
        ▼
Image Preprocessing
        │
        ▼
Image Resizing & Normalization
        │
        ▼
Data Augmentation
        │
        ▼
Train / Validation Split
        │
        ▼
CNN Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Disease Prediction
        │
        ▼
Prediction Logging
```

---

## 📚 Dataset

The project uses the **PlantVillage dataset**, a widely used dataset for research and experimentation in plant disease recognition.

The dataset contains images of plant leaves representing healthy and diseased conditions.

The images are processed before being supplied to the neural network so that they have a consistent input format.

---

## 🛠️ Tech Stack

### Programming Language

* **Python**

### Deep Learning

* **TensorFlow**
* **Keras**
* Convolutional Neural Networks (CNN)

### Computer Vision

* Image preprocessing
* Image augmentation
* Image classification

### Data Processing

* **NumPy**
* **Pandas**

### Visualization

* **Matplotlib**

### Application

* Python-based prediction interface
* Image upload and inference
* Prediction logging

---

## 📂 Project Structure

```text
PlantDiseaseDetection/
│
├── app.py
├── main1.py
│
├── prediction_log.csv
├── training_history.json
│
├── sample_predictions.png
├── training_curves.png
│
└── README.md
```

### File Description

| File                     | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `app.py`                 | Application/inference interface                   |
| `main1.py`               | Main project/model implementation                 |
| `prediction_log.csv`     | Stores prediction results                         |
| `training_history.json`  | Stores model training history                     |
| `training_curves.png`    | Training and validation performance visualization |
| `sample_predictions.png` | Sample model prediction results                   |
| `README.md`              | Project documentation                             |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/divyansh-3371/PlantDiseaseDetection.git
cd PlantDiseaseDetection
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Install the required Python libraries:

```bash
pip install tensorflow keras numpy pandas matplotlib
```

If the repository contains a `requirements.txt` file in a future version, use:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the application using:

```bash
python app.py
```

For the main training/prediction workflow:

```bash
python main1.py
```

---

## 🧪 Model Training

The model training process consists of several stages:

### 1. Dataset Preparation

The PlantVillage images are loaded and organized according to their respective disease classes.

### 2. Image Preprocessing

Images are transformed into a consistent format suitable for neural network training.

```text
Raw Image
    ↓
Resize
    ↓
Normalize
    ↓
Augment
    ↓
Model Input
```

### 3. CNN Training

The CNN learns visual patterns from plant leaves, including features associated with disease symptoms.

The network progressively learns:

```text
Edges
  ↓
Textures
  ↓
Shapes
  ↓
Leaf Patterns
  ↓
Disease-Specific Features
```

### 4. Evaluation

The model is evaluated using training and validation performance.

The recorded training history can be used to analyze:

* Model convergence
* Overfitting
* Underfitting
* Training stability
* Validation performance

---

## 📈 Model Performance

Training performance is stored in:

```text
training_history.json
```

and visualized using:

```text
training_curves.png
```

Example evaluation workflow:

```text
Training Data
     │
     ▼
CNN Model
     │
     ├──────────────► Training Accuracy
     │
     └──────────────► Training Loss

Validation Data
     │
     ▼
CNN Model
     │
     ├──────────────► Validation Accuracy
     │
     └──────────────► Validation Loss
```

---

## 🔎 Prediction Workflow

```text
User Uploads Leaf Image
          │
          ▼
    Image Validation
          │
          ▼
      Preprocessing
          │
          ▼
      CNN Inference
          │
          ▼
  Predicted Disease Class
          │
          ▼
   Prediction Confidence
          │
          ▼
    Save Prediction Log
```

Predictions can be recorded in:

```text
prediction_log.csv
```

This makes it possible to analyze previous predictions and model behavior.

---

## 💡 Applications

This type of computer vision system can be useful for:

* 🌾 Smart agriculture
* 🌱 Early plant disease identification
* 👨‍🌾 Farmer assistance systems
* 🔬 Agricultural research
* 📊 Crop health monitoring
* 🤖 AI-based agricultural applications
* 📱 Future mobile/web-based disease detection systems

---

## 🔮 Future Improvements

The project can be extended in several directions:

* [ ] Improve model accuracy
* [ ] Experiment with transfer learning
* [ ] Compare CNN architectures
* [ ] Add more plant species
* [ ] Add more disease categories
* [ ] Implement real-time camera detection
* [ ] Add treatment recommendations
* [ ] Add disease severity estimation
* [ ] Deploy the model as a REST API
* [ ] Build a mobile application
* [ ] Deploy the application to the cloud
* [ ] Add explainable AI using Grad-CAM
* [ ] Add model confidence thresholds
* [ ] Detect non-leaf/non-plant images

### 🔬 Possible Advanced Version

A future version could combine:

```text
Image
  │
  ▼
Plant Detection
  │
  ▼
Disease Classification
  │
  ▼
Disease Severity Estimation
  │
  ▼
Treatment Recommendation
  │
  ▼
Agricultural Assistance
```

---

## ⚠️ Limitations

The model is trained using a specific dataset, so real-world images may differ from the training data.

Performance can be affected by:

* Poor lighting
* Blurry images
* Different camera qualities
* Background noise
* Multiple leaves in one image
* Unseen plant species
* Disease symptoms that differ from the training examples

Therefore, predictions should be treated as **model-based estimates rather than professional agricultural diagnosis**.

---

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

* Deep Learning
* Computer Vision
* CNN architecture
* TensorFlow & Keras
* Image preprocessing
* Data augmentation
* Multi-class image classification
* Model evaluation
* Prediction pipelines
* Data visualization
* Machine learning experimentation

---

## 👨‍💻 Author

**Divyansh Bansal**

GitHub: [@divyansh-3371](https://github.com/divyansh-3371)

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

**Repository:** [PlantDiseaseDetection](https://github.com/divyansh-3371/PlantDiseaseDetection)

---

## 📄 Disclaimer

This project is developed for **educational and research purposes**.

The predictions generated by the model should not be considered a replacement for professional agricultural or plant pathology advice.
