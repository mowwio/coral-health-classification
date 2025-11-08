# Coral Reef Health Classification (Deep Learning API & Web App)

This is a project that implements a full-stack application to classify coral reef health.

The system uses a Deep Learning (CNN) model to analyze an uploaded image and classify it as **'Healthy'** or **'Bleached'**. The project consists of:
1.  A **Jupyter Notebook (`coral.ipynb`)** for data analysis, model training (using ResNet50V2), and evaluation.
2.  A **Backend API (`backend_api/`)** built with FastAPI, which serves the trained model (`.h5`) and logs predictions.
3.  A **Frontend UI (`app.py`)** built with Streamlit, providing a simple web interface for users.

---

## 🚀 Key Results

The model was trained on 9,662 images and validated on 463 images. The final model (`coral_model_best.h5`) achieved the following performance on the independent test set:

* **Test Accuracy:** **92.22%**
* **Precision (Weighted):** 0.92
* **Recall (Weighted):** 0.92

### Evaluation Plots
The model trained well with no significant overfitting, as shown by the training/validation graphs and the confusion matrix.

## 🛠️ Tech Stack

* **Model:** TensorFlow, Keras (ResNet50V2), Scikit-learn
* **Backend:** FastAPI, Uvicorn, SQLite
* **Frontend:** Streamlit, Requests, Pillow
---
## 🏃 How to Run This Project

This application requires **two terminals** to run simultaneously: one for the backend API and one for the frontend UI.

### 1. (Terminal 1) Run the Backend API

1.  Open a new terminal.
2.  Navigate into the `backend_api` folder:
    ```bash
    cd coral-health-classification/backend_api
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  Start the backend server:
    ```bash
    python -m uvicorn main:app --reload
    ```
5.  The API is now running at `http://127.0.0.1:8000`. **Leave this terminal running.**

### 2. (Terminal 2) Run the Frontend UI

1.  Open a **second, new terminal**.
2.  Navigate to the **main project folder** (NOT the backend folder):
    ```bash
    cd coral-health-classification
    ```
3.  Install the required libraries (if you haven't already):
    ```bash
    pip install streamlit requests pillow
    ```
4.  Start the Streamlit app:
    ```bash
    python -m streamlit run app.py
    ```
5.  Your web browser will automatically open to the app. You can now upload an image to get a prediction.

<img width="577" height="729" alt="image" src="https://github.com/user-attachments/assets/6a8b89c1-75be-4183-bf43-a9321a55b018" />
