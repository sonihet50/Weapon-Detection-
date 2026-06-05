# SiteSecure VISION - Setup & Dependency Guide

This document provides a comprehensive guide to setting up the environment, installing dependencies, and running the various scripts in this repository.

SiteSecure VISION is a real-time surveillance and weapon detection system that utilizes deep learning (YOLO26n via Ultralytics) combined with a PyQt6-based dashboard for site monitoring and security alerts.

---

## 1. Prerequisites

- **Python Version**: Python 3.8, 3.9, 3.10, or 3.11 is highly recommended.
- **Hardware Acceleration (Highly Recommended)**: An NVIDIA GPU with CUDA support for real-time model inference at high frame rates.

---

## 2. Common Dependencies

These packages are required by almost all scripts in this repository, including the primary GUI, testing, and training tools.

| Dependency | Package Name | Installation Command | Purpose |
| :--- | :--- | :--- | :--- |
| **PyTorch** | `torch`, `torchvision` | *See installation options below* | Backend tensor computations and YOLO model execution. |
| **Ultralytics** | `ultralytics` | `pip install ultralytics` | YOLO object detection framework, prediction, and training utilities. |
| **OpenCV** | `opencv-python` | `pip install opencv-python` | Video streaming, video writer (alert clips), image resizing, and overlays. |
| **NumPy** | `numpy` | `pip install numpy` | Fast array operations (installed automatically with OpenCV/Ultralytics). |

### Installing PyTorch (CPU vs. GPU CUDA Support)

Depending on your hardware, install the appropriate version of PyTorch:

*   **Option A: CPU Only**
    ```bash
    pip install torch torchvision torchaudio
    ```

*   **Option B: NVIDIA GPU (CUDA 12.1 - Recommended)**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    *(For older hardware/drivers requiring CUDA 11.8, use `--index-url https://download.pytorch.org/whl/cu118`)*

---

## 3. Script-Specific Dependencies

Different scripts in the repository serve distinct functions and have additional requirements.

### 🖥️ Security Command Center GUI: [app.py](file:///D:/Weapon%20Detection/app.py)
The main application features a responsive multi-feed camera grid, interactive dashboard, live threat activity logs, and a video clip verification popup.

*   **Required Dependencies**:
    *   **PyQt6**: Modern desktop application GUI framework.
        ```bash
        pip install PyQt6
        ```
*   **Key Classes & Structures**:
    *   [Weapon_Detector](file:///D:/Weapon%20Detection/app.py#L22): Loads [weapon_detection.pt](file:///D:/Weapon%20Detection/model/weapon_detection.pt) into GPU memory.
    *   [Camera_Worker](file:///D:/Weapon%20Detection/app.py#L407): Captures frame streams asynchronously.
    *   [Inference_Worker](file:///D:/Weapon%20Detection/app.py#L459): Runs YOLO model predictions on batches of camera frames.
    *   [ClipRecorder](file:///D:/Weapon%20Detection/app.py#L86): Saves pre- and post-trigger frames to `alerts/` when an anomaly is detected.
    *   [AlarmVerificationDialog](file:///D:/Weapon%20Detection/app.py#L153): Popup for security officers to review recorded footage and confirm/dismiss alerts.
    *   [SecurityDashboard](file:///D:/Weapon%20Detection/app.py#L620): Main GUI layout and controller.
*   **Data Requirements**:
    *   Requires camera configurations listed in [cctv.csv](file:///D:/Weapon%20Detection/cctv.csv) (one camera source path/URL per line).
    *   Requires a pre-trained YOLO model at [model/weapon_detection.pt](file:///D:/Weapon%20Detection/model/weapon_detection.pt).
    *   Creates a directory named `alerts/` to store recorded threat clips and the local incident logs database `alarm_log.json`.

---

### 🏋️ Model Training: [scripts/train_v3.py](file:///D:/Weapon%20Detection/scripts/train_v3.py)
This script is used to train custom YOLO weights on a dataset using specialized CCTV data augmentations to simulate real-world CCTV noise, compression, and motion blur.

*   **Required Dependencies**:
    *   **Albumentations**: Powerful image augmentation library.
        ```bash
        pip install albumentations
        ```
*   **Key Functions**:
    *   [custom_cctv_init](file:///D:/Weapon%20Detection/scripts/train_v3.py#L7): Injects customized motion blur, JPEG compression, Gaussian noise, and contrast adjustments into the training pipeline.
    *   [train_production_model](file:///D:/Weapon%20Detection/scripts/train_v3.py#L36): Starts a YOLO26n training run.
*   **Data Requirements**:
    *   Requires a dataset path/configuration YAML file. The default configuration points to `D:\Weapon Detection\WD_v3\data.yaml` (update this path in the script if necessary).

---

### 🔍 Simple Inference Dashboard: [scripts/inf.py](file:///D:/Weapon%20Detection/scripts/inf.py)
A lightweight command-line script to test YOLO inference on multiple video feeds displayed in a basic OpenCV window grid.

*   **Required Dependencies**:
    *   *None* beyond the Common Dependencies.
*   **Key Classes**:
    *   [ThreadedCamera](file:///D:/Weapon%20Detection/scripts/inf.py#L8): Simple class to read and decode camera frames on a dedicated thread to prevent playback lag.

---

### 📊 Model Evaluation: [scripts/evaluate_v2.py](file:///D:/Weapon%20Detection/scripts/evaluate_v2.py)
A verification script to test a trained model's performance on the validation/test partition of the dataset, outputting key evaluation metrics (Precision, Recall, F1-Score, mAP50, mAP50-95).

*   **Required Dependencies**:
    *   *None* beyond the Common Dependencies.
*   **Key Functions**:
    *   [evaluate_model](file:///D:/Weapon%20Detection/scripts/evaluate_v2.py#L4): Runs validation inference and computes overall metrics.

---

## 4. Setup & Running Instructions

### Step 1: Create a Virtual Environment (Recommended)
Open your terminal (PowerShell or Command Prompt) in the project root directory and run:
```bash
python -m venv venv
```
Activate the virtual environment:
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### Step 2: Install PyTorch
Run the appropriate PyTorch command. For GPU acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Remaining Dependencies
Install all remaining common and specific packages:
```bash
pip install ultralytics opencv-python numpy PyQt6 albumentations
```

### Step 4: Run the Surveillance App
Start the main SiteSecure VISION surveillance command center dashboard:
```bash
python app.py
```
*(By default, it will load camera sources from `cctv.csv`. You can also pass a custom CSV file path as an argument: `python app.py custom_cctv.csv`)*
