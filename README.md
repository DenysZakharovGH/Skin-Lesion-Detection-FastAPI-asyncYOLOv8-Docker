# 🩺 Skin Lesion Classification API
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)](https://github.com/ultralytics/ultralytics)

An AI-powered microservice built with **FastAPI** and **YOLOv8** to classify pigmented skin lesions into 7 diagnostic categories using the **HAM10000 dataset**.

## Overview

This project is a **Deep Learning API for Skin Lesion Classification**, now fully **containerized, async-ready, and optimized for concurrent requests**. It classifies pigmented skin lesions into 7 diagnostic categories, including **Melanoma**, using the **HAM10000 dataset**.

**Key Updates & Achievements:**

* Implemented **async inference** with `ThreadPoolExecutor` to handle multiple concurrent requests.
* **YOLOv8 model** is preloaded once at startup for faster inference.
* Added **rate limiting** support (e.g., 5 requests per minute) using `slowapi`.
* API now supports **German and Ukrainian languages** in descriptions.
* Logging has been optimized, and unnecessary Ultralytics startup prints are suppressed.
* Fully **Dockerized** for portable deployment.
* **Swagger UI** documentation is included for easy testing.

## Tech Stack

* **Model:** YOLOv8 (Ultralytics)
* **Backend:** FastAPI
* **Concurrency:** Async + ThreadPoolExecutor
* **Deployment:** Docker
* **Rate Limiting:** slowapi
* **Languages:** English, German, Ukrainian


### 🧠 Model Training
The model was trained using the **YOLOv8** (You Only Look Once) architecture by Ultralytics. While YOLO is famous for object detection, this project utilizes the **YOLOv8-cls** variant for high-accuracy image classification on medical datasets.



### 🧪 Supported Categories
The model identifies the following classes:
* **akiec**: Actinic keratoses and intraepithelial carcinoma
* **bcc**: Basal cell carcinoma
* **bkl**: Benign keratosis-like lesions
* **df**: Dermatofibroma
* **mel**: Melanoma
* **nv**: Melanocytic nevi
* **vasc**: Vascular lesions

---

## 📸 API Documentation
Once the service is running, you can access the interactive Swagger UI to test the model.

![API Documentation Screenshot](./docs/Screenshot%20from%202026-01-10%2022-02-28.png)

---

## 🛠️ Tech Stack
* **Backend:** FastAPI (Python)
* **ML Framework:** TensorFlow / Keras
* **Image Processing:** Pillow & NumPy
* **Containerization:** Docker
* **Web Server:** Uvicorn

---

## 💻 Getting Started

### Prerequisites
* [Docker](https://www.docker.com/get-started) installed.
* *OR* Python 3.9+ installed locally.



## 🛰️ API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check / Welcome message. |
| `POST` | `/predict` | Upload an image and receive a diagnosis. |
| `GET` | `/docs` | Swagger UI documentation. |

## JSON Request & Response

### Request

Endpoint: `/predict`
Method: `POST`
Content-Type: `multipart/form-data`

| Field  | Type         | Description                       |
| ------ | ------------ | --------------------------------- |
| `file` | `UploadFile` | The image file of the skin lesion |

Example using `curl`:

### Example Request (cURL)
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_image.jpg'
```

### Response

Content-Type: `application/json`

```json
{
  "filename": "test.jpg",
  "detections": [
    {"class": "NV", "confidence": 0.5934},
    {"class": "MEL", "confidence": 0.3099}
  ],
  "annotated_image": "<base64-encoded-image>"
}
```

* `filename`: original uploaded file name.
* `detections`: list of detected lesions with class and confidence score.
* `annotated_image`: base64-encoded JPEG image with bounding boxes drawn.

## Usage

1. **Clone the repository**

```bash
git clone https://github.com/DenysZakharovGH/Classification-of-pigmented-skin-lesions.git
```

2. **Build Docker image**

```bash
docker build -t skin-lesion-classifier .
```

3. **Run the container**

```bash
docker run -p 8000:8000 skin-lesion-classifier
```

4. **Test API**

* Open Swagger UI at `http://localhost:8000/docs`
* Or send POST requests to `/predict`.

## Notes

* The first request may take slightly longer due to model warm-up.
* For production and high concurrency, consider multiple Uvicorn workers and/or GPU inference.
* Rate limiting ensures fair usage and prevents overload.

## GitHub
[Project Repository](https://github.com/DenysZakharovGH/Classification-of-pigmented-skin-lesions)


