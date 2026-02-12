

#  Detection along with cropping & alignment using second detector

This repository implements the **face detection → landmark prediction → alignment & cropping** pipeline using:

* **Tensorflow based converted ONNX face detector**
* **ONNX facial landmark model**

The goal is to **reproduce the production alignment/cropping pipeline** and verify that the outputs match what production generates.

We also have a separate script if we just want to perform detection on a set of images named after ```face_detection_demo.py```.
---

##  Directory Structure

```
.
├── detect_crop_align_tf.py          # Main pipeline script
├── requirements.txt                 # Dependencies
├── face_detection_demo.py           # Just run face detection
│
├── onnx_models/
│   ├── model_face_v2.onnx           # Detector 2 ONNX model
│   └── landmark_model.onnx          # ONNX landmark model
│
└── crop_after_detection_alignment/  # Generated aligned & cropped faces
```

---

##  What This Repo Does

### **1. Face Detection (Detector 2 ONNX)**

* Loads and runs the Detector-2 ONNX model.
* Extracts bounding boxes and scores.
* Applies preprocessing and postprocessing consistent with production.

### **2. Landmark Estimation (ONNX Landmarks)**

* Runs the ONNX landmark model.
* Validates that landmark indices and ordering match production behavior.

### **3. Alignment + Cropping**

* Tries to reproduce the **exact alignment logic** used in production:

  * Reference landmark mapping
  * Similarity transform
  * Output crop size & padding
* Saves the aligned + cropped face images.



---

## 🛠 Installation

```bash
pip install -r requirements.txt
```

---

## Usage

1. Run the full pipeline on an input image:

```bash
python detect_crop_align_tf.py --image PATH_TO_IMAGE
```

Output crops will be saved under:

```
crop_after_detection_alignment/
```

2. Run only detection on images present in a given folder:

Use `face_detection_demo.py` to quickly verify detection performance without alignment.

**Run on a single image:**
```bash
python face_detection_demo.py --input path/to/image_folder
```
---



