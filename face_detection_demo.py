import onnxruntime as ort
import numpy as np
import cv2
import os

# --- CONFIGURATION ---
MODEL_PATH = "onnx_models/model_face_v2.onnx"
IMAGE_DIR = r"C:\Users\tthaker\Downloads\ruo_test_images\Batch-1-0000162-Francesco-Totti_zipped\Batch-1-0000162-Francesco-Totti\WIKI"
OUTPUT_DIR = r"C:\Users\tthaker\Downloads\ruo_test_images\Batch-1-0000162-Francesco-Totti_zipped\Batch-1-0000162-Francesco-Totti\WIKI_OUTPUT"
CONFIDENCE_THRESHOLD = 0.5
TARGET_HEIGHT = 360
TARGET_WIDTH = 480

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load ONNX model once
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_names = [x.name for x in session.get_outputs()]

def process_image(image_path):
    """Reads, preprocesses, runs inference, draws boxes, saves result to OUTPUT_DIR."""
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w, _ = img.shape
    resized_image = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
    rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

    # Run inference
    outputs = session.run(None, {input_name: tensor})
    output_dict = dict(zip(output_names, outputs))

    boxes   = output_dict.get("detection_boxes:0").squeeze()
    classes = output_dict.get("detection_classes:0").squeeze()
    scores  = output_dict.get("detection_scores:0").squeeze()

    if boxes is None or scores is None:
        return

    # Draw detections (The loop handles multiple faces automatically)
    face_count = 0
    for i in range(len(scores)):
        score = scores[i]
        if score < CONFIDENCE_THRESHOLD:
            continue

        face_count += 1
        ymin, xmin, ymax, xmax = boxes[i]
        
        # Scale normalized coordinates to pixel values
        left   = int(xmin * w)
        top    = int(ymin * h)
        right  = int(xmax * w)
        bottom = int(ymax * h)
        cls    = int(classes[i])

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Face {score:.2f}",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    save_path = os.path.join(OUTPUT_DIR, f"{name}_detected{ext}")
    
    cv2.imwrite(save_path, img)
    print(f"Processed: {filename} | Faces found: {face_count}")


print(f"Starting inference on {IMAGE_DIR}...")
for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        process_image(os.path.join(IMAGE_DIR, file))

print(f"\nCompleted! Results saved to: {OUTPUT_DIR}")
