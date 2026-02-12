import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse
TARGET_HEIGHT = 360
TARGET_WIDTH = 480

# --- CONFIGURATION ---
DEFAULT_MODEL = "onnx_models/model_face_v2.onnx"
DEFAULT_OUTPUT = "WIKI_OUTPUT"
CONF_THRESHOLD = 0.5

def process_single_image(image_path, output_dir, session, input_name, output_names):
    """Runs detection on one image and saves result to output_dir."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading: {image_path}")
        return

    h, w, _ = img.shape

    # Preprocess
    resized_image = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
    rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

    # Inference
    outputs = session.run(None, {input_name: tensor})
    output_dict = dict(zip(output_names, outputs))

    boxes   = output_dict.get("detection_boxes:0").squeeze()
    classes = output_dict.get("detection_classes:0").squeeze()
    scores  = output_dict.get("detection_scores:0").squeeze()

    face_count = 0
    for i in range(len(scores)):
        score = scores[i]
        if score < CONF_THRESHOLD:
            continue

        face_count += 1
        ymin, xmin, ymax, xmax = boxes[i]
        left   = int(xmin * w)
        top    = int(ymin * h)
        right  = int(xmax * w)
        bottom = int(ymax * h)
        cls    = int(classes[i])

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        
        label = f"Face: {score:.2f}"
        cv2.putText(img, label, (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, img)
    print(f" Processed {filename} | Faces: {face_count} | Saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Standalone Face Detection Demo (ONNX)")
    parser.add_argument("--input", required=True, help="Path to an image or directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model_face_v2.onnx")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Directory to save results")
    args = parser.parse_args()

    # Initialize ONNX Session
    if not os.path.exists(args.model):
        print(f"Model file not found at {args.model}")
        return

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [x.name for x in session.get_outputs()]

    # Handle Directory or Single File
    if os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(args.input, file)
                process_single_image(full_path, args.output, session, input_name, output_names)
    else:
        process_single_image(args.input, args.output, session, input_name, output_names)

if __name__ == "__main__":
    main()
