#import required libraries

import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from time import time
from torchvision import transforms

import onnxruntime as ort
from onnxruntime_extensions import get_library_path
import argparse

parser= argparse.ArgumentParser('parser for face detector inference')

parser.add_argument('--image_folder', type= str, default='test_data',
                    required= False,  help= 'provide image path for face detection')

parser.add_argument('--detector', type= str, default= 'onnx_models/model_face_v2.onnx',
                    required= False, help= 'provide onnx face detector path')

parser.add_argument('--landmark', type= str, default= 'onnx_models/landmark_model.onnx',
                    required= False, help= 'provide onnx landmark detector path')

parser.add_argument('--output_folder', type= str, default= 'crop_after_detection_alignment',
                    required= False, help= 'provide onnx landmark detector path')


args= parser.parse_args()

# Load and execute the ONNX model with the custom operator
so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())  # Register the custom operator library

# Create an inference session for detection model
session = ort.InferenceSession(args.detector, sess_options=so, providers=["CUDAExecutionProvider"])

# Create an inference session for landmark model
session_lm    = ort.InferenceSession(args.landmark,sess_options=so, providers=["CUDAExecutionProvider"])

transform = transforms.Compose([
            # transforms.Resize((360,480)),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


# def postprocess(out):
#     # Convert list to numpy array if it's not already
#     out = np.array(out)
#     print(out.shape)
#     box  = out[:,3:7]
#     cls  = out[:,1]
#     conf = out[:,2]
#     return (box, conf, cls)

# def filter_bbox(bboxes, threshold= 0.8):
#     threshold= 0.3
#     filtered_bboxes= []
    
#     for bbox in bboxes:
#         conf= bbox[2]
#         if conf > threshold:
#             filtered_bboxes.append(bbox)
            
#     return filtered_bboxes

def filter_bbox(boxes, scores, classes, threshold=0.5):
    """
    Filter detections by score threshold.

    Args:
        boxes: np.array with shape (1, N, 4) or (N, 4) in normalized [ymin, xmin, ymax, xmax]
        scores: np.array with shape (1, N) or (N,)
        classes: np.array with shape (1, N) or (N,)
        threshold: float, keep detections with score > threshold

    Returns:
        (boxes_flt, scores_flt, classes_flt)
          boxes_flt: (M,4) normalized [ymin, xmin, ymax, xmax]
          scores_flt: (M,)
          classes_flt: (M,)
    """
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    # remove possible batch dim
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if classes.ndim == 2 and classes.shape[0] == 1:
        classes = classes[0]

    # select indices above threshold
    keep = np.where(scores > threshold)[0]
    if keep.size == 0:
        return np.zeros((0,4), dtype=boxes.dtype), np.zeros((0,), dtype=scores.dtype), np.zeros((0,), dtype=classes.dtype)

    return boxes[keep], scores[keep], classes[keep]

def detect(image_array):
         
    input_name = session.get_inputs()[0].name
    # We run inference on the processed tensor
    start= time()
    outputs = session.run(None, {input_name: image_array})
    end= time()
    # 3. Process Outputs
    output_names = [x.name for x in session.get_outputs()]
    output_dict = dict(zip(output_names, outputs))

    boxes = output_dict.get("detection_boxes:0")   # Shape: [1, Num, 4]
    classes = output_dict.get("detection_classes:0") # Shape: [1, Num]
    scores = output_dict.get("detection_scores:0")   # Shape: [1, Num]
    print("boxes: ",boxes)
    
    box, conf, clss = filter_bbox(boxes, scores, classes, threshold=0.2) 
    print(f"The bounding box: {box}")
    return box, conf, end-start

# def detect(image_array):
#     inputs     = {session.get_inputs()[0].name:image_array}
#     start= time()     
#     output = session.run(["detection_out"], inputs)
#     end= time()
#     filtered_output= filter_bbox(output[0]) 
#     if(len(filtered_output)==0):
#             return None, None, None
#     box, conf, cls = postprocess(filtered_output)
#     return box, conf, end-start


TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])


TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]


def align(imgDim1,imgDim2,rgbImg, landmarks):
        assert rgbImg is not None
        assert landmarks is not None

        # landmarks = [[77, 113], [109, 117], [85, 190]]
        print(f"landmark in align: {landmarks}")
        
        npLandmarks = np.float32(landmarks)    
        landmarkIndices = INNER_EYES_AND_BOTTOM_LIP
        npLandmarkIndices = np.array(landmarkIndices)

        T=MINMAX_TEMPLATE[npLandmarkIndices]
        T[:,0]=imgDim1*T[:,0]
        T[:,1]=imgDim2*T[:,1]
        H = cv2.getAffineTransform(npLandmarks, imgDim1 * MINMAX_TEMPLATE[npLandmarkIndices])
        thumbnail = cv2.warpAffine(rgbImg, H, (imgDim1, imgDim2))
        
        # Transform landmarks to the aligned image space
        transformed_landmarks = cv2.transform(np.array([npLandmarks]), H)[0]
        print(f"transformed landmark: {transformed_landmarks}")
        return thumbnail, transformed_landmarks


def detect_landmark(ori_img_path, output_folder='crop_before_alignment'):
    filename = os.path.basename(ori_img_path).split('.')[0]
    img_bgr = cv2.imread(ori_img_path)
    image_h, image_w, _ = img_bgr.shape
    print(f'height & width: {image_h}, {image_w}')
    
    # # Convert BGR to RGB (since OpenCV loads images in BGR format)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # # Convert the NumPy array to a PIL Image
    # img_pil = Image.fromarray(img_rgb)
    # # Apply transformations (assuming `transform` is a torchvision transform pipeline)
    # transformed_image = transform(img_pil)
    # # Convert transformed image to NumPy array and add batch dimension
    # image_array = np.expand_dims(np.array(transformed_image), axis=0)
    # # Convert to float32
    # image_array = image_array.astype(np.float32)

    # Convert BGR to RGB (TensorFlow models usually expect RGB)
    input_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Add Batch Dimension: (360, 480, 3) -> (1, 360, 480, 3)
    input_tensor = np.expand_dims(input_image, axis=0)

    # Ensure type is uint8 (or float32 if your specific model requires normalization)
    input_tensor = input_tensor.astype(np.uint8)
    
          
    boxes,score, time_taken    = detect(input_tensor)
    
    if boxes is None:
        return None, None, None, None, None

    lm= []
    tt_landmark= 0
    print(f"boxes shape: {boxes.shape}")
    
    for i, bbox in enumerate(boxes):
        print("bbox: ", bbox)
        # If coordinates are normalized, convert to absolute pixel values
        x_min, y_min, x_max, y_max = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        # x_min_pixel = int(x_min * image_w) if x_min < 1  else int(x_min)
        # y_min_pixel = int(y_min * image_h) if y_min < 1  else int(y_min)
        # x_max_pixel = int(x_max * image_w) if x_max <= 1 else int(x_max)
        # y_max_pixel = int(y_max * image_h) if y_max <= 1 else int(y_max)

        left = int(x_min * image_w)
        right = int(x_max * image_w)
        top = int(y_min * image_h)
        bottom = int(y_max * image_h)

        print(left, right, top, bottom)


        # Ensure coordinates are within valid image dimensions
        xmin = max(0, min(left, image_w))
        ymin = max(0, min(top, image_h))
        xmax = max(0, min(right, image_w))
        ymax = max(0, min(bottom, image_h))

        h, w   = bottom - top , right - left     #height & width of cropped image
        scaler = np.array([h, w])
        print(f" bbox values")
        print("ymin", top)
        print("xmin", left)
        print("height", h)
        print("width", w)
        if left >= right or top >= bottom:
            print(f"Invalid box skipped: {bbox}")
            continue

        crop = input_image[ymin:ymax, xmin:xmax]
        # crop = input_image[74:232, 34:168]
        # crop = input_image[ymin-10:ymax+10, xmin-10:xmax+10]
        if crop.size == 0:
            print("Empty crop skipped.")
            continue
            
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        # Save the image with landmarks
        output_path = os.path.join(output_folder, filename) + '_' + str(i) + '.jpg'
        cv2.imwrite(output_path, crop_bgr)
        
        if crop.size == 0:
            continue  # Skip empty crops

        crop_resized = cv2.resize(crop, (64, 64))
        crop_resized = np.expand_dims(crop_resized, axis=0)  # Add batch dim
        # Ensure the resized image is in uint8 format
        crop_resized = crop_resized.astype(np.uint8)
        
        inputs     = {session_lm.get_inputs()[0].name:crop_resized}
        start= time()
        ort_outputs= session_lm.run(None, inputs)
        end= time()
        tt_landmark += (end-start)
        keypoints  = np.array(ort_outputs).reshape(98,2)
        np.savetxt("keypoints.txt", keypoints)
        landmarks  = (keypoints * scaler) + (top, left)
        
        
        landmarks_xy = []
        lm_cnt=0

        np.savetxt("landmarks_test.txt", np.array(landmarks))
        for y , x in landmarks:
                lm_cnt += 1
                if(lm_cnt==64 or lm_cnt==68 or lm_cnt==85):
                # if(lm_cnt==39 or lm_cnt==42 or lm_cnt==57):
                    landmarks_xy.append([x , y])
        lm.append(landmarks_xy)
        print(f"The landmark is: {lm}")

    return boxes, score, lm, time_taken, tt_landmark



# output_folder will contain images after detection, cropping & alignment

def detect_align_crop(input_folder, output_folder='crop_after_detection_alignment'):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder created: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")

    avg_inf= 0
    avg_inf_landmark= 0
    count= 0
    
    for directory, folder, files in os.walk(input_folder):
        for file in files:
            image_path = os.path.join(directory, file)

            # --- Create mirrored directory path inside output_folder ---
            rel_path = os.path.relpath(directory, input_folder)
            save_dir = os.path.join(output_folder, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            bboxs, scores, landmarks, time_taken, time_taken_landmark = detect_landmark(image_path, output_folder=output_folder) 
            if time_taken is not None:
                count += 1
                avg_inf += time_taken
                avg_inf_landmark += time_taken_landmark

            if bboxs is None:
                continue


            # for i in range(len(landmarks)):
            #     filename   = os.path.basename(image_path).split('.')[0]
            #     output_path= os.path.join(output_folder, filename) + '_' + str(i) + '.jpg'
            #     aligned_img, transformed_landmarks = align(112, 112, image_rgb, landmarks[i])
            #     cv2.imwrite(output_path, cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR))
            # --- Save aligned images in mirrored directory ---
            for i in range(len(landmarks)):
                filename = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(save_dir, f"{filename}_{i}.jpg")

                aligned_img, transformed_landmarks = align(
                    112, 112, image_rgb, landmarks[i]
                )
                cv2.imwrite(output_path, cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR))

    print(f'average inference time taken for detection: {avg_inf/count}')
    print(f'average inference time taken for landmark detection: {avg_inf_landmark/count}')


detect_align_crop(args.image_folder, args.output_folder)


