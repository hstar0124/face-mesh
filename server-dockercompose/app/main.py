from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import onnxruntime as ort
from ultraface.dependencies.box_utils import predict
face_detector_onnx = "ultraface/models/version-RFB-640.onnx"
face_detector = ort.InferenceSession(face_detector_onnx, providers=['CPUExecutionProvider'])


# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

# crop image
def cropImage(image, box):
    num = image[box[1]:box[3], box[0]:box[2]]
    return num

# face detection method
def faceDetector(orig_image, threshold = 0.7):
    # preprocessing
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    # inference (forward)
    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})


    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs
# ------------------------------------------------------------------------------------------------------------------------------------------------

# Face gender classification using GoogleNet onnx model
gender_classifier_onnx = "models/gender_googlenet.onnx"

gender_classifier = ort.InferenceSession(gender_classifier_onnx, providers=['CPUExecutionProvider'])
genderList=['Male','Female']

# gender classification method
def genderClassifier(orig_image):

    # preprocessing
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_mean = np.array([104, 117, 123])
    image = image - image_mean
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = gender_classifier.get_inputs()[0].name
    genders = gender_classifier.run(None, {input_name: image})


    gender = genderList[genders[0].argmax()]
    return gender

# ------------------------------------------------------------------------------------------------------------------------------------------------

# Face age classification using GoogleNet onnx model
age_classifier_onnx = "models/age_googlenet.onnx"

age_classifier = ort.InferenceSession(age_classifier_onnx, providers=['CPUExecutionProvider'])
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# age classification method
def ageClassifier(orig_image):
    
    # preprocessing
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_mean = np.array([104, 117, 123])
    image = image - image_mean
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = age_classifier.get_inputs()[0].name
    ages = age_classifier.run(None, {input_name: image})
    
    age = ageList[ages[0].argmax()]
    return age

app = FastAPI()

# 동기식은 먼저 들어온 요청이 처리되어야 이후 요청을 처리할 수 있다.
@app.get("/")
def read_root():
    return {"Hello": "World"}

# async 가 있으면 비동기 처리를 할 수 있다.
# bytes로 받으면 OS에서 이미 파일을 다받아서 처리를 하기 때문에 사이즈를 알 수 있다.
@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}


# await를 통해 비동기 처리를 진행할 수 있다.
@app.post("/uploadimage/")
async def create_upload_file(file: UploadFile):
    file_buf = await file.read()

    # 이미지 리더들은 바이너리를 읽어야 하기 때문에 변환이 한번 더 필요하다.    
    # 넘어온 데이터는 String 이기 때문
    encoded_img = np.fromstring(file_buf, dtype = np.uint8)
    image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    boxes, labels, probs = faceDetector(image)

    ages = []
    genders = []
    for i in range(boxes.shape[0]):
        box = scale(boxes[i, :])
        cropped = cropImage(image, box)
        if cropped is None:
            continue
        gender = genderClassifier(cropped)
        age = ageClassifier(cropped)
        ages.append(age)
        genders.append(gender)        
    
    return {
        "file_name": file.filename, 
        "image_shape": image.shape,
        "ages": ages,
        "genders": genders
    }