from turtle import up
import cv2
import mediapipe as mp
import requests

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

left_eye_under_list = [263, 249, 390, 373, 374, 380, 381, 382, 362]
left_eye_upper_list = [263, 466, 388, 387, 386, 385, 384, 398, 362]

cap = cv2.VideoCapture(0)       
with mp_face_mesh.FaceMesh(
    max_num_faces=1,                    # 얼굴 갯수 인식할 수 있는 갯수
    refine_landmarks=True,
    min_detection_confidence=0.5,       # 얼굴을 얼마나 정확히 찾을 것인지(낮을수록 아닌거 같아도 찾아줌)
    min_tracking_confidence=0.5) as face_mesh: # 트래커가 있어야 새로운 얼굴이 들어와도 각각 인식해서 얼굴을 찾아준다
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_copy = image.copy()
    image_rows, image_cols, _ = image_copy.shape

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:

        eye_colsed_list = []

        for i in range(len(left_eye_upper_list)):

            upper_landmark = face_landmarks.landmark[left_eye_upper_list[i]]
            under_landmark = face_landmarks.landmark[left_eye_under_list[i]]

            upper_landmark_px = mp_drawing._normalized_to_pixel_coordinates(
                upper_landmark.x, upper_landmark.y, image_cols, image_rows)

            under_landmark_px = mp_drawing._normalized_to_pixel_coordinates(
                under_landmark.x, under_landmark.y, image_cols, image_rows)

            # [0]이 x 좌표,  [1] 이 y 좌표
            eye_colsed_list.append(under_landmark_px[1] - upper_landmark_px[1])
            # eye_colsed_list.append(under_landmark.y - upper_landmark.y)

        closed = sum(eye_colsed_list)/len(eye_colsed_list)
        print(closed)

        if closed > 2:
            continue

        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        if False==result:
            continue

        url = 'http://127.0.0.1:8000/uploadimage'
        files = {'file': encimg}

        res = requests.post(url, files=files)
        if res.status_code != 200:
            continue

        print(res.content)        
        age_list = res.json()["ages"]
        if len(age_list) > 0:
            text=age_list[0]
            org=(50,100)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, org, font, 1, (255,0,0), 2)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()