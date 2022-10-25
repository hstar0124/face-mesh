from turtle import up
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

left_eye_under_list = [263, 249, 390, 373, 374, 380, 381, 382, 362]
left_eye_upper_list = [263, 466, 388, 387, 386, 385, 384, 398, 362]

# For static images:
# IMAGE_FILES = ['iu.png']
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# with mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5) as face_mesh:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     # Convert the BGR image to RGB before processing.
#     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # left_eye_list = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
#     #                            (374, 380), (380, 381), (381, 382), (382, 362),
#     #                            (263, 466), (466, 388), (388, 387), (387, 386),
#     #                            (386, 385), (385, 384), (384, 398), (398, 362)])
#     # left_eye_under_list = [263, 249, 390, 373, 374, 380, 381, 382, 362]
#     # left_eye_upper_list = [263, 466, 388, 387, 386, 385, 384, 398, 362]

#     # FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
#     #                             (145, 153), (153, 154), (154, 155), (155, 133),
#     #                             (33, 246), (246, 161), (161, 160), (160, 159),
#     #                             (159, 158), (158, 157), (157, 173), (173, 133)])

#     # right_eye_under_list = [33,   7, 163, 144, 145, 153, 154, 155, 133]
#     # right_eye_upper_list = [33, 246, 161, 160, 159, 158, 157, 173, 133]

#     # Print and draw face mesh landmarks on the image.
#     if not results.multi_face_landmarks:
#       continue
#     annotated_image = image.copy()
#     image_rows, image_cols, _ = annotated_image.shape

#     for face_landmarks in results.multi_face_landmarks:

#     #   print(face_landmarks.landmark)    # -1.0 ~ 1.0 값 범위에서 출력이 되는데 실제 응용서비스에서 사용할수 있게 해주어야 한다.

#       for pt in left_eye_upper_list:
#           landmark = face_landmarks.landmark[pt]
#           landmark_px = mp_drawing._normalized_to_pixel_coordinates(
#                           landmark.x, landmark.y, image_cols, image_rows)

#           cv2.circle(annotated_image, landmark_px, 1, (255, 0, 255), drawing_spec.thickness)

#       for pt in left_eye_under_list:
#           landmark = face_landmarks.landmark[pt]
#           landmark_px = mp_drawing._normalized_to_pixel_coordinates(
#                           landmark.x, landmark.y, image_cols, image_rows)

#           cv2.circle(annotated_image, landmark_px, 1, (255, 0, 255), drawing_spec.thickness)


#     #   for pt in right_eye_under_list:
#     #       landmark = face_landmarks.landmark[pt]
#     #       landmark_px = mp_drawing._normalized_to_pixel_coordinates(
#     #                       landmark.x, landmark.y, image_cols, image_rows)

#     #       cv2.circle(annotated_image, landmark_px, 1, (244, 218, 0), drawing_spec.thickness)

#     #   for pt in right_eye_upper_list:
#     #       landmark = face_landmarks.landmark[pt]
#     #       landmark_px = mp_drawing._normalized_to_pixel_coordinates(
#     #                       landmark.x, landmark.y, image_cols, image_rows)

#     #       cv2.circle(annotated_image, landmark_px, 1, (244, 218, 0), drawing_spec.thickness)



#     #   mp_drawing.draw_landmarks(
#     #       image=annotated_image,
#     #       landmark_list=face_landmarks,
#     #       connections=mp_face_mesh.FACEMESH_TESSELATION,
#     #       landmark_drawing_spec=None,
#     #       connection_drawing_spec=mp_drawing_styles
#     #       .get_default_face_mesh_tesselation_style())
#     #   mp_drawing.draw_landmarks(
#     #       image=annotated_image,
#     #       landmark_list=face_landmarks,
#     #       connections=mp_face_mesh.FACEMESH_CONTOURS,
#     #       landmark_drawing_spec=None,
#     #       connection_drawing_spec=mp_drawing_styles
#     #       .get_default_face_mesh_contours_style())
#     #   mp_drawing.draw_landmarks(
#     #       image=annotated_image,
#     #       landmark_list=face_landmarks,
#     #       connections=mp_face_mesh.FACEMESH_IRISES,
#     #       landmark_drawing_spec=None,
#     #       connection_drawing_spec=mp_drawing_styles
#     #       .get_default_face_mesh_iris_connections_style())

#     cv2.imwrite('./annotated_image' + str(idx) + '.png', annotated_image)


# For webcam input:drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
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

        if closed > 4:
            continue

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()