import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

def identify_face(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for minNeighbors in range(0, 50, 10):
        # Itera até encontrar apenas um rosto ou até chegar no limite dos vizinhos
        image_faces = face_cascade.detectMultiScale(image_gray, 1.05, minNeighbors)
        if len(image_faces) == 1:
            break

    if len(image_faces) == 0:
        # Itera até encontrar apenas dois rostos
        for minNeighbors in range(0, 50, 10):
            image_faces = face_cascade.detectMultiScale(image_gray, 1.05, minNeighbors)
            if len(image_faces) == 2:
                break

    if len(image_faces) > 1:
        curr_face = 0
        best_face = [[0, 0, 0, 0]]

        for (x,y,w,h) in image_faces:
            if (w**2 + h**2)**0.5 > curr_face:
                best_face[0] = [x, y, w, h]
                curr_face = (w**2 + h**2)**0.5

        image_faces = best_face

    return image_faces

def detect_eyes(face):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_eyes = eye_cascade.detectMultiScale(face_gray)

    if len(face_eyes) > 2:
        true_eyes = [[0, 0, 0, 0], [0, 0, 0, 0]]
        dim_eyes = [0, 0]

        for face_eye in face_eyes:
            curr_eye = int(((face_eye[2])**2 + (face_eye[3])**2)**(0.5))

            if curr_eye > max(dim_eyes):
                dim_eyes[dim_eyes.index(min(dim_eyes))] = dim_eyes[dim_eyes.index(max(dim_eyes))]
                dim_eyes[dim_eyes.index(max(dim_eyes))] = curr_eye
                true_eyes[1] = true_eyes[0]
                true_eyes[0] = face_eye

            elif curr_eye > min(dim_eyes):
                dim_eyes[dim_eyes.index(min(dim_eyes))] = curr_eye
                true_eyes[1] = face_eye

    else:
        true_eyes = face_eyes

    return true_eyes
