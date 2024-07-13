import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# dataset
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# webcam
cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

fig, ax = plt.subplots()
im = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))  # size
ax.axis('off')

def update_frame(i):
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        return im

    # grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    im.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return im

ani = animation.FuncAnimation(fig, update_frame, interval=1)
plt.show()

cap.release()
