from keras.models import load_model
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load model and labels
model = load_model("keras_model.h5", compile=False)
with open("labels.txt", "r") as f:
    class_names = open("labels.txt", "r").readlines()
# Start video
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Webcam not detected.")
    exit()

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    resized = cv2.resize(frame, (224, 224))
    image = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    label = class_name[1:3:1]

    cv2.putText(frame, label, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam Image", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()