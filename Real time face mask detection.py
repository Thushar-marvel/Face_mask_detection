
from keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

def detect_and_predict_mask(frame, faceNet, maskNet):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400),
        (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")


            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (256, 256))


            face = img_to_array(face)
            face=face/255

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:

        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)


    return (locs, preds)

mask_detect = load_model("face_model97.78.h5")


prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)



capture = cv2.VideoCapture(0)



while True:
    # Start capturing frames
    ret, capturing = capture.read()


    resize_frame = cv2.resize(capturing,(400,400),interpolation=cv2.INTER_AREA)

    (locs, preds) = detect_and_predict_mask(resize_frame, faceNet, mask_detect)
    blob = cv2.dnn.blobFromImage(resize_frame, 1, (400, 400), (100, 177, 123))

    for (box, pred) in zip(locs, preds):

        (startX, startY, endX, endY) = box
        ( mask,withoutMask) = pred


        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(resize_frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(resize_frame, (startX, startY), (endX, endY), color, 2)

    #
    cv2.imshow("Real-time Detection", resize_frame)


    c = cv2.waitKey(1)
    if c == 27:
        break
result.release()
capture.release()
# Close all windows
cv2.destroyAllWindows()
