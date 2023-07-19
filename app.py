# Import the Flask class from the flask module
from flask import Flask, render_template, Response, session
from twilio.rest import Client
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from twilio.rest import Client
import numpy as np
import imutils
import cv2
import time
import json

acc_sid = "AC6c198a4e05e3991b1fe583c1bb5d5e26"
auth_program = "bf988110b7c39c73677adea121d326ee"
twilio_num = "+14847598067"
target_num="+919344994959"

delayForMessage=10






# Create an instance of the Flask class
app = Flask(__name__)
app.secret_key = "your-secret-key"
camera = None


# Load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model from disk
maskNet = load_model("mask_detector.model")


hasMask = False
lastMessageSentTime=0













# Register a route
@app.route('/')
def home():
    if 'camera_open' in session and session['camera_open']:
        session['camera_open'] = False
        release_camera()

    return render_template('index.html')




def send_message():
    global acc_sid,auth_program,twilio_num,target_num,lastMessageSentTime,delayForMessage
    now=time.time()
    if(now-lastMessageSentTime<delayForMessage):
        return
    else:
        lastMessageSentTime=now
        client=Client(acc_sid,auth_program)
        message=client.messages.create(
            body="wassup",
            from_=twilio_num,
            to=target_num
        )
        print(message.sid)
        print("message sent")






def gen():
    global camera, hasMask

    while True:
        success, frame = camera.read()
        frame = imutils.resize(frame, width=400)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            hasMask = mask > withoutMask

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        if(hasMask):
            print("has mask")
        else:
            print("no mask")
            send_message()

        ret, jpeg = cv2.imencode('.jpg', frame)
        augmentedFrame = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + augmentedFrame + b'\r\n\r\n')


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
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
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None




@app.route('/video_feed')
def video_feed():
    global camera
    if 'camera_open' not in session or not session['camera_open']:
        session['camera_open'] = True
        camera = cv2.VideoCapture(0)
    return Response(gen(), mimetype='multipart/x-mixed-replace;boundary=frame')



@app.route('/stop_video_feed')
def stop_video_feed():
    if 'camera_open' in session and session['camera_open']:
        session['camera_open'] = False
        release_camera()
    return Response(json.dumps({'success': True}), mimetype='application/json')



@app.route('/get_mask_status')
def get_mask_result():
    global hasMask
    return Response(json.dumps({'hasMask': bool(hasMask)}), mimetype='application/json')



@app.route('/ImageStream')
def ImageStream():
    if 'camera_open' in session and session['camera_open']:
        session['camera_open'] = False
        release_camera()

    return render_template('RealtimeImage.html')


# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 
