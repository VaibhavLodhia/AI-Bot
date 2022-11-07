import pyttsx3 as a
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import os
import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
from deepface import DeepFace
import webbrowser 
from test import bota , wishMe

from flask import Flask, render_template, Response

app=Flask(__name__)
camera = cv2.VideoCapture(0)


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def emotion():
    while True: 
        ret, img = camera.read()
        result = DeepFace.analyze(img , actions =['emotion'])
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,4)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,result['dominant_emotion'],(50,50),font,3,(0,0,255),2,cv2.LINE_4)
        # if(result['dominant_emotion']=='happy'):
        #     webbrowser.open("https://www.youtube.com/watch?v=A-sfd1J8yX4")
        # elif(result['dominant_emotion']=='sad'):
        #     webbrowser.open("https://www.youtube.com/watch?v=i_k3K772Zyk")
        # elif(result['dominant_emotion']=='angry'):
        #     webbrowser.open("https://www.youtube.com/watch?v=Ux-BoW8h6BA")
        # elif(result['dominant_emotion']=='energetic'):
        #     webbrowser.open("https://www.youtube.com/watch?v=n1oaPb_UTxs")
        # elif(result['dominant_emotion']=='neutral'):
        #     webbrowser.open("https://www.youtube.com/watch?v=g3M10O_eGV4")
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
            
        yield (b'--frame\r\n' 
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

def gen_frames():
    global name , encode 
    bradley_image = face_recognition.load_image_file("vaibhav.png")
    bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

# Create arrays of known face encodings and their names
# known_face_encodings = list(data['Image'])
# known_face_names = list(data['Name'])
    known_face_encodings = [
        bradley_face_encoding
    ]
    known_face_names = [
    "Vaibhav",
    "Bradley"
    ]
    while True:
        try:
            success, frame = camera.read()  # read the camera frame
        except Exception as e:
            success = False
            print(e)
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []



            
        
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
            
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    # dict1 = {name:matches}
                    # dict.update(dict1)
                    
                    
                # print(name)
                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # print(frame)
            # encode = face_recognition.face_encodings(frame,face_recognition.face_locations(frame))
            # cv2.imshow("frame" , frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


            
@app.route('/',methods = ['GET' , 'POST'] )
def home():
    print("h")
    h ="hi"
    return render_template('home.html',out=h)


# @app.route('/bot',methods = ['GET' , 'POST'])
# def bot():
#     wishMe()
#     bota()
    

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emo')
def emo():
    return Response(emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)