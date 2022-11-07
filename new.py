# from test import bota, wishMe
from flask import Flask, render_template, Response
import pyttsx3 as a
import datetime
import speech_recognition as sr
app=Flask(__name__)

engine = a.init('sapi5')
voices = engine.getProperty('voices')
print(voices)

engine.setProperty('voice', voices[0].id)


engine.say("what can i do for you")
engine.runAndWait()
if engine._inLoop:
    engine.endLoop()

@app.route('/',methods = ['GET' , 'POST'] )
def home():

    return render_template('home.html')
@app.route('/bot',methods = ['GET' , 'POST'] )
def bot():
    print("y")
    # wishMe()
    # bota()
    # return render_template('home.html')


if __name__=='__main__':
    app.run(debug=True)