import pyttsx3 as a
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import os
import smtplib
engine = a.init('sapi5')
voices = engine.getProperty('voices')
print(voices)

engine.setProperty('voice', voices[0].id)


def speak(audio):
    engine.say(audio)
    # engine.runAndWait()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("good morning sir")
    elif hour >= 12 and hour < 18:
        speak("good afternoon sir")
    else:
        speak("good evening sir")

    speak("hello sir i am gogo , how may i help you?")


def take():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
        try:
            print("recognizing...")
            query = r.recognize_google(audio)
            print(query)
        except Exception as e:
            print(e)
            print("say that again please")
            return "None"
        return query



def bota():
    while True:
        query = take().lower()
        if 'wikipedia' in query:
            speak('Searching wikipedia')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query , sentences=1)
            speak("according to wikipedia")
            print(results)
            speak(results)
        elif 'open youtube' in query:
            speak("sure sir")
            webbrowser.open("youtube.com")
            if engine._inLoop:
                engine.endLoop()

        elif 'open google' in query:
            speak("sure sir")
            webbrowser.open("google.com")

        elif 'open stackoverflow' in query:
            speak("sure sir")
            webbrowser.open("stackoverflow.com")

        elif 'play music' in query:
            speak("sure sir")
            music_dir = 'D:\\songs'
            songs = os.listdir(music_dir)
            print(songs)
            os.startfile(os.path.join(music_dir, songs[0]))

        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"Sir the time is {strTime}")

bota()