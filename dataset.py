import cv2
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr
import threading
import time

r = sr.Recognizer()
mic = sr.Microphone()

words = []
model = YOLO("yolov8n.pt")
engine = pyttsx3.init()
loop_running = False

def speak(text):
    global loop_running
    if not loop_running:
        loop_running = True
        engine.say(text)
        engine.runAndWait()
        loop_running = False
    else:
        engine.say(text)
        engine.runAndWait()

cap = cv2.VideoCapture(0)

def yo():
    engine.say("Please tell the name of items you want to buy. If all items are added then please say 'close'.")
    engine.runAndWait()
    
    while True:
        with mic as source:
            print("Listening...")
            audio = r.listen(source)
        
        try:
            print("Recognizing...")
            text = r.recognize_google(audio)
            print(text)
            engine.say(text)
            engine.runAndWait()
            
            words.append(text)
            if text == "close":
                break
        except Exception as e:
            if isinstance(e, sr.UnknownValueError):
                print("Error: Could not recognize the speech")
                engine.say("Error. Could not recognize the speech, please repeat.")
                engine.runAndWait()
            else:
                print("Error: " + str(e))

    obj = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls = box.cls
                conf = box.conf.item()
                class_name = model.names[int(cls)]
                
                if class_name not in obj:
                    obj.append(class_name)
                    
                for sound in words:
                    if class_name.lower() == sound.lower():
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            position = "left" if x1 < frame.shape[1] // 3 else "right" if x1 > frame.shape[1] * 2 // 3 else "center"
                            cv2.putText(frame, position, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            speak(f"{sound} detected {position}")
                            
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    intro = """Hello, I'm your personal assistant. What do you want me to do?
    For navigation, please say 'detection'.
    For shopping, please say 'market'."""
    
    engine.say(intro)
    engine.runAndWait()
    
    while True:
        with mic as source:
            audio = r.listen(source)

        try:
            print("Recognizing...")
            order = r.recognize_google(audio)
            print(order)
            engine.say(order)
            engine.runAndWait()
            
            if order == "market":
                engine.say("Market mode activated.")
                engine.runAndWait()
                yo()
                break
            elif order == 'detection':
                engine.say("Currently not working.")
                engine.runAndWait()
                engine.say("Would you like to use market mode? Please respond with 'yes' or 'no'.")
                engine.runAndWait()
                
                while True: 
                    with mic as source:
                        audio = r.listen(source)
                        try:
                            order = r.recognize_google(audio)
                            if order == 'yes':
                                engine.say("Opening market mode.")
                                engine.runAndWait()
                                yo()
                                break
                            elif order == 'no':
                                engine.say("Okay.")
                                engine.runAndWait()
                                break
                        except Exception as e:
                            if isinstance(e, sr.UnknownValueError):
                                print("Error: Could not recognize the speech")
                                engine.say("Error. Could not recognize the speech, please repeat.")
                                engine.runAndWait()
                            else:
                                print("Error: " + str(e))
        except Exception as e:
            if isinstance(e, sr.UnknownValueError):
                print("Error: Could not recognize the speech")
                engine.say("Could not recognize the speech. Please repeat.")
                engine.runAndWait()
            else:
                print("Error: " + str(e))

main()
