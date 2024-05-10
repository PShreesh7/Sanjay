import cv2
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()


words = []


while True:

    with mic as source:
        print("Listening...")
        audio = r.listen(source)


    try:
        print("Recognizing...")
        text = r.recognize_google(audio)
        print("You said: " + text)
        
        words.append(text)
        if text == "exit":
                break
 
    except Exception as e:
        if isinstance(e, sr.UnknownValueError):
            print("Error: Could not recognize the speech")
        else:
            print("Error: " + str(e))


print("Recognized words:")
print(words[:-1])
model = YOLO("yolov8n.pt")
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
obj = []
while True:

    ret, frame = cap.read()

    results = model(frame)


    for result in results:
        
        print("Detected objects:")
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls = box.cls
            conf = box.conf.item()  
            class_name = model.names[int(cls)]
            if class_name not in obj:
                obj.append(class_name)
            else:
                pass
            for sound in words:
                if class_name == sound:
                    engine.say(sound+'Detected')
                    engine.runAndWait()
                    words.remove(sound)
                    
        
            
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    
 
    cv2.imshow('frame', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
for item in obj:
    print(item)


cap.release()
cv2.destroyAllWindows()

