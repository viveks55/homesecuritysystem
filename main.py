import cv2
import os
import numpy as np
import tkinter as tk
import tkinter.font as font
from datetime import datetime
import time
import smtplib
from email.message import EmailMessage

class HomeSecuritySystem:
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("800x500")
        self.window.title("Home Security System")
        self.window.configure(bg='#2C3E50')

        title_label = tk.Label(self.window, text="Home Security System", bg='#2C3E50', fg='white')
        title_label.grid(row=0, column=0, columnspan=2, pady=(20, 0), padx=(20, 20))
        title_label_font = font.Font(size=40, weight='bold', family='Helvetica')
        title_label['font'] = title_label_font

        button1 = tk.Button(self.window, text="Start Security System", command=self.start_security_system, height=2, width=30, bg='#2980B9', fg='white')
        button1.grid(row=1, column=0, columnspan=2, pady=(50, 20), padx=(20, 20))
        button1['font'] = font.Font(size=20, weight='bold')

        button2 = tk.Button(self.window, text="Add Members", command=self.add_members, height=2, width=30, bg='#2980B9', fg='white')
        button2.grid(row=2, column=0, columnspan=2, pady=(20, 50), padx=(20, 20))
        button2['font'] = font.Font(size=20, weight='bold')

        self.motion_detected = False

    def start_security_system(self):
        cap = cv2.VideoCapture(0)
        cascade_filename = "haarcascade_frontalface_default.xml"
        
        paths = [os.path.join("persons", im) for im in os.listdir("persons")]
        labels_list = {}
        for path in paths:
            labels_list[path.split('/')[-1].split('-')[2].split('.')[0]] = path.split('/')[-1].split('-')[0]

        recog = cv2.face.LBPHFaceRecognizer_create()
        recog.read('model.yml')

        cascade = cv2.CascadeClassifier(cascade_filename)

        start_time = time.time()
        unknown_detected_time = 0
        unknown_face_detected = False
        known_face_detected = False

        while True:
            _, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = cascade.detectMultiScale(gray, 1.3, 2)

            if len(faces) > 0:
                unknown_detected_time = time.time() - start_time
                unknown_face_detected = True
                known_face_detected = False
            else:
                unknown_face_detected = False

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = gray[y:y + h, x:x + w]

                label_id, confidence = recog.predict(roi)

                if confidence < 100:
                    name = labels_list[str(label_id)]
                    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    known_face_detected = True
                    unknown_face_detected = False
                else:
                    name = "Unknown"
                    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if unknown_face_detected and unknown_detected_time >= 5:
                        self.save_video(cap)
                        self.send_email_alert()
                        return

            cv2.imshow("Home Security System", frame)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def add_members(self):
        name = input("Enter name: ")

        count = 1
        ids = input("Enter ID: ")

        cap = cv2.VideoCapture(0)

        cascade_filename = "haarcascade_frontalface_default.xml"

        cascade = cv2.CascadeClassifier(cascade_filename)

        while True:
            _, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = cascade.detectMultiScale(gray, 1.4, 1)

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = gray[y:y + h, x:x + w]

                cv2.imwrite(f"persons/{name}-{count}-{ids}.jpg", roi)
                count = count + 1
                cv2.putText(frame, f"{count}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                cv2.imshow("New Member", roi)

            cv2.imshow("Add Members", frame)

            if cv2.waitKey(1) == 27 or count > 300:
                cv2.destroyAllWindows()
                cap.release()
                self.train()
                break

    def train(self):
        print("Training started!")

        recog = cv2.face.LBPHFaceRecognizer_create()

        dataset = 'persons'

        paths = [os.path.join(dataset, im) for im in os.listdir(dataset)]

        faces = []
        ids = []
        labels = []
        for path in paths:
            labels.append(path.split('/')[-1].split('-')[0])

            ids.append(int(path.split('/')[-1].split('-')[2].split('.')[0]))

            faces.append(cv2.imread(path, 0))

        recog.train(faces, np.array(ids))

        recog.save('model.yml')

        print("Training completed!")

    def save_video(self, cap):
        if not os.path.exists("recordings"):
            os.makedirs("recordings")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'recordings/{datetime.now().strftime("%H-%M-%S")}.avi', fourcc, 20.0, (640, 480))

        start_time = time.time()
        end_time = start_time + 10  # Recording duration: 10 seconds

        while time.time() < end_time:
            _, frame = cap.read()  # Read a new frame from the camera
            out.write(frame)  # Write the frame to the video file

            cv2.imshow("Recording", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        out.release()
        cv2.destroyAllWindows()

        print("Video saved!")

    def send_email_alert(self):
        sender_email = "prithvivichu007@gmail.com"
        receiver_email = "vichus345@gmail.com"
        password = "gjlsupbubivoklwa"

        subject = "Security Alert: Unrecognized Face Detected"
        body = "An unknown face has been detected in your home. A video recording has been saved."

        msg = EmailMessage()
        msg.set_content(body)

        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(sender_email, password)
                smtp.send_message(msg)
                print("Email sent successfully!")
        except Exception as e:
            print("Error sending email:", str(e))

    def run(self):
        self.window.mainloop()

# Create an instance of the HomeSecuritySystem class and run the application
home_security_system = HomeSecuritySystem()
home_security_system.run()
