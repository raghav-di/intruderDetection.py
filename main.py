import cv2
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import os
# from twilio.rest import Client



# def send_whatsapp_alert():
#     account_sid = "your_account_sid"
#     auth_token = os.getenv("whp_auth_token")
#     client = Client(account_sid, auth_token)

#     message = client.messages.create(
        # from_ = os.getenv("whp_from"),  # Twilio sandbox number
        # to = os.getenv("whp_to"),    # Your verified WhatsApp number
#         body="🚨 Human detected with >50% confidence for over 10 seconds!"
#     )
#     print("WhatsApp alert sent:", message.sid)

def send_email_alert(frame):
    sender = os.getenv("gmail_sender")
    recipient = os.getenv("gmail_recipient")
    subject = "🚨⚠️🚨INTRUDER Detected Alert"
    body = "⚠️🚨A human figure was detected🚨⚠️."

    # Save frame as image
    image_path = "alert_frame.jpg"
    cv2.imwrite(image_path, frame)

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    msg.attach(MIMEText(body))

    with open(image_path, "rb") as img:
        mime_img = MIMEImage(img.read(), name="alert_frame.jpg")
        msg.attach(mime_img)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, os.getenv("gmail_key"))
        server.sendmail(sender, recipient, msg.as_string())

    print("✅ Gmail alert with screenshot sent")



# Load MobileNet SSD
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)

DETECTION_THRESHOLD = 0.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    person_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > DETECTION_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                person_detected = True
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                send_email_alert(frame)
                time.sleep(10)

    cv2.imshow("MobileNet SSD - Human Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
