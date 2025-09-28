import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import pyttsx3
import time

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load custom background image
custom_bg = cv2.imread("image2.png")  # Replace with your custom image path
if custom_bg is None:
    raise FileNotFoundError(" Custom background image not found! Make sure 'custom_bg.jpg' exists and path is correct.")
custom_bg = cv2.resize(custom_bg, (640, 480))

# Load model
model = load_model("alphbet_recognition.keras")
engine = pyttsx3.init()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Globals
prev_x, prev_y = None, None
canvas_np = None
current_letter = 'A'
current_point = 0
path_radius = 20
last_point_time = 0

# Tracing paths for all letters (simplified sample)
letter_paths = {
    'A': [(300, 400), (280, 300), (320, 300), (300, 350)],
    'B': [(300,150),(300,200),(300,250),(300,300),(300,350),
          (300,150),(325,200),(350,250),(325,300),
          (300,200),(325,225),(350,250),(325,275)],
    'C': [(350, 80), (300, 120), (280, 170), (300, 220), (350, 260)],
    'D': [(300,150),(300,200), (300,250), (300,300), (300,350), (300,400), (300,150), 
          (350,160), (400,200), (425,275), (400,350), (350,390), (300,400)],
    'Z': [(250,150), (300,150), (350,150), (300,200), (250,250), (300,250), (350,250)],
    'E': [(300,150), (300,200), (300,250),
          (300,150),(350,150), (400,150), 
           (300,200),(350,200),
           (300,250),(350,250), (400,250)],
    'F': [(300,150), (300,200), (300,250),
           (300,150),(350,150), (400,150),
           (300,200),(350,200)],
    'I': [(300,150),(350,150), (400,150),
          (300,150), (350,200),
          (350,250),
          (300,300),(350,300), (400,300)], 
    'H': [(300,150), (300,200), (300,250),
          (300,150),(350,200), (400,150),
          (400,150), (400,200), (400,250)],
    'L': [(300,150), (300,200), (300,250),
          (300,250),(350,250)],   
    'M': [(300,150), (300,200), (300,250),(300,300),
          (300,150),(350,200),(400,250),
          (450,200),(500,150),
          (500,150),(500,200),(500,250),(500,300)],
    'N': [(300,150), (300,200), (300,250),(300,300)]
}

# Predict function
def predict_letter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "No drawing"

    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    roi = gray[y:y+h, x:x+w]
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    resized = cv2.resize(roi, (28, 28)) / 255.0
    input_data = resized.reshape(1, 28, 28, 1)
    pred = model.predict(input_data)
    index = np.argmax(pred)
    letter = chr(ord('A') + index)
    return letter

# Button actions
def clear_canvas():
    global canvas_np, current_point
    canvas_np = np.zeros((480, 640, 3), dtype=np.uint8)
    current_point = 0

def save_canvas():
    filename = f"canvas_{cv2.getTickCount()}.png"
    cv2.imwrite(filename, canvas_np)
    status_label.config(text=f"Saved: {filename}")

def run_prediction():
    letter = predict_letter(canvas_np)
    status_label.config(text=f"Predicted: {letter}")
    engine.say(f"The predicted letter is {letter}")
    engine.runAndWait()

def change_letter(event):
    global current_letter, current_point
    current_letter = letter_var.get()
    current_point = 0

# Update video frame
def update_frame():
    global prev_x, prev_y, canvas_np, current_point, last_point_time
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas_np is None:
        canvas_np = np.zeros_like(frame)

    # Background removal + custom background
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_seg = segmentor.process(rgb)
    mask = results_seg.segmentation_mask > 0.1
    frame = np.where(mask[..., None], frame, custom_bg)

    results = hands.process(rgb)

    # Draw trace path
    current_path = letter_paths.get(current_letter, [])
    for i, point in enumerate(current_path):
        color = (0, 255, 0) if i == current_point else (255, 255, 255)
        cv2.circle(frame, point, path_radius, color, 2)

    drawing_status = "Drawing: OFF"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            index_joint = hand_landmarks.landmark[6]
            middle_tip = hand_landmarks.landmark[12]
            middle_joint = hand_landmarks.landmark[10]

            index_up = index_tip.y < index_joint.y
            middle_up = middle_tip.y < middle_joint.y

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            if index_up and not middle_up:
                drawing_status = "Drawing: ON"
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas_np, (prev_x, prev_y), (x, y), (255, 0, 0), 5)  # Blue pen
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

            if current_point < len(current_path):
                px, py = current_path[current_point]
                dist = ((x - px)**2 + (y - py)**2)**0.5
                if dist < path_radius and time.time() - last_point_time > 0.5:
                    current_point += 1
                    last_point_time = time.time()

                if current_point == len(current_path):
                    cv2.putText(frame, f"✔ {current_letter} Completed!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.putText(frame, drawing_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0) if "ON" in drawing_status else (0, 0, 255), 2)

    display = cv2.add(frame, canvas_np)
    img = Image.fromarray(display)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, update_frame)

# GUI setup
root = tk.Tk()
root.title("Air Canvas - Gesture Alphabet Recognizer")

video_label = tk.Label(root)
video_label.pack()

controls = tk.Frame(root)
controls.pack()

btn_clear = tk.Button(controls, text="Clear", command=clear_canvas)
btn_clear.grid(row=0, column=0, padx=5)

btn_save = tk.Button(controls, text="Save", command=save_canvas)
btn_save.grid(row=0, column=1, padx=5)

btn_predict = tk.Button(controls, text="Predict", command=run_prediction)
btn_predict.grid(row=0, column=2, padx=5)

letter_var = tk.StringVar(value='A')
letter_menu = ttk.Combobox(controls, textvariable=letter_var, values=list(letter_paths.keys()))
letter_menu.grid(row=0, column=3, padx=5)
letter_menu.bind("<<ComboboxSelected>>", change_letter)

status_label = tk.Label(root, text="", fg="blue")
status_label.pack()

cap = cv2.VideoCapture(0)
update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
