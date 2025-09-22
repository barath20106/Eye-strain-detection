import cv2
import mediapipe as mp
import time
import tkinter as tk
from tkinter import ttk, messagebox
from playsound import playsound
from PIL import Image, ImageTk
import threading
from datetime import timedelta

# Constants
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
EAR_THRESHOLD = 0.23
CONSEC_FRAMES = 3
MIN_BLINKS_PER_MIN = 12
FATIGUE_TIME = 15  # seconds eyes closed continuously

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

def calculate_ear(eye_points, landmarks):
    A = euclidean_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    B = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    C = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
    return (A + B) / (2.0 * C)

class EyeStrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Strain Detector")

        # GUI setup
        ttk.Label(root, text="Select Camera:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.cam_combo = ttk.Combobox(root, values=self.get_camera_list(), state="readonly", width=5)
        self.cam_combo.grid(row=0, column=1, padx=5, pady=5)
        self.cam_combo.current(0)

        self.start_btn = ttk.Button(root, text="Start", command=self.start_detection)
        self.start_btn.grid(row=1, column=0, padx=5, pady=5)

        self.stop_btn = ttk.Button(root, text="Stop", command=self.stop_detection, state="disabled")
        self.stop_btn.grid(row=1, column=1, padx=5, pady=5)

        self.status_label = ttk.Label(root, text="Status: Idle", font=("Arial", 10))
        self.status_label.grid(row=2, column=0, columnspan=2)

        self.video_label = tk.Label(root)
        self.video_label.grid(row=3, column=0, columnspan=2)

        self.log_text = tk.Text(root, height=10, width=60, state="disabled")
        self.log_text.grid(row=4, column=0, columnspan=2, pady=10)

        # State variables
        self.cap = None
        self.running = False
        self.blink_count = 0
        self.counter = 0
        self.paused = False
        self.fatigue_start = None
        self.last_minute_check = None
        self.minute_blink_start = 0

        # Alert control
        self.eye_strain_alerted = False
        self.fatigue_alerted = False

        # Timer for elapsed time and pause handling
        self.start_time = None
        self.paused_time_accumulator = 0
        self.pause_start_time = None

        # Mediapipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_camera_list(self):
        available = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(str(i))
                cap.release()
        return available if available else ["0"]

    def start_detection(self):
        cam_index = int(self.cam_combo.get())
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {cam_index}")
            return

        self.running = True
        self.blink_count = 0
        self.counter = 0
        self.paused = False
        self.fatigue_start = None
        self.eye_strain_alerted = False
        self.fatigue_alerted = False
        self.paused_time_accumulator = 0
        self.pause_start_time = None

        self.start_time = time.time()
        self.last_minute_check = 0
        self.minute_blink_start = 0

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="Status: Running")
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")

        self.update_frame()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Status: Stopped")
        self.video_label.config(image='')

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        face_detected = False
        eye_closed = False
        current_time = time.time()

        if results.multi_face_landmarks:
            face_detected = True
            self.status_label.config(text="Status: Running - Face Detected")

            if self.paused:
                if self.pause_start_time:
                    self.paused_time_accumulator += current_time - self.pause_start_time
                self.paused = False
                self.pause_start_time = None

            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                left_ear = calculate_ear(LEFT_EYE_IDX, landmarks)
                right_ear = calculate_ear(RIGHT_EYE_IDX, landmarks)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    self.counter += 1
                    eye_closed = True
                    if self.fatigue_start is None:
                        self.fatigue_start = current_time
                    elif current_time - self.fatigue_start >= FATIGUE_TIME and not self.fatigue_alerted:
                        self.fatigue_alerted = True
                        threading.Thread(target=self.show_fatigue_alert, daemon=True).start()
                else:
                    if self.counter >= CONSEC_FRAMES:
                        self.blink_count += 1
                    self.counter = 0
                    self.fatigue_start = None
                    self.fatigue_alerted = False
        else:
            self.status_label.config(text="Status: Paused - Face Not Detected")
            if not self.paused:
                self.paused = True
                self.pause_start_time = current_time

        if self.paused:
            elapsed_time = self.pause_start_time - self.start_time - self.paused_time_accumulator
        else:
            elapsed_time = current_time - self.start_time - self.paused_time_accumulator

        current_elapsed = int(elapsed_time)
        if self.last_minute_check is None:
            self.last_minute_check = 0

        if current_elapsed - self.last_minute_check >= 60:
            start = self.last_minute_check
            end = self.last_minute_check + 60
            self.last_minute_check += 60

            blinks_in_minute = self.blink_count - self.minute_blink_start
            strain = blinks_in_minute < MIN_BLINKS_PER_MIN
            self.log_strain_result(start, end, strain)

            if strain and not self.eye_strain_alerted:
                self.eye_strain_alerted = True
                threading.Thread(target=self.show_eye_strain_alert, daemon=True).start()
            else:
                self.eye_strain_alerted = False

            self.minute_blink_start = self.blink_count

        if self.paused:
            cv2.putText(frame, "Paused - Face Not Detected", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Time: {str(timedelta(seconds=int(elapsed_time)))}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Eye Closed: {'Yes' if eye_closed else 'No'}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Face Detected: {'Yes' if face_detected else 'No'}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        im_pil = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(10, self.update_frame)

    def log_strain_result(self, start_time, end_time, strain):
        timestamp = f"{str(timedelta(seconds=int(start_time))).zfill(8)} - {str(timedelta(seconds=int(end_time))).zfill(8)}"
        message = f"{timestamp}: {'Eye strain detected' if strain else 'No eye strain'}\n"
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def show_eye_strain_alert(self):
        playsound('alert.wav')
        messagebox.showinfo("Eye Strain Alert", "Eye strain detected! Please take a break.")
        self.eye_strain_alerted = False

    def show_fatigue_alert(self):
        playsound('alert.wav')
        messagebox.showwarning("Fatigue Alert", "You have been Fatigued! Take a short break.")
        self.fatigue_alerted = False

    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeStrainApp(root)
    root.mainloop()
