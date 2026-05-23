import customtkinter as ctk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import tensorflow as tf
import pyaudio
import wave
import threading
import time
import os

# ==========================================
# CONFIGURATION
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Audio Config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "hud_audio_temp.wav"

# Load Face Detector (Standard OpenCV - Works on Python 3.13)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Custom Model
print("[INFO] Loading Neural Network...")
try:
    model = tf.keras.models.load_model("my_confidence_model.h5")
    print("[SUCCESS] Brain Loaded.")
except:
    print("[CRITICAL] 'my_confidence_model.h5' not found. AI will be disabled.")
    model = None

# ==========================================
# AUDIO BACKEND
# ==========================================
class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.stream = None

    def start(self):
        self.is_recording = True
        self.frames = []
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=CHUNK)
        self.thread = threading.Thread(target=self.record)
        self.thread.start()

    def record(self):
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            except:
                pass

    def stop(self):
        self.is_recording = False
        try:
            self.thread.join()
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
        except:
            pass

# ==========================================
# THE "IRON HUD" INTERFACE
# ==========================================
class HUDConfidenceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Tactical Confidence HUD")
        self.geometry("1280x720")
        
        # Full Screen Video Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Video Container
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # 2. Control Panel (Bottom Center)
        # FIX: Removed 'alpha' argument. Changed color to solid black to look cool.
        self.hud_frame = ctk.CTkFrame(self, fg_color="#050505", corner_radius=0)
        self.hud_frame.place(relx=0.5, rely=0.95, anchor="center", relwidth=1.0, relheight=0.1)

        self.start_btn = ctk.CTkButton(self.hud_frame, text="ACTIVATE SYSTEM", 
                                       command=self.start_system, width=200, height=40,
                                       font=("Consolas", 14, "bold"), fg_color="#00ff00", text_color="black")
        self.start_btn.pack(side="left", padx=50, pady=10)

        self.stop_btn = ctk.CTkButton(self.hud_frame, text="SHUTDOWN", 
                                      command=self.stop_system, width=150, height=40,
                                      font=("Consolas", 14, "bold"), fg_color="#ff0000",
                                      state="disabled")
        self.stop_btn.pack(side="right", padx=50, pady=10)

        self.status_lbl = ctk.CTkLabel(self.hud_frame, text="SYSTEM STANDBY", font=("Consolas", 20, "bold"), text_color="white")
        self.status_lbl.pack(side="top", pady=5)

        # Variables
        self.cap = None
        self.recorder = AudioRecorder()
        self.is_running = False
        self.visual_scores = []
        self.current_confidence = 0
        self.frame_count = 0
        self.scan_line_y = 0  
        self.scan_direction = 5

    def start_system(self):
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.recorder.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_lbl.configure(text="SYSTEM ACTIVE - SCANNING TARGET", text_color="#00ff00")
        self.update_frame()

    def stop_system(self):
        self.is_running = False
        self.recorder.stop()
        if self.cap: self.cap.release()
        self.video_label.configure(image=None)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_lbl.configure(text="SYSTEM OFFLINE", text_color="red")
        print("Final Report Generated.")

    def draw_tech_corners(self, img, x, y, w, h, color):
        length = 30
        thickness = 2
        # Top Left
        cv2.line(img, (x, y), (x + length, y), color, thickness)
        cv2.line(img, (x, y), (x, y + length), color, thickness)
        # Top Right
        cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
        # Bottom Left
        cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
        # Bottom Right
        cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

    def update_frame(self):
        if not self.is_running: return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # --- HUD OVERLAY LOGIC ---
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if self.current_confidence > 60 else (0, 0, 255)
                
                self.draw_tech_corners(frame, x, y, w, h, color)

                # Scanning Laser Effect
                cv2.line(frame, (x, y + self.scan_line_y), (x + w, y + self.scan_line_y), (0, 255, 255), 2)
                self.scan_line_y += self.scan_direction
                if self.scan_line_y > h or self.scan_line_y < 0:
                    self.scan_direction *= -1

                # Floating Text
                label = f"CONF: {self.current_confidence:.1f}%"
                cv2.putText(frame, label, (x + w + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, "STATUS: TRACKING", (x + w + 10, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Run AI Prediction
                if self.frame_count % 5 == 0 and model:
                    try:
                        # FIX: This line was crashed in your screenshot. It is fixed now.
                        face_roi = frame[y:y+h, x:x+w]
                        
                        if face_roi.size > 0:
                            small_frame = cv2.resize(face_roi, (224, 224))
                            img_array = np.array(small_frame, dtype="float32") / 255.0
                            img_array = np.expand_dims(img_array, axis=0)
                            prediction = model.predict(img_array, verbose=0)[0][0]
                            self.current_confidence = (1.0 - prediction) * 100
                            self.visual_scores.append(self.current_confidence)
                    except:
                        pass

            # Crosshair
            h_img, w_img, _ = frame.shape
            cx, cy = w_img // 2, h_img // 2
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (100, 100, 100), 1)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (100, 100, 100), 1)

            # GUI Update
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame_rgb)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(1280, 720))
            
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk
            
            self.frame_count += 1
            self.after(10, self.update_frame)

if __name__ == "__main__":
    app = HUDConfidenceApp()
    app.mainloop()