import customtkinter as ctk
import cv2
import PIL.Image
import numpy as np
import tensorflow as tf
import pyaudio
import wave
import threading
import time
import librosa
import os
# ==========================================
# CONFIGURATION
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
# Audio Config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "live_audio_temp.wav"



# ==========================================

# 1. LOAD AI MODEL

# ==========================================

print("[INFO] Loading AI Brain...")

try:
    model = tf.keras.models.load_model("my_confidence_model.h5")
    print("[SUCCESS] Model loaded!")

except:
    print("[ERROR] Could not load 'my_confidence_model.h5'. Make sure the file exists!")
    model = None

# ==========================================

# 2. AUDIO RECORDER CLASS

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
            # Save the recorded data as a WAV file  
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
        except:
            pass



# ==========================================

# 3. GUI APPLICATION CLASS

# ==========================================

class ConfidenceApp(ctk.CTk):

    def __init__(self):

        super().__init__()
        # Window Setup
        self.title("AI Confidence Analyzer Pro")
        self.geometry("1100x700")
        # Variables
        self.cap = None
        self.recorder = AudioRecorder()
        self.is_running = False
        self.frame_count = 0
        self.visual_scores = []
        # --- LAYOUT ---
        # Grid: 2 Columns. Left = Video, Right = Stats
        self.grid_columnconfigure(0, weight=3) # Video gets more space
        self.grid_columnconfigure(1, weight=1) # Stats bar
        self.grid_rowconfigure(0, weight=1)
        # 1. LEFT FRAME (Video)
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)
        # 2. RIGHT FRAME (Controls & Stats)
        self.stats_frame = ctk.CTkFrame(self, fg_color="#2B2B2B")
        self.stats_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        # Title
        self.title_label = ctk.CTkLabel(self.stats_frame, text="ANALYSIS HUD", 
                                        font=("Roboto", 24, "bold"))

        self.title_label.pack(pady=20)
        # Confidence Score (Big Text)
        self.score_label = ctk.CTkLabel(self.stats_frame, text="0%", 
                                        font=("Roboto", 60, "bold"), text_color="#00FF00")
        self.score_label.pack(pady=10)
        self.status_label = ctk.CTkLabel(self.stats_frame, text="READY", 
                                         font=("Roboto", 20))
        self.status_label.pack(pady=5)

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.stats_frame, width=200, height=20)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=20)

        # Start Button
        self.start_btn = ctk.CTkButton(self.stats_frame, text="START CAMERA", 

                                       command=self.start_analysis,

                                       fg_color="green", hover_color="darkgreen")

        self.start_btn.pack(pady=10)
        # Stop Button
        self.stop_btn = ctk.CTkButton(self.stats_frame, text="STOP & REPORT", 
                                      command=self.stop_analysis,

                                      fg_color="red", hover_color="darkred",

                                      state="disabled")

        self.stop_btn.pack(pady=10)     
       # Audio Score Label (Hidden until end)
        self.audio_result_label = ctk.CTkLabel(self.stats_frame, text="", font=("Roboto", 16))
        self.audio_result_label.pack(pady=20)
    def start_analysis(self):
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.visual_scores = []
        self.recorder.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="ANALYZING...", text_color="yellow")
        
        self.update_frame()

    def stop_analysis(self):
        self.is_running = False
        self.recorder.stop()
        if self.cap:
            self.cap.release()
        self.start_btn.configure(state="normal")

        self.stop_btn.configure(state="disabled")

        self.video_label.configure(image=None) # Clear video

        # FINAL REPORT CALCULATION
        self.calculate_final_report()
    def update_frame(self):
        if not self.is_running:
            return
        ret, frame = self.cap.read()
        if ret:
            # 1. Flip & Color Correct
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. AI PREDICTION (Every 5 frames)

            if self.frame_count % 5 == 0 and model:
                try:

                    # Resize to 224x224 for MobileNet

                    small_frame = cv2.resize(frame, (224, 224))

                    img_array = np.array(small_frame, dtype="float32") / 255.0

                    img_array = np.expand_dims(img_array, axis=0)
                    # Predict
                    prediction = model.predict(img_array, verbose=0)[0][0]
                    confidence = (1.0 - prediction) * 100
                    self.visual_scores.append(confidence)
                    # Update GUI Labels
                    self.score_label.configure(text=f"{confidence:.0f}%")
                    self.progress_bar.set(confidence / 100)
                    if confidence > 50:

                        self.score_label.configure(text_color="#00FF00") # Green

                        self.status_label.configure(text="CONFIDENT", text_color="#00FF00")

                    else:

                        self.score_label.configure(text_color="#FF0000") # Red

                        self.status_label.configure(text="NERVOUS", text_color="#FF0000")



                except Exception as e:

                    print(e)



            # 3. Display Video in GUI

            image = PIL.Image.fromarray(rgb_frame)

            # Scale image to fit window

            ctk_img = ctk.CTkImage(light_image=image, dark_image=image, size=(640, 480))

            self.video_label.configure(image=ctk_img)

            self.video_label.image = ctk_img



            self.frame_count += 1

            

            # Loop recursively every 10ms

            self.after(10, self.update_frame)



    def calculate_final_report(self):

        # Visual Average

        if self.visual_scores:

            avg_visual = sum(self.visual_scores) / len(self.visual_scores)

        else:

            avg_visual = 0



        # Audio Analysis

        self.status_label.configure(text="PROCESSING AUDIO...", text_color="white")

        self.update() # Force GUI update

        

        avg_audio = self.get_audio_score()

        

        final_score = (avg_visual * 0.6) + (avg_audio * 0.4)

        

        # Show Result

        self.status_label.configure(text="FINAL VERDICT", text_color="cyan")

        self.score_label.configure(text=f"{final_score:.1f}%", text_color="cyan")

        self.audio_result_label.configure(

            text=f"Visual Score: {avg_visual:.1f}%\nAudio Score: {avg_audio:.1f}%"

        )



        #if os.path.exists(WAVE_OUTPUT_FILENAME):

            
            #os.remove(WAVE_OUTPUT_FILENAME)



    def get_audio_score(self):

        try:

            y, sr = librosa.load(WAVE_OUTPUT_FILENAME)

            if len(y) == 0: return 0

            rms = librosa.feature.rms(y=y)

            avg_vol = np.mean(rms)

            flatness = librosa.feature.spectral_flatness(y=y)

            instability = np.std(flatness)

            vol_score = min((avg_vol / 0.05) * 100, 100)

            stab_score = max(0, 100 - (instability * 2000))

            return (vol_score * 0.4) + (stab_score * 0.6)

        except:

            return 0



if __name__ == "__main__":

    app = ConfidenceApp()

    app.mainloop()

