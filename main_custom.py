import cv2
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
# Audio Config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "live_audio_temp.wav"

# Load YOUR Custom Brain
print("[INFO] Loading your custom AI model...")
model = tf.keras.models.load_model("my_confidence_model.h5")
print("[INFO] Model loaded successfully!")

# ==========================================
# AUDIO RECORDER (Background)
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

# ==========================================
# AUDIO ANALYSIS
# ==========================================
def analyze_audio_accuracy(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        if len(y) == 0: return 0

        rms = librosa.feature.rms(y=y)
        avg_vol = np.mean(rms)
        flatness = librosa.feature.spectral_flatness(y=y)
        instability = np.std(flatness)

        vol_score = min((avg_vol / 0.05) * 100, 100)
        stab_score = max(0, 100 - (instability * 2000))
        
        return (vol_score * 0.4) + (stab_score * 0.6)
    except:
        return 50 # Default if audio fails

# ==========================================
# MAIN APP
# ==========================================
def main():
    cap = cv2.VideoCapture(0)
    
    recorder = AudioRecorder()
    recorder.start()

    visual_scores = []
    frame_count = 0
    current_display_score = 0
    
    print("\n" + "="*50)
    print(" CUSTOM CONFIDENCE AI RUNNING")
    print(" Press 'q' to stop.")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. PREPROCESS IMAGE FOR AI
        # The model expects 224x224 size, and values between 0-1
        small_frame = cv2.resize(frame, (224, 224))
        img_array = np.array(small_frame, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        # 2. PREDICT (Every 5th frame for speed)
        if frame_count % 5 == 0:
            try:
                # Prediction is a number between 0 (Confident) and 1 (Nervous)
                prediction = model.predict(img_array, verbose=0)[0][0]
                
                # Convert Nervousness to Confidence
                # If prediction is 0.1 (Low Nervous), Confidence is 90%
                confidence = (1.0 - prediction) * 100
                
                visual_scores.append(confidence)
                current_display_score = confidence
                
                # Print to terminal for debugging
                state = "Confident" if confidence > 50 else "Nervous"
                print(f"[AI] Score: {confidence:.1f}% ({state})")

            except Exception as e:
                print(e)

        # 3. DRAW ON SCREEN
        color = (0, 255, 0) if current_display_score > 60 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {current_display_score:.1f}%", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Custom Confidence AI', frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # STOP & REPORT
    recorder.stop()
    cap.release()
    cv2.destroyAllWindows()

    if visual_scores:
        avg_visual = sum(visual_scores) / len(visual_scores)
    else:
        avg_visual = 0

    print("\n[INFO] Analyzing Audio...")
    avg_audio = analyze_audio_accuracy(WAVE_OUTPUT_FILENAME)
    
    final_score = (avg_visual * 0.6) + (avg_audio * 0.4)

    print("\n" + "#"*30)
    print(" FINAL REPORT")
    print("#"*30)
    print(f"Visual Score: {avg_visual:.1f}%")
    print(f"Audio Score:  {avg_audio:.1f}%")
    print("-" * 30)
    print(f"TOTAL CONFIDENCE: {final_score:.1f}%")
    print("#"*30)
    
    if os.path.exists(WAVE_OUTPUT_FILENAME):
        os.remove(WAVE_OUTPUT_FILENAME)

if __name__ == "__main__":
    main()