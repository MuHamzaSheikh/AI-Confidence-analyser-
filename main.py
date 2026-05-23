import cv2
import pyaudio
import wave
import threading
import time
import numpy as np
import librosa
import os
from deepface import DeepFace

# ==========================================
# CONFIGURATION
# ==========================================
RECORDING_DURATION = 15  # <--- STOPS AUTOMATICALLY AFTER 15 SECONDS
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "live_audio_temp.wav"

WEIGHTS = {
    "happy": 1.0, "neutral": 0.9, "angry": 0.7,
    "surprise": 0.3, "sad": 0.2, "fear": 0.1, "disgust": 0.1
}

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

def analyze_audio_accuracy(audio_path):
    print("\n[AUDIO] Analyzing Voice Stability...")
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
        return 0

def main():
    cap = cv2.VideoCapture(0)
    recorder = AudioRecorder()
    recorder.start()

    visual_scores = []
    frame_count = 0
    current_display_score = 0
    start_time = time.time()  # <--- Timer Starts Here

    print("\n" + "="*50)
    print(f" LIVE CONFIDENCE CHECKER")
    print(f" Recording for {RECORDING_DURATION} seconds...")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. CHECK TIMER
        elapsed_time = int(time.time() - start_time)
        time_left = RECORDING_DURATION - elapsed_time
        
        if time_left <= 0:
            print("[INFO] Time limit reached. Stopping...")
            break

        # Draw Timer & Score
        color = (0, 255, 0) if current_display_score > 70 else (0, 165, 255)
        cv2.putText(frame, f"Time Left: {time_left}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Confidence: {current_display_score:.1f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow('Confidence AI', frame)

        if frame_count % 5 == 0:
            try:
                objs = DeepFace.analyze(
                    img_path = frame, 
                    actions = ['emotion'], 
                    detector_backend = 'ssd', 
                    enforce_detection = False, 
                    silent = True
                )
                if len(objs) > 0:
                    emotions = objs[0]['emotion']
                    current_score = 0
                    for emo, score in emotions.items():
                        current_score += (score * WEIGHTS.get(emo, 0))
                    visual_scores.append(current_score)
                    current_display_score = current_score
                    print(f"[Time {elapsed_time}s] Confidence: {current_score:.1f}%")
            except:
                pass 

        frame_count += 1
        
        # Keep 'q' as an option, but timer is the main way to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    recorder.stop()
    cap.release()
    cv2.destroyAllWindows()

    if visual_scores:
        avg_visual = sum(visual_scores) / len(visual_scores)
    else:
        avg_visual = 0

    avg_audio = analyze_audio_accuracy(WAVE_OUTPUT_FILENAME)
    final_score = (avg_visual * 0.6) + (avg_audio * 0.4)

    print("\n" + "#"*30)
    print(" FINAL REPORT")
    print("#"*30)
    print(f"Visual: {avg_visual:.1f}%")
    print(f"Audio:  {avg_audio:.1f}%")
    print("-"*30)
    print(f"TOTAL:  {final_score:.1f}%")
    print("#"*30)
    
    if os.path.exists(WAVE_OUTPUT_FILENAME):
        os.remove(WAVE_OUTPUT_FILENAME)

if __name__ == "__main__":
    main()