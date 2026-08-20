# 🏋️ AI Posture Coach - Physiotherapy Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link-here.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-2.0+-orange.svg)](https://mediapipe.dev/)

---

## 🌟 Project Overview 

**AI Posture Coach** is a zero-cost, AI-powered physiotherapy assistant built for **Government Hospitals, Primary Health Centers (PHCs), and Rural Healthcare**. 

It uses your **laptop/phone webcam** to track a patient's body posture in real-time while performing squats or knee exercises. The AI calculates the **exact knee angle** and provides instant **audio-visual feedback** (Text alerts on screen) to prevent injuries and ensure correct form.

**🇮🇳 Government Use Case:** 
This software can be deployed in rural India to reduce the workload on physiotherapists. Patients can do their daily rehab at home while the AI acts as a virtual coach, sending reports to the doctor.

---

## 🚀 Key Features 

| Feature | Description |
| :--- | :--- |
| **Real-Time Tracking** | Tracks 33 body landmarks in real-time using just a standard webcam. |
| **Knee Angle Calculation** | Measures left and right knee angles (in degrees) with 90%+ accuracy. |
| **Instant Feedback** | Text alerts like *"Perfect Squat!"*, *"Too Deep!"*, or *"Lean Back!"* appear on screen. |
| **Full Body Skeleton** | Draws a pose skeleton on the patient so they can visualize their movement. |
| **Zero Hardware Cost** | Works on any basic laptop/desktop. No expensive sensors needed. |
| **Privacy First** | Processes everything locally on the device. No video is stored or uploaded. |

---

## 🛠️ Tech Stack 

- **Frontend & Interface:** [Streamlit](https://streamlit.io/) (Pure Python Web App)
- **AI Pose Detection:** [MediaPipe](https://mediapipe.dev/) by Google
- **Image Processing:** OpenCV
- **WebRTC Streaming:** `streamlit-webrtc` (for low-latency camera feed)
- **Language:** Python 3.8+

---

## 📥 Installation & Setup 

Follow these exact steps to run the project on your local machine:

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/ai-posture-coach.git
cd ai-posture-coach
