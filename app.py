import streamlit as st
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Physiotherapy Posture Coach", page_icon="🏋️", layout="centered")

st.title("🏋️ AI Posture Coach - Physiotherapy Assistant")
st.markdown("**Perform Squats in front of your camera. The AI will track your knee angles and back posture in real-time!**")

# ------------------ SIDEBAR INSTRUCTIONS ------------------
with st.sidebar:
    st.header("📋 How to use")
    st.markdown("""
    1. Click **START** below.
    2. Allow camera access.
    3. Stand **at least 6 feet away** so your full body is visible.
    4. Perform a **Squat** (bend your knees like sitting on a chair).
    5. Watch the screen for real-time angle feedback.
    """)
    st.warning("**🏆 Perfect Squat**: Bend knees to 90° and keep your back straight.")
    st.info("💡 If the screen is black, restart the app or check your camera permissions.")

# ------------------ MEDIAPIPE SETUP ------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ------------------ ANGLE CALCULATION FUNCTION ------------------
def calculate_angle(a, b, c):
    """Calculate the angle between three points (in degrees)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ------------------ VIDEO PROCESSOR CLASS (CORE AI) ------------------
class PoseProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    def transform(self, frame):
        # Convert frame to BGR for OpenCV
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        h, w, _ = img.shape
        feedback = "🧍 Stand in front of camera"
        color = (0, 255, 255)  # Yellow
        left_knee_angle = 0
        right_knee_angle = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # ----- EXTRACT KEY LANDMARKS (LEFT & RIGHT SIDE) -----
                # Left side
                l_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, 
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)
                l_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, 
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)
                l_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, 
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h)
                l_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, 
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h)

                # Right side
                r_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)
                r_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)
                r_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)
                r_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)

                # ----- CALCULATE KNEE ANGLES -----
                left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

                # ----- POSTURE CHECK (BACK STRAIGHTNESS) -----
                # If the shoulder X is far to the right/left of hip X, they are leaning forward
                shoulder_hip_diff = abs(l_shoulder[0] - l_hip[0]) 
                is_leaning = shoulder_hip_diff > 30  # threshold for leaning forward

                # ----- FEEDBACK LOGIC (SQUAT COACH) -----
                if left_knee_angle < 160 and right_knee_angle < 160:
                    # If both knees are bent
                    if 70 < left_knee_angle < 110 and 70 < right_knee_angle < 110:
                        if is_leaning:
                            feedback = "⚠️ Perfect Depth! But Lean Back!"
                            color = (0, 165, 255)  # Orange
                        else:
                            feedback = "✅ PERFECT SQUAT! Great Back Posture!"
                            color = (0, 255, 0)  # Green
                    elif left_knee_angle < 70 or right_knee_angle < 70:
                        feedback = "⚠️ Too Deep! Stop at 90° angle."
                        color = (0, 0, 255)  # Red
                    else:
                        feedback = "🔄 Going down... Bend to 90°."
                        color = (255, 255, 0)  # Cyan
                else:
                    # Standing straight
                    feedback = "🧍 Stand Straight. Bend your knees to 90°."
                    color = (255, 255, 0)  # Yellow

                # Override feedback if leaning heavily
                if is_leaning and "PERFECT" not in feedback:
                    feedback = "🔴 Lean Back! Keep your chest up."

                # ----- DISPLAY ANGLES ON SCREEN (ENGLISH) -----
                cv2.putText(img, f"Left Knee: {int(left_knee_angle)} deg", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(img, f"Right Knee: {int(right_knee_angle)} deg", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # ----- DRAW FEEDBACK TEXT (LARGE) -----
                cv2.putText(img, feedback, (20, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                
                # ----- DRAW SKELETON ON BODY -----
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                cv2.putText(img, "Error detecting pose. Try repositioning.", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, "❌ No Body Detected. Stand clearly in frame.", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------ STREAMLIT WEBRTC (LIVE CAMERA) ------------------
# STUN server config for smooth streaming
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="posture-coach",
    video_processor_factory=PoseProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
)

# ------------------ FOOTER ------------------
st.caption("💡 Tip: Ensure proper lighting. The AI works best when your full body is visible.")
st.success("📧 Ready to sell to Government? Save this demo video and show it to your local District Hospital's Physiotherapy department.")
