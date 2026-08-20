import os
# सबसे जरूरी लाइन - यह MediaPipe को Python मोड में चलाती है (C++ वाली टक्कर खत्म)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# ---------- MEDIAPIPE को ऐसे इम्पोर्ट करें जो कभी एरर न दे ----------
# 'mp.solutions' को छोड़कर सीधे उसके अंदरूनी पाइथन मॉड्यूल को बुलाओ
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Physiotherapy Posture Coach", page_icon="🏋️", layout="centered")

st.title("🏋️ AI Posture Coach - Physiotherapy Assistant")
st.markdown("**अपने कैमरे के सामने Squat (बैठक-उठक) करें। AI आपके घुटनों के कोण और पीठ की मुद्रा को रियल-टाइम में ट्रैक करेगा!**")

# ------------------ SIDEBAR INSTRUCTIONS ------------------
with st.sidebar:
    st.header("📋 उपयोग करने का तरीका")
    st.markdown("""
    1. नीचे **START** बटन दबाएँ।
    2. कैमरा एक्सेस की अनुमति दें।
    3. **कम से कम 6 फीट** पीछे खड़े हों ताकि पूरा शरीर दिखे।
    4. **Squat** करें (जैसे किसी कुर्सी पर बैठ रहे हों)।
    5. स्क्रीन पर रियल-टाइम फीडबैक देखें।
    """)
    st.warning("**🏆 सही Squat**: घुटनों को 90° तक मोड़ें और पीठ सीधी रखें।")
    st.info("💡 अगर स्क्रीन काली आए, तो ऐप को रीस्टार्ट करें या कैमरा परमिशन चेक करें।")

# ------------------ ANGLE CALCULATION FUNCTION ------------------
def calculate_angle(a, b, c):
    """तीन बिंदुओं के बीच का कोण (डिग्री में) निकालें।"""
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
        # पोज़ डिटेक्टर को इनिशियलाइज़ करें (बिना mp.solutions के)
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        h, w, _ = img.shape
        feedback = "🧍 कृपया कैमरे के सामने खड़े हो जाएं"
        color = (0, 255, 255)  # Yellow
        left_knee_angle = 0
        right_knee_angle = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # ----- शरीर के अंक (Landmarks) निकालें -----
                # Left side (बायां हिस्सा)
                l_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, 
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)
                l_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, 
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)
                l_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, 
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h)
                l_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, 
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h)

                # Right side (दाहिना हिस्सा)
                r_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)
                r_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)
                r_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)
                r_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)

                # ----- घुटनों के कोण (Angle) कैलकुलेट करें -----
                left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

                # ----- पीठ की मुद्रा चेक करें (क्या झुक रहे हैं?) -----
                shoulder_hip_diff = abs(l_shoulder[0] - l_hip[0]) 
                is_leaning = shoulder_hip_diff > 30  

                # ----- फीडबैक लॉजिक (Squat कोच) -----
                if left_knee_angle < 160 and right_knee_angle < 160:
                    if 70 < left_knee_angle < 110 and 70 < right_knee_angle < 110:
                        if is_leaning:
                            feedback = "⚠️ गहराई सही है! लेकिन पीठ पीछे झुकाएं!"
                            color = (0, 165, 255)  # Orange
                        else:
                            feedback = "✅ बिल्कुल सही SQUAT! पीठ एकदम सीधी!"
                            color = (0, 255, 0)  # Green
                    elif left_knee_angle < 70 or right_knee_angle < 70:
                        feedback = "⚠️ बहुत गहरा! 90° पर रुकें (घुटनों को चोट लग सकती है)"
                        color = (0, 0, 255)  # Red
                    else:
                        feedback = "🔄 नीचे जा रहे हैं... 90° तक मोड़ें।"
                        color = (255, 255, 0)  # Cyan
                else:
                    feedback = "🧍 सीधे खड़े रहें। घुटनों को 90° तक मोड़ें।"
                    color = (255, 255, 0)  # Yellow

                if is_leaning and "सही" not in feedback:
                    feedback = "🔴 पीठ पीछे झुकाएं! सीना बाहर रखें।"

                # ----- स्क्रीन पर एंगल दिखाएं (हिंदी + English) -----
                cv2.putText(img, f"Left Knee: {int(left_knee_angle)} deg", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(img, f"Right Knee: {int(right_knee_angle)} deg", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # ----- बड़ा फीडबैक टेक्स्ट दिखाएं -----
                cv2.putText(img, feedback, (20, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                
                # ----- शरीर पर कंकाल (स्केलेटन) बनाएं -----
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                cv2.putText(img, "पोज़ डिटेक्ट करने में परेशानी। थोड़ा पीछे हटें।", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, "❌ शरीर नहीं दिख रहा। कृपया स्पष्ट रूप से खड़े हों।", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------ STREAMLIT WEBRTC (LIVE CAMERA) ------------------
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
st.caption("💡 टिप: अच्छी रोशनी रखें। AI तब सबसे अच्छा काम करता है जब आपका पूरा शरीर दिखे।")
st.success("📧 अब आपका AI पोस्चर कोच तैयार है! इसका डेमो वीडियो रिकॉर्ड करके सरकारी अस्पताल के फिजियोथेरेपी विभाग को दिखाएं।")    """Calculate the angle between three points (in degrees)."""
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
