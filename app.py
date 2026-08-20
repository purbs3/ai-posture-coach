import os
# यह लाइन Protobuf को Python मोड में चलाती है (C++ वाली टक्कर खत्म)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# ---------- MEDIAPIPE को स्टैण्डर्ड तरीके से इम्पोर्ट करें ----------
import mediapipe as mp

# अब 'mp.solutions' पूरी तरह से उपलब्ध होगा (Python 3.11 + Protobuf 4.25.3 पर)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Posture Coach - Physiotherapy", page_icon="🏋️", layout="centered")

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
    """तीन बिंदुओं (a, b, c) के बीच का कोण डिग्री में निकालें।"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ------------------ VIDEO PROCESSOR CLASS (AI इंजन) ------------------
class PoseProcessor(VideoTransformerBase):
    def __init__(self):
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
                # ----- बाएं हिस्से के अंक -----
                l_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, 
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)
                l_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, 
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)
                l_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, 
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h)
                l_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, 
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h)

                # ----- दाहिने हिस्से के अंक -----
                r_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, 
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)
                r_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)
                r_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)
                r_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)

                # ----- घुटनों के कोण -----
                left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

                # ----- पीठ की जांच -----
                shoulder_hip_diff = abs(l_shoulder[0] - l_hip[0]) 
                is_leaning = shoulder_hip_diff > 30  

                # ----- फीडबैक -----
                if left_knee_angle < 160 and right_knee_angle < 160:
                    if 70 < left_knee_angle < 110 and 70 < right_knee_angle < 110:
                        if is_leaning:
                            feedback = "⚠️ गहराई सही है! लेकिन पीठ पीछे झुकाएं!"
                            color = (0, 165, 255)
                        else:
                            feedback = "✅ बिल्कुल सही SQUAT! पीठ एकदम सीधी!"
                            color = (0, 255, 0)
                    elif left_knee_angle < 70 or right_knee_angle < 70:
                        feedback = "⚠️ बहुत गहरा! 90° पर रुकें (घुटनों को चोट लग सकती है)"
                        color = (0, 0, 255)
                    else:
                        feedback = "🔄 नीचे जा रहे हैं... 90° तक मोड़ें।"
                        color = (255, 255, 0)
                else:
                    feedback = "🧍 सीधे खड़े रहें। घुटनों को 90° तक मोड़ें।"
                    color = (255, 255, 0)

                if is_leaning and "सही" not in feedback and "गहराई" not in feedback:
                    feedback = "🔴 पीठ पीछे झुकाएं! सीना बाहर रखें।"

                # ----- डिस्प्ले -----
                cv2.putText(img, f"Left Knee: {int(left_knee_angle)} deg", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(img, f"Right Knee: {int(right_knee_angle)} deg", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                cv2.putText(img, feedback, (20, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                cv2.putText(img, "पोज़ डिटेक्ट करने में परेशानी। थोड़ा पीछे हटें।", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, "❌ शरीर नहीं दिख रहा। कृपया स्पष्ट रूप से खड़े हों।", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------ STREAMLIT WEBCAM ------------------
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
st.success("📧 अब आपका AI पोस्चर कोच तैयार है! इसका डेमो वीडियो रिकॉर्ड करके सरकारी अस्पताल के फिजियोथेरेपी विभाग को दिखाएं।")
