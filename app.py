import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import pandas as pd
from datetime import datetime

# ---------- पेज कॉन्फ़िगरेशन (वाइड लेआउट) ----------
st.set_page_config(
    page_title="National AI Physiotherapy Portal",
    page_icon="🏛️",  # Fallback, but we will override with SVG in header
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- कस्टम CSS (Govt Healthcare Theme) ----------
st.markdown("""
<style>
    /* सरकारी ब्लू + ग्रीन थीम */
    :root {
        --primary-blue: #003366;
        --primary-green: #27AE60;
        --gold-accent: #F1C40F;
        --light-bg: #F4F7FA;
    }
    /* हेडर को हटाएं */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background-color: #F4F7FA;
    }
    /* साइडबार को सरकारी लुक दें */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #003366 !important;
        color: white !important;
    }
    .css-1d391kg .stSelectbox label, .css-1d391kg .stRadio label {
        color: white !important;
    }
    /* कार्ड स्टाइल */
    .gov-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 6px solid #27AE60;
        margin-bottom: 1rem;
    }
    .gov-card-blue {
        border-left: 6px solid #003366;
    }
    .gov-header {
        background: linear-gradient(135deg, #003366 0%, #1A5276 100%);
        padding: 1.2rem 2rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
        color: white;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 4px 15px rgba(0,51,102,0.3);
    }
    .gov-footer {
        background: #003366;
        color: #B0C4DE;
        padding: 1.5rem 2rem;
        border-radius: 20px 20px 0 0;
        margin-top: 3rem;
        text-align: center;
        font-size: 0.9rem;
        border-top: 4px solid #27AE60;
    }
    .stat-box {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .stat-box h3 {
        color: #003366;
        margin: 0;
        font-size: 2rem;
    }
    .stat-box p {
        color: #7F8C8D;
        margin: 0;
        font-weight: 500;
    }
    /* फीडबैक टेक्स्ट */
    .feedback-text {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        background: #FEF9E7;
        border-left: 5px solid #F1C40F;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SVG आइकॉन (इमोजी की जगह) ----------
def svg_icon(name, size=24, color="#FFFFFF"):
    icons = {
        "gov_logo": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>',
        "health": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M22 12h-4l-3 9-4-18-3 9H2"/></svg>',
        "camera": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>',
        "user": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        "reports": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
        "settings": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>',
        "check": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>',
        "alert": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
        "arrow_right": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>'
    }
    return icons.get(name, "")

# ---------- हेडर (Govt National Healthcare) ----------
st.markdown(f"""
<div class="gov-header">
    <div style="display:flex; align-items:center; gap:15px;">
        <div style="background:#27AE60; padding:10px; border-radius:50%;">
            {svg_icon("gov_logo", 40, "#FFFFFF")}
        </div>
        <div>
            <h2 style="margin:0; font-weight:300; letter-spacing:1px;">🇮🇳 NATIONAL HEALTH MISSION</h2>
            <h1 style="margin:0; font-size:1.8rem; font-weight:700;">AI Physiotherapy Assistance Program</h1>
            <p style="margin:0; opacity:0.8; font-size:0.9rem;">Ministry of Health & Family Welfare | Govt. of India</p>
        </div>
    </div>
    <div style="margin-left:auto; text-align:right; background:rgba(255,255,255,0.1); padding:0.5rem 1.5rem; border-radius:30px;">
        <span style="font-size:0.8rem;">v2.0 | Secure</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- साइडबार (नेविगेशन) ----------
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 20px 0; border-bottom: 2px solid #27AE60;">
        <div style="background:#27AE60; width:80px; height:80px; border-radius:50%; margin:0 auto; display:flex; align-items:center; justify-content:center;">
            {svg_icon("user", 50, "#FFFFFF")}
        </div>
        <h4 style="color:white; margin-top:10px;">Dr. R. Sharma</h4>
        <p style="color:#B0C4DE; font-size:0.8rem;">Physiotherapist | ID: NHM-7421</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # नेविगेशन (राज्य प्रबंधन)
    page = st.radio(
        label="Navigation",
        options=["📋 Dashboard", "🎥 Start Session", "📊 Patient Reports", "⚙️ Settings"],
        index=0,
        key="nav_radio"
    )
    # रेडियो के लेबल से इमोजी हटाओ (हम SVG का इस्तेमाल करेंगे बटन के लिए, लेकिन Streamlit radio में emoji default है, हम इसे अभी के लिए रखते हैं क्योंकि इसे बदलना मुश्किल है, लेकिन हम मुख्य कंटेंट में SVG दिखाएंगे)
    # असल में हम page वेरिएबल से काम चलाएंगे।

# ---------- मुख्य कंटेंट (पेज हैंडलिंग) ----------
if "Dashboard" in page:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <h3>1,284</h3>
            <p>{svg_icon("user", 20, "#003366")} Total Patients</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <h3>87%</h3>
            <p>{svg_icon("check", 20, "#27AE60")} Recovery Rate</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <h3>342</h3>
            <p>{svg_icon("camera", 20, "#003366")} Active Sessions</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <h3>52</h3>
            <p>{svg_icon("reports", 20, "#27AE60")} Reports Generated</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div class="gov-card">
        <h3>{svg_icon("health", 24, "#27AE60")} AI Posture Coach Overview</h3>
        <p>This system uses Google MediaPipe AI to track knee angles and back posture during physiotherapy sessions. 
        Patients can perform squats at home while the AI provides real-time feedback, reducing the need for physical visits.</p>
        <p><strong>Current Status:</strong> <span style="color:#27AE60;">● Online</span> | Model: Pose Detection v0.10.8</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 Click on 'Start Session' from the sidebar to begin the live physiotherapy session.")

elif "Start Session" in page:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">
        <div style="background:#27AE60; padding:8px 15px; border-radius:30px;">
            <span style="color:white; font-weight:bold;">LIVE</span>
        </div>
        <h2 style="margin:0;">{svg_icon("camera", 28, "#003366")} Real-Time Posture Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # कोर AI कैमरा (पुराना कोड ही यहाँ आएगा)
    # MediaPipe पोज़ फंक्शन
    # ---------- MEDIAPIPE सेटअप ----------
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

    def calculate_angle(a, b, c):
        a = np.array(a); b = np.array(b); c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    class PoseProcessor(VideoTransformerBase):
        def __init__(self):
            self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            h, w, _ = img.shape
            feedback = "🧍 Stand in front of camera"  # Keep for overlay, but we use SVG in UI
            color = (0, 255, 255)
            l_ang = r_ang = 0

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                try:
                    l_hip = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].x*w, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y*h)
                    l_kn = (lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x*w, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y*h)
                    l_ank = (lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x*w, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y*h)
                    r_hip = (lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x*w, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y*h)
                    r_kn = (lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x*w, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y*h)
                    r_ank = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x*w, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y*h)
                    l_ang = calculate_angle(l_hip, l_kn, l_ank)
                    r_ang = calculate_angle(r_hip, r_kn, r_ank)
                    
                    if l_ang < 160 and r_ang < 160:
                        if 70 < l_ang < 110 and 70 < r_ang < 110:
                            feedback, color = "✅ PERFECT SQUAT!", (0,255,0)
                        elif l_ang < 70 or r_ang < 70:
                            feedback, color = "⚠️ TOO DEEP! Stop at 90°", (0,0,255)
                        else:
                            feedback, color = "🔄 Bend to 90°", (255,255,0)
                    else:
                        feedback, color = "🧍 Stand Straight. Bend knees.", (255,255,0)
                    
                    cv2.putText(img, f"L: {int(l_ang)} deg | R: {int(r_ang)} deg", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(img, feedback, (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                except: pass
            else:
                cv2.putText(img, "No Body Detected. Stand clearly.", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    # लाइव कैमरा दिखाने से पहले पेशेंट डिटेल्स
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(f"""
        <div style="background:white; padding:15px; border-radius:10px; border-left:5px solid #003366;">
            <p style="margin:0;"><strong>Patient ID:</strong> PT-2024-0087</p>
            <p style="margin:0;"><strong>Treatment:</strong> Quadriceps Strengthening (Squats)</p>
            <p style="margin:0;"><strong>Session:</strong> #12 of 20</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div style="background:#27AE60; color:white; padding:15px; border-radius:10px; text-align:center;">
            <h3 style="margin:0;">{svg_icon("check", 30, "#FFFFFF")}</h3>
            <p style="margin:0;">AI Tracking Active</p>
        </div>
        """, unsafe_allow_html=True)

    # कैमरा विजेट
    webrtc_streamer(
        key="gov-posture-coach",
        video_processor_factory=PoseProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    )

elif "Reports" in page:
    st.markdown(f"<h2>{svg_icon('reports', 30, '#003366')} Patient Progress Reports</h2>", unsafe_allow_html=True)
    st.markdown("---")
    data = {
        "Date": ["2026-08-10", "2026-08-12", "2026-08-14", "2026-08-16", "2026-08-18"],
        "Left Knee Angle (avg)": [120, 105, 95, 88, 82],
        "Right Knee Angle (avg)": [115, 100, 92, 85, 80],
        "Sessions Done": [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    st.line_chart(df.set_index("Date")[["Left Knee Angle (avg)", "Right Knee Angle (avg)"]])
    st.markdown("""
    <div style="background:white; padding:1rem; border-radius:10px; margin-top:20px;">
        <h4>📋 Detailed Assessment</h4>
        <p><span style="color:#27AE60;">●</span> Patient is showing consistent improvement in knee flexion.</p>
        <p><span style="color:#F1C40F;">●</span> Right leg is 5% weaker than left. Recommend focused exercises.</p>
        <p><span style="color:#003366;">●</span> Predicted full recovery in 8 sessions.</p>
    </div>
    """, unsafe_allow_html=True)

else:  # Settings
    st.markdown(f"<h2>{svg_icon('settings', 30, '#003366')} System Settings</h2>", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="gov-card gov-card-blue">
            <h3>🖥️ Device Configuration</h3>
            <p>Camera: Integrated Webcam</p>
            <p>Resolution: 640x480</p>
            <p>Frame Rate: 30 FPS</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="gov-card">
            <h3>📡 API & Integration</h3>
            <p>MediaPipe Version: 0.10.8</p>
            <p>Python Runtime: 3.10</p>
            <p>Status: <span style="color:#27AE60;">● All Systems Operational</span></p>
        </div>
        """, unsafe_allow_html=True)

# ---------- फुटर (Govt स्टाइल) ----------
st.markdown(f"""
<div class="gov-footer">
    <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
        <div>© 2026 Ministry of Health & Family Welfare, Govt. of India</div>
        <div>National AI Physiotherapy Program | Version 2.0</div>
        <div>Helpline: 1800-11-1800</div>
    </div>
    <div style="margin-top:10px; font-size:0.8rem; opacity:0.6;">
        This system is strictly for authorized healthcare personnel. All data is encrypted and secured.
    </div>
</div>
""", unsafe_allow_html=True)
