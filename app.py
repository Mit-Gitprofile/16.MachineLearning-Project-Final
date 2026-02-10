import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
import time

# ================= PATHS =================
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "model", "face_model.h5")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
ATT_FILE = os.path.join(BASE_DIR, "attendance.csv")
USER_FILE = os.path.join(BASE_DIR, "users.csv")

os.makedirs(DATASET_PATH, exist_ok=True)

# ================= PAGE =================
st.set_page_config("AI Face Attendance", layout="wide")

st.markdown("""
<style>
body{background:#020617;}
h1,h2{color:#38bdf8;}
.stButton>button{
background:#22c55e;color:white;
border-radius:8px;height:45px;width:100%;
}
</style>
""", unsafe_allow_html=True)

# ================= FILE CREATE =================
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username","password"]).to_csv(USER_FILE,index=False)

if not os.path.exists(ATT_FILE):
    pd.DataFrame(columns=["Name","Date","Time"]).to_csv(ATT_FILE,index=False)

# ================= SESSION =================
if "login" not in st.session_state:
    st.session_state.login = False
if "camera" not in st.session_state:
    st.session_state.camera = False

# ================= LOGIN =================
users = pd.read_csv(USER_FILE)

def save_user(u,p):
    users.loc[len(users)] = [u,p]
    users.to_csv(USER_FILE,index=False)

if not st.session_state.login:
    st.title("🔐 Login")

    c1,c2 = st.columns(2)

    with c1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if ((users.username==u)&(users.password==p)).any():
                st.session_state.login=True
                st.rerun()
            else:
                st.error("Invalid Login")

    with c2:
        nu = st.text_input("New Username")
        npw = st.text_input("New Password", type="password")
        if st.button("Register"):
            if nu in users.username.values:
                st.warning("User exists")
            else:
                save_user(nu,npw)
                st.success("Account Created")

    st.stop()

# ================= MAIN =================
st.title("🤖 AI Face Attendance System")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

labels = sorted(os.listdir(DATASET_PATH))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= SIDEBAR =================
st.sidebar.header("Controls")

if st.sidebar.button("▶ Start Camera"):
    st.session_state.camera = True

if st.sidebar.button("⏹ Stop Camera"):
    st.session_state.camera = False

if st.sidebar.button("🚪 Logout"):
    st.session_state.login=False
    st.session_state.camera=False
    st.rerun()

# ================= CAMERA =================
FRAME = st.image([])
today = datetime.now().strftime("%Y-%m-%d")

if st.session_state.camera:
    cap = cv2.VideoCapture(0)

    while st.session_state.camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        df = pd.read_csv(ATT_FILE)
        marked = set(df[df.Date==today].Name.values)

        for x,y,w,h in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(100,100))/255.0
            face = face.reshape(1,100,100,1)

            pred = model.predict(face,verbose=0)
            idx = np.argmax(pred)
            name = labels[idx]
            conf = np.max(pred)*100

            if conf > 80 and name not in marked:
                df.loc[len(df)] = [
                    name,
                    today,
                    datetime.now().strftime("%H:%M:%S")
                ]
                df.to_csv(ATT_FILE,index=False)
                marked.add(name)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,name,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        FRAME.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        time.sleep(0.03)

    cap.release()

# ================= DASHBOARD =================
st.markdown("---")
st.subheader("📋 Attendance Records")

df = pd.read_csv(ATT_FILE)
st.dataframe(df,use_container_width=True)

st.download_button(
    "⬇ Download CSV",
    df.to_csv(index=False),
    "attendance.csv"
)

st.success("🚀 System Ready")
