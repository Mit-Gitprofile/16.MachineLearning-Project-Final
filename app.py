import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
import time

# ---------------- PATH CONFIG ----------------
BASE = r"F:\Github_Desktop\16.MachineLearning-Project-Final"
MODEL_PATH = "F:\Github_Desktop\16.MachineLearning-Project-Final\model\face_model.h5"
DATASET_PATH = BASE + r"\dataset"
ATT_FILE = BASE + r"\attendance.csv"
USER_FILE = BASE + r"\users.csv"
# -------------------------------------------

st.set_page_config(page_title="AI Attendance", layout="wide")

# ---------------- BEAUTIFUL UI ----------------
st.markdown("""
<style>
body{background:linear-gradient(135deg,#0f172a,#020617);}
h1,h2{color:#38bdf8;}
.stButton>button{
background:linear-gradient(90deg,#22c55e,#06b6d4);
color:white;border-radius:10px;height:45px;width:100%;
}
</style>
""",unsafe_allow_html=True)

# ---------------- FILE AUTO CREATE ----------------
if not os.path.exists(USER_FILE) or os.stat(USER_FILE).st_size==0:
    pd.DataFrame(columns=["username","password"]).to_csv(USER_FILE,index=False)

if not os.path.exists(ATT_FILE):
    pd.DataFrame(columns=["Name","Date","Time"]).to_csv(ATT_FILE,index=False)

# ---------------- SESSION ----------------
if "login" not in st.session_state:
    st.session_state.login=False

if "cam" not in st.session_state:
    st.session_state.cam=False

# ---------------- LOGIN + REGISTER ----------------
users=pd.read_csv(USER_FILE)

def save_user(u,p):
    users.loc[len(users)] = [u,p]
    users.to_csv(USER_FILE,index=False)

if not st.session_state.login:

    st.title("🔐 AI Attendance Login")

    c1,c2=st.columns(2)

    with c1:
        st.subheader("Login")
        lu=st.text_input("Username")
        lp=st.text_input("Password",type="password")

        if st.button("Login"):
            if ((users["username"]==lu)&(users["password"]==lp)).any():
                st.session_state.login=True
                st.rerun()
            else:
                st.error("Invalid Login")

    with c2:
        st.subheader("Register")
        ru=st.text_input("New Username")
        rp=st.text_input("New Password",type="password")

        if st.button("Create Account"):
            if ru in users["username"].values:
                st.warning("Username exists")
            elif ru=="" or rp=="":
                st.warning("Fill all fields")
            else:
                save_user(ru,rp)
                st.success("Account Created")

    st.stop()

# ---------------- MAIN APP ----------------
st.title("🤖 AI Face Attendance System")

@st.cache_resource
def load_model():
    if not os.path.isfile("model/face_model.h5"):
        st.error("❌ face_model.h5 not found in model folder.")
        st.stop()
    return tf.keras.models.load_model("model/face_model.h5", compile=False)

model = load_model()


def get_labels():
    return os.listdir(DATASET_PATH)

face_cascade=cv2.CascadeClassifier(
cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

if st.sidebar.button("▶ Start Camera"):
    st.session_state.cam=True

if st.sidebar.button("⏹ Stop Camera"):
    st.session_state.cam=False

if st.sidebar.button("🚪 Logout"):
    st.session_state.login=False
    st.session_state.cam=False
    st.rerun()

if st.sidebar.button("🗑 Clear Attendance"):
    pd.DataFrame(columns=["Name","Date","Time"]).to_csv(ATT_FILE,index=False)

new_person=st.sidebar.text_input("➕ Add Person Name")
capture_faces=st.sidebar.button("📸 Add Person")

FRAME=st.image([])

# ---------------- ADD PERSON ----------------
# ---------------- ADD PERSON (CLOUD SAFE) ----------------
if capture_faces and new_person != "":

    person_path = os.path.join(DATASET_PATH, new_person)
    os.makedirs(person_path, exist_ok=True)

    img = st.camera_input("📸 Capture face images (20 samples)")

    if img is not None:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("❌ Failed to read image")
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            count = len(os.listdir(person_path))
            cv2.imwrite(f"{person_path}/{count}.jpg", gray)

            st.success(f"✅ Image {count+1}/20 saved")

            if count + 1 >= 20:
                st.success("🎉 Person added successfully!")
                st.warning("⚠ Please retrain the model")


# ---------------- ATTENDANCE (CLOUD SAFE) ----------------
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

today = datetime.now(IST).strftime("%Y-%m-%d")
time_now = datetime.now(IST).strftime("%H:%M:%S")

df_att = pd.read_csv(ATT_FILE)
marked = set(df_att[df_att["Date"] == today]["Name"].values)

if st.session_state.cam:

    img = st.camera_input("📷 Capture face for attendance")

    if img is None:
        st.info("Camera ready. Please capture an image.")
    else:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("❌ Failed to decode image")
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            labels = get_labels()

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (100, 100)) / 255.0
                face = face.reshape(1, 100, 100, 1)

                pred = model.predict(face, verbose=0)
                conf = np.max(pred) * 100
                idx = np.argmax(pred)

                # ---------- UNKNOWN PERSON ----------
                if conf < 80 or idx >= len(labels):
                    name = "Unknown"
                    color = (0, 0, 255)  # Red
                    label_text = "Unknown Person"

                # ---------- KNOWN PERSON ----------
                else:
                    name = labels[idx]
                    color = (0, 255, 0)  # Green

                    if name in marked:
                        label_text = f"{name} (Already Marked)"
                        st.warning(f"⚠ {name} already marked today")
                    else:
                        df_att.loc[len(df_att)] = [
                            name,
                            today,
                            time_now
                        ]
                        df_att.to_csv(ATT_FILE, index=False)
                        marked.add(name)
                        label_text = f"{name} (Marked)"
                        st.success(f"✅ Attendance marked for {name}")

                # ---------- DRAW ROUND FACE ----------
                center = (x + w // 2, y + h // 2)
                radius = w // 2
                cv2.circle(frame, center, radius, color, 2)

                cv2.putText(
                    frame,
                    label_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            FRAME.image(frame, channels="BGR")


# ---------------- DASHBOARD ----------------
st.markdown("---")
st.subheader("📋 Attendance Dashboard")

df=pd.read_csv(ATT_FILE)
st.dataframe(df,use_container_width=True)

st.download_button("⬇ Download CSV",
df.to_csv(index=False),
"attendance.csv","text/csv")

st.success("System Ready 🚀")
