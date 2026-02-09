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
MODEL_PATH = BASE + r"\model\face_model.h5"
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
    return tf.keras.models.load_model(MODEL_PATH)

model=load_model()

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
if capture_faces and new_person!="":
    path=os.path.join(DATASET_PATH,new_person)
    os.makedirs(path,exist_ok=True)
    cam=cv2.VideoCapture(0)
    st.info("Capturing 20 images...")
    c=0
    while c<20:
        ret,frame=cam.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{path}/{c}.jpg",gray)
        FRAME.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        c+=1
        time.sleep(0.2)
    cam.release()
    st.success("Person Added. Retrain model!")

# ---------------- ATTENDANCE ----------------
today=datetime.now().strftime("%Y-%m-%d")
marked=set(pd.read_csv(ATT_FILE)["Name"].values)

if st.session_state.cam:
    cam=cv2.VideoCapture(0)

    while st.session_state.cam:
        ret,frame=cam.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for(x,y,w,h) in faces:
            face=gray[y:y+h,x:x+w]
            face=cv2.resize(face,(100,100))/255.0
            face=face.reshape(1,100,100,1)

            pred=model.predict(face,verbose=0)
            labels=get_labels()
            name=labels[np.argmax(pred)]
            conf=np.max(pred)*100

            if conf>80:
                if name in marked:
                    st.warning(f"{name} already marked today")
                else:
                    df=pd.read_csv(ATT_FILE)
                    df.loc[len(df)]=[
                    name,today,
                    datetime.now().strftime("%H:%M:%S")]
                    df.to_csv(ATT_FILE,index=False)
                    marked.add(name)

            cv2.putText(frame,name,(x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        FRAME.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    cam.release()

# ---------------- DASHBOARD ----------------
st.markdown("---")
st.subheader("📋 Attendance Dashboard")

df=pd.read_csv(ATT_FILE)
st.dataframe(df,use_container_width=True)

st.download_button("⬇ Download CSV",
df.to_csv(index=False),
"attendance.csv","text/csv")

st.success("System Ready 🚀")
