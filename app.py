import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

import os
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from scipy import stats

mp_holistic = mp.solutions.holistic  # Holistic Model (To make Detections)
mp_drawing = mp.solutions.drawing_utils # Drawing Utilities (To Draw Detections)

# mp_drawing = mp.solutions.drawing_utils
# mp_face_mesh = mp.solutions.face_mesh

st.title("Sign Language Detection")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded=true] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded=false] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True
)

st.sidebar.title("SignLang Sidebar")
st.sidebar.subheader("parameters")

@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))
    
    #Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

app_mode = st.sidebar.selectbox('Choose the App Mode',
                                   ['About App', 'Run on Video']
                                   )

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR Conversion BGR to RGB
    image.flags.writeable = False                  # Image is no longer Writeable  
    results = model.process(image)                 # Make Prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color Conversion RGB to BGR
    
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# 1. New detection variables
# sequence = []
# sentence = []
# predictions = []
# # threshold = 0.7
# threshold = 0.8

# actions = np.array(['hello', 'thanks'])
actions = np.array(['name','_','thanks','hello','is'])

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("action.h5")

# res = [.7, 0.2, 0.1]

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5



if app_mode == 'About App':
    st.markdown('In this Application we are using OpenCV to Detect Sign Language')

elif app_mode == 'Run on Video':
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")

    if record:
        st.checkbox("Recording", value=True)
    
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded=true] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded=false] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html = True
    )

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=5, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    st.markdown('## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    ##We get our input video here
    if not video_file_buffer:
        # if use_webcam:
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(0)
        # else:
            # vid = cv2.VideoCapture(DEMO_VIDEO)
            # tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        cap = cv2.VideoCapture(tffile.name)

    kpi1, kpi2 = st.columns(2)    

    with kpi1:
        kpi1_text = st.markdown("0")
    
    with kpi2:
        kpi2_text = st.markdown("0")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            # sequence = sequence[-30:]
            sequence = sequence[-20:]
            
            # if len(sequence) == 30:
            if len(sequence) == 20:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                # image = prob_viz(res, actions, image, colors)
                
            # cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            # cv2.putText(image, ' '.join(sentence), (3,30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # print(sentence)
            try:
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{sentence[-1]}</h1>", unsafe_allow_html=True)
                # st.markdown(f"{sentence[-1]}", unsafe_allow_html=False)
            except Exception as e:
                pass

            frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)

            # Show to screen
            # cv2.imshow('OpenCV Feed', image)

        st.subheader('Output Image')