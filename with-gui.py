import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe and labels
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

# Streamlit app title
st.title("Sign Language Detection with Streamlit GUI")

# Display introductory image with caption
col1, col2 = st.columns([1, 2])  # Create two columns to position the image on the left
with col1:
    image = Image.open("img.jpg")
    st.image(image, caption="This are the sign !", width=220)  # Adjust width as needed

with col2:
    st.write("## Welcome to Sign Language Detection!")
    st.write("Press Start to begin using the sign language detector.")

# Buttons for controlling prediction
start_prediction = st.button("Start Prediction")
stop_prediction = st.button("Stop Prediction")

# Placeholder for video frame
stframe = st.empty()

# Initialize video capture outside the loop to retain state
cap = None
if start_prediction:
    # Open video capture when Start is pressed
    cap = cv2.VideoCapture(0)

while start_prediction and cap and cap.isOpened():
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        st.write("Error: Could not read frame.")
        break

    # Convert frame for Mediapipe processing
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Bounding box coordinates
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw bounding box and prediction on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display frame in Streamlit
    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Stop if the Stop button is pressed
    if stop_prediction:
        break

# Release resources if the Stop button was pressed or loop ended
if cap:
    cap.release()
cv2.destroyAllWindows()
