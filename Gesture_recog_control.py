#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mediapipe as mp 
import cv2
import time


# In[3]:


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize mediapipe hand object
hands = mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)


# In[7]:


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open Video Capture.")
    exit()

def is_thumbs_up(hand_landmarks):
    # Thumb tip(landmark number 4) should be above thumb MCP (landmark number 2)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]

    # Check if thumb is up
    if thumb_tip.y < thumb_mcp.y:
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read the frame")
        break

    flipped_frame = cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check for thumbs up gesture
            if is_thumbs_up(hand_landmarks):
                cv2.putText(frame_rgb, "Keep up the spirit!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame_rgb, "Come on, don't lose HOPE!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Convert back to BGR for displaying with OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Tracking', frame_bgr)

    if cv2.waitKey(5) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




