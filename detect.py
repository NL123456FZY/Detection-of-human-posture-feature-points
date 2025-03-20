import cv2
import mediapipe as mp
import numpy as np

from json import dump


data = []

def run(cap,image_callback,signal_callback,is_data_save=False,accmax=60,vmax=10):
    """
    image_callback: 
        将每一帧传回去的回调函数
    signal_callback: 
        传回跌倒信号的回调函数
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    lastx1,lasty1 = None,None
    lastx2,lasty2 = None,None


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in landmarks]
            p = np.array(points)
            mx, my = np.mean(p[:, 0]), np.mean(p[:, 1])
            if all((lastx1, lasty1, lastx2, lasty2)):
                vx1, vy1 = lastx1 - lastx2, lasty1 - lasty2
                vx2, vy2 = mx - lastx1, my - lasty1
                accx, accy = vx2 - vx1, vy2 - vy1

                if is_data_save:
                    data.append([accx, accy])

                print(accy,vy2)
                
                if accy > accmax and vy2 < vmax:
                    signal_callback()

                cv2.line(frame, (int(lastx1), int(lasty1)), (int(mx), int(my)), (0, 255, 0), 2)
            
            lastx2, lasty2 = lastx1, lasty1
            lastx1, lasty1 = mx, my

            # for i, (x, y) in enumerate(points):
            #     print(f"Landmark {i}: ({x}, {y})")
            #     cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        image_callback(frame)


    if is_data_save:
        with open("data.json", "w",encoding="utf-8") as f:
            dump(data,f)