import cv2
import mediapipe as mp
import numpy as np

gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}

cap = cv2.VideoCapture(0)

#Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#mediapipe 중 손의 관절 위치를 인식할 수 있는 모델 초기화
hands = mp_hands.Hands(
    max_num_hands = 1, # 몇 개의 손을 인식할 것이냐
    min_detection_confidence = 0.5, # 탐지 임계치
    min_tracking_confidence = 0.5 # 추정 임계치
)
 
file = np.genfromtxt('gesture_train.csv', delimiter=',') # 파일을 읽어온다
angle = file[:, :-1].astype(np.float32) # 0번 idx부터 마지막 idx전까지 사용하기
label = file[:, -1].astype(np.float32) # 마지막 idx만 사용하기

knn = cv2.ml.KNearest_create() # knn 모델 초기화
knn.train(angle, cv2.ml.ROW_SAMPLE, label) # knn 학습

while cap.isOpened():
    ret, img = cap.read()
    if not ret: break

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img) # 프레임에서 손, 관절의 위치를 탐색한다.

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR )
    
    if result.multi_hand_landmarks is not None: # 만약 손이 정상적으로 인식 되었을 때
        for res in result.multi_hand_landmarks: # 여러 손일 경우 루프를 사용한다.
            joint = np.zeros((21,3))

            for j, lm in enumerate(res.landmark): # 21개의 랜드마크를 한 점씩 반복문을 사용해서 처리한다.
                joint[j] = [lm.x, lm.y, lm.z]
            
            # 관절 사이의 각도를 계산한다.
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint

            v = v2 - v1 # 팔목과 각 손가락 관절 사이의 벡터를 구한다.

            v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=-1) # 단위벡터를 구한다.(벡터/벡터의 길이) 
            
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # 단위 벡터를 내적한 값에 accos 값을 구하면 관절 사이의 각도를 구할 수 있다.

            angle = np.degrees(angle) # 라디안 -> 도 
            angle = np.expand_dims(angle.astype(np.float32), axis=0) # 머신러닝 모델에 넣어서 추론할 때는 항상 맨 앞 차원 하나를 추가한다.
             
            # 제스처 추론
            _, results, _, _ = knn.findNearest(angle, 3)
            
            idx = int(results[0][0])

            gesture_name = gesture[idx]
            
            cv2.putText(img, text=gesture_name, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255),thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 손의 관절을 프레임에 그린다.

    cv2.imshow('result', img)

    if cv2.waitKey(1) == ord('q'): break