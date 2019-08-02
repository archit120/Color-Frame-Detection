import sys
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    cv2.imshow("input", img)
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
     
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 16  , 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,4,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))  
    #print(center.shape)
    for i in range(k):
       # print(np.all(res2==center[i], 2))
        a =  np.uint8(np.all(res2==center[i], 2)*255)
        contours, hierarchy = cv2.findContours(a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt2 = []
        for cnt in contours: 
            epsilon = 0.1*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            # Contours detection
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.012*cv2.arcLength(cnt, True), True) # 0.012 param
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            

            if area > 300:#param
                if len(approx) == 4:
                    (cx,cy),(MA,ma),angle = cv2.fitEllipse(cnt)
                    ar = MA/ma
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area)/hull_area

                    if solidity > 0.9 and ar < 0.4:
                        cnt2.append(approx)

        cv2.drawContours(res2, cnt2, -1, (0,255,0), 3)
    img = cv2.resize(res2, (img.shape[1]*2, img.shape[0]*2))

    cv2.imshow("input2", img)

    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
