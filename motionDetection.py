import numpy as np
import cv2


cap = cv2.VideoCapture('3.mov')



# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
blur = cv2.GaussianBlur(old_gray, (5, 5), 0)
ret3, old_th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#fgbg = cv2.createBackgroundSubtractorKNN();

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #fgmask = fgbg.apply(frame)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)


    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #img = cv2.add(frame,mask)
    #img=frame_gray-old_gray
    img1=th3-old_th3
    img2=old_th3 - th3
    img=cv2.add(img1,img2)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    old_th3=th3.copy()

    #p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()