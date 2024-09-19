import cv2
import numpy as np
import math

def distance(x1,y1,x2,y2):
    dist=math.sqrt(math.fabs(x2-x1)**2+math.fabs(y2-y1)**2)
    return dist
def find_color1(frame):
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hsv_lowerbound = np.array([139,149,131])
    hsv_upperbound = np.array([179,255,219])
    mask = cv2.inRange(hsv_frame,hsv_lowerbound,hsv_upperbound)
    res = cv2.bitwise_and(frame,frame,mask=mask)
    cnts,hir = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)>0:
        maxcontour = max(cnts,key=cv2.contourArea)
        M=cv2.moments(maxcontour)
        if M['m00']>0 and cv2.contourArea(maxcontour)>1000:
            cx = int(M['m10']/m['m00'])
            cy = int(M['m01']/m['m00'])
            return (cx,cy),True
        else:
            return(700,700),False
    else:
        return(700,700),False

def find_color2(frame):
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hsv_lowerbound = np.array([101,152,91])
    hsv_upperbound = np.array([149,255,243])
    mask = cv2.inRange(hsv_frame,hsv_lowerbound,hsv_upperbound)
    res = cv2.bitwise_and(frame,frame,mask=mask)
    cnts,hir = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)>0:
        maxcontour = max(cnts,key=cv2.contourArea)
        M=cv2.moments(maxcontour)
        if M['m00']>0 and cv2.contourArea(maxcontour)>2000:
            cx=int(M['m10']/m['m00'])
            cy=int(M['m01']/m['m00'])
            return (cx,cy),True
        else:
            return(700,700),False
    else:
        return(700,700),False
# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to segment out the object of interest
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through the contours and find the largest one
    
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_contour = contour
            max_area = area
    
    # Calculate the angle of the largest contour
    if max_contour is not None:
        # Calculate the moments of the contour
        M = cv2.moments(max_contour)
        
        # Calculate the centroid of the contour
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Calculate the orientation of the contour
        (x, y), (MA, ma), angle = cv2.fitEllipse(max_contour)
        
        # Draw the contour and the angle on the original frame
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
        cv2.line(frame, (cx, cy), (int(x), int(y)), (0, 0, 255), 2)
        cv2.putText(frame, f"Angle: {angle:.2f} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Display the output
    cv2.imshow('Angle Calculator', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
