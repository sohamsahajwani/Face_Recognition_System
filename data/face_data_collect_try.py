# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

#import statements
import cv2
import numpy as np
#cv2-> computer vision module

#initialise webcam
cap = cv2.VideoCapture(0)

#Face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
dataset_path = "./data/"


while True:
    ret, frame = cap.read()
    #ret-> bool value, that checks whether frame is captured or not 
    
    if ret==False:
        continue
        
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detecting face
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    #1.3 -> scaling factor(as haarcascade training scale(fixed size) is different from detected face)
    #1.3 -> in each iteration, dimension is reduced by 30%
    #5 -> number of neighbours

    # if len(faces)==0:
        # continue
    #if no face is detected

    # [x,y,w,h] -> x,y are co-ordinates of starting of face box, w,h are width and height of box
    faces = sorted(faces, key = lambda f:f[2]*f[3])
    #sorting to find the max area of face, where f[2]=w(width) and f[3]=h(height),w*h =area

    #pick the last face (because it has largest area)
    for face in faces[-1:]:
        #draw bounding box or the reactangle
        x,y,w,h = face
        cv2.rectangle(gray_frame, (x,y) , (x+w,y+h) , (0,255,255) , 2)

        #Extract (Crop out the required face) : Region of Interest
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        face_data.append(face_section)
        print(len(face_data))
    
    # cv2.imshow("Frame", frame)
    cv2.imshow("gray_frame", gray_frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    #key_pressed-> if person presses 'q', it can break . Wait key is in milli second

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

file_name = input("Enter the name of the person: ")
url = dataset_path+file_name+'.npy'

# Save this data into file system
np.save(url,face_data)
print("Data Successfully saved!! :")
    
cap.release()
cv2.destroyAllWindows()