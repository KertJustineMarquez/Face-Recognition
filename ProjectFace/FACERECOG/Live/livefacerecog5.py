# ABAN, MARQUEZ, QUINTANO, SALAZAR, TECSON, TUAZON,
# CPE 0332-2

#IMPORTED LIBRARIES
import cv2
import numpy as np
import face_recognition
import os

#initialization of path of the folder to a variable
path = r'C:\Users\RID\Desktop\ProjectFace\ABAN_FACERECOG\Live\faceimages2'

#initialization of image list 
images = []

#initialization of image names in the folder
className = []

#initialization of mylist variable to store list directory from path
myList = os.listdir(path)


#loop function to grab image from path which is the image folder 
for cls in myList:
    #variable for reading the path in the folder
    currentImage = cv2.imread(f'{path}/{cls}')
    #appending the images from the folder to the list 'images'
    images.append(currentImage)
    #appending the names of the images from the folder without grabbing the image format
    className.append(os.path.splitext(cls)[0])

# define function for finding the image that will be encoded in the program 
def findEncondings(images):
    #initialization of encodeList list
    encodeList = []
    #for loop function of image in image folder
    for img in images:
        #conversion of image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #initialization of encode variable for face recognition face encoding image
        face_encodings = face_recognition.face_encodings(img)
        #to accept multiple images for better accuracy
        if face_encodings:
            encodeList.extend(face_encodings)
        # encodeList.append(face_encodings)
    return encodeList

# initialization of variable for encoded images
encodeListKnown = findEncondings(images)

print('Encoding Complete')

# initialization of video capture. PS change the number '0' to any number corresponding to what camera you want to use
cap = cv2.VideoCapture(1)

# while loop function to read live video feed and the detect face
while True:

    # Read image 
    success, img = cap.read()
    #Initialization of imgSmall variable to resize image to 1/4 of its original size and convert the color to RGB
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    #Initialization of FaceCurrentFrame variable to recognize face oint locations in the image
    faceCurrentFrame = face_recognition.face_locations(imgSmall)
    #Initialization of encodeCurrentFrame variable to recognize detected faces to be encoded in the program
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall,faceCurrentFrame)

    #for loop function to find matching images
    for encodeFace, faceLoc in zip(encodeCurrentFrame,faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        
        if any(matches):
            matchIndex = np.argmin(faceDis)
        # function to insert corresponding names to correct images and to insert rectangle in the face points
        # if matches[matchIndex]:
            #function to return file name in uppercase
            name = className[matchIndex].upper()
            #function to initialize face location points as y1,x2,y2,x1
            y1,x2,y2,x1 = faceLoc
            #logic function to resize the live image feed back to its original size by multiplying the points by 4
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

            #to display rectangle indictaor in the face locations points and put the name along side it
            cv2.rectangle(img,(x1,y1,),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    #to show the webcam
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)





