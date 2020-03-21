import cv2
import os
import numpy as np
import face_recognition
import pandas as pd


#instance
group=cv2.imread("D:/python/Project/input/Group.jpg")

gray= cv2.cvtColor(group,cv2.COLOR_BGR2GRAY)
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
faces=faceCascade.detectMultiScale(gray, scaleFactor = 1.06, minNeighbors = 3)

face_image=[]
for(x,y,w,h) in faces:
    i=group[y:y+h, x:x+w]
    i = cv2.resize(i, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    face_image.append(i)
    cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)
    
cv2.imwrite("D:/python/Project/output/Detected faces/group.jpg",gray)

frame_num=1
finded=[]
group_encodings=[]
face_no=1

for i in face_image:
    face_locations = face_recognition.face_locations(i,model='cnn')
   
    if len(face_locations)>0:
        print("faces found: ",face_no)
        face_no+=1
        encodings = face_recognition.face_encodings(i,known_face_locations=face_locations,num_jitters=10)
        if len(encodings) > 0:
            cv2.imwrite("D:/python/Project/output/Detected faces/{:3}enc.jpg".format(frame_num),i)
            
            frame_num += 1
            finded.append(i)
            group_encodings.append(encodings[0])
    else:
        encodings = face_recognition.face_encodings(i,num_jitters=20)
        if len(encodings) > 0:
            cv2.imwrite("D:/python/Project/output/Detected faces/crop{:3}.jpg".format(frame_num),i)
            frame_num += 1
            finded.append(i)
            group_encodings.append(encodings[0])



#individual
os.chdir("D:/python/Project/database/")
data = pd.read_excel (r'Before_Attendence.xlsx')    
df = pd.DataFrame(data, columns= ['roll no','name','image','present'])
print(df)

roll=df["roll no"].astype(str)
name=df["name"].astype(str)
link = df["image"].astype(str)

pre=df['present'].astype(int)

os.chdir("D:/python/Project/input/")
#frame_num=1

present=[]

for i in range(len(link)):
    if link[i]!='no':
        test = face_recognition.load_image_file(link[i])
        face_locations = face_recognition.face_locations(test)
        top, right, bottom, left = face_locations[0]
        face_image = test[top:bottom, left:right]
        test_encoding = face_recognition.face_encodings(face_image)[0]
        #cv2.imwrite("D:/python/Project/output/test{:3}.jpg".format(frame_num),test)
        comp=face_recognition.face_distance(group_encodings, test_encoding)
        print(comp)
        minpos = np.argmin(comp)
        print(np.amin(comp)," index =", minpos)
        
        if np.amin(comp)<0.4:
            #cv2.imwrite("D:/python/Project/output/predicted{:3}.jpg".format(frame_num),finded[minpos])
            results = face_recognition.compare_faces([test_encoding], group_encodings[minpos],tolerance=0.50)
            present.append(i)
            if(results): 
                print(name[i],"-",roll[i]," is Present")
                print(results)
                pre[i]=1
            
    else:
        print(name[i],"'s image is not found in the database")
   #frame_num+=1   
        
dict = {'roll no':roll,'name': name, 'image': link, 'present': pre}  
    
df = pd.DataFrame(dict)
df.to_excel("After_Attendence.xlsx") 


for i in range(len(roll)):
    if i in present:
        print(name[i],"-",roll[i]," is Present")
    else:
        print(name[i],"-",roll[i]," is Absent")
        
















    
    

    
