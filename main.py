import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture("demovideo/1.mp4")

mpdraw=mp.solutions.drawing_utils
mpfacemesh=mp.solutions.face_mesh
facemesh=mpfacemesh.FaceMesh()
draw_spec=mpdraw.DrawingSpec(thickness=1,circle_radius=2)
ptime=0
while True:
    success , imgs=cap.read()
    

    if not success:
        cv2.destroyAllWindows()
        break
    imgRGB=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    result=facemesh.process(imgRGB)
    if result.multi_face_landmarks:
        for facelms in result.multi_face_landmarks:
            mpdraw.draw_landmarks(imgs,facelms,mpfacemesh.FACEMESH_TESSELATION,draw_spec,draw_spec)
            for id ,lm in enumerate(facelms.landmark):
                h,w=imgs.shape[:2]
                x,y=int(lm.x*h),int(lm.y*w)
                print(id,x,y)

    #scale 
    scale=0.5
    h,w=imgs.shape[:2]
    height=int(imgs.shape[1]*scale)
    width=int(imgs.shape[0]*scale)
    img=cv2.resize(imgs,(height,width))
    
    #fps
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    
    #process
    
    #put Text
    cv2.putText(img,str(f"fps:{int(fps)}"),(20,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    cv2.imshow("img",img)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        cv2.destroyAllWindows()
        break