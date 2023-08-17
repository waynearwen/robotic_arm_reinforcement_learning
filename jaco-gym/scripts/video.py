import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videowrite = cv2.VideoWriter("2.mp4",fourcc,30,(640,480))

for filename in ["/home/rl/jaco-gym/images/2/"+str(i+1)+".jpg" for i in range(30)]:
    print(filename)
    aa=cv2.imread(filename)
    videowrite.write(aa)