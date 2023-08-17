import cv2
import numpy as np 

# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# ori = cv2.imread("/home/rl/jaco-gym/gazebo_image.jpg")
# # gray = cv2.cvtColor(ori,cv2.COLOR_BGR2GRAY)
# image=ori.copy()
# # image[:,:]=[0,255,0]
# image[:,:,:]=255
# # gray1=cv2.erode(gray,kernel)
# # gray1=cv2.dilate(gray1,kernel)
# # ori=cv2.erode(ori,kernel)
# # ori=cv2.dilate(ori,kernel)
# # print(gray[220,300])
# # while(True):
# #     cv2.imshow("bb",gray)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # arm_pos=np.where(gray>200 and gray<250)
# arm_pos=np.where(ori[:,:,1]<112)
# # background=np.where(gray>150)
# obj_pos=np.where((ori[:,:,1].astype(int)-ori[:,:,0].astype(int))>30)
# # print(abs(-4))
# # print(obj_pos)
# # image[background_pos[0],background_pos[1],:]=255
# # print(ori[328,279])
# # print(abs(ori[328,279,1].astype(int)-ori[328,279,0].astype(int))>30)
# image[arm_pos[0],arm_pos[1],:]=0
# image[obj_pos[0],obj_pos[1],:]=[0,255,0]
# # image[0:5,0:50,:]=[255,0,0]
# # image[background[0],background[1],:]=255
# # image=cv2.erode(image,kernel)
# # image=cv2.dilate(image,kernel)

# # image[arm_pos[0],arm_pos[1],:]=[0,0,0]



# while(True):
#     cv2.imshow("aa",ori)
#     # cv2.imshow("cc",gray)
#     # cv2.imshow("dd",gray1)
#     cv2.imshow("bb",image)
#     # cv2.imshow("cc",gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

for i in range(1,13):
    status=cv2.imread("/home/rl/aa/aa"+str(i)+".jpg")
    print(status)
    image=status.copy()
    image[:,:,:]=255
    arm_pos=np.where(status[:,:,1]<140)
    obj_pos=np.where((status[:,:,1].astype(int)-status[:,:,0].astype(int))>30)
    image[arm_pos[0],arm_pos[1],:]=0
    image[obj_pos[0],obj_pos[1],:]=[0,255,0]
    cv2.imwrite(str(i)+".jpg",image)