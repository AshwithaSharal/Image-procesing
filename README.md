# Image-procesing
**1) Develop a program to display grayscale image usinng read & write operation**<br>

import cv2<br>
img=cv2.imread('f1.jpg',0)<br>
cv2.imshow('fl',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/98145023/173810741-3ee02157-785f-423a-bbeb-4f994d7bbaf9.png)<br>

**2) Develop a program to display the image using matplotlib**<br>

import cv2<br>
import matplotlib.pyplot as plt<br>
image=cv2.imread('p1.jpg')<br>
plt.imshow(image)<br>
plt.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/98145023/173811644-3f36c483-6b49-4dfc-9ae9-76984bbe5848.png)

**3)Develop a program to perform linear transformation-rotation.**<br>

from PIL import Image<br>
Original_Image=Image.open('p1.jpg')<br>
rotate_image1=Original_Image.rotate(180)<br>
rotate_image1.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/98145023/173813582-9ea72447-8b38-4134-8b66-6860c6cbd754.png)<br>
