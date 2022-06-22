# Image-procesing
**1) Develop a program to display grayscale image usinng read & write operation**<br>
import cv2<br>
img=cv2.imread('f1.jpg',0)<br>
cv2.imshow('fl',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/98145023/173810741-3ee02157-785f-423a-bbeb-4f994d7bbaf9.png)<br><br>

**2) Develop a program to display the image using matplotlib**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image=cv2.imread('p1.jpg')<br>
plt.imshow(image)<br>
plt.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/98145023/173811644-3f36c483-6b49-4dfc-9ae9-76984bbe5848.png)<br><br>

**3)Develop a program to perform linear transformation-rotation.**<br>
from PIL import Image<br>
Original_Image=Image.open('p1.jpg')<br>
rotate_image1=Original_Image.rotate(180)<br>
rotate_image1.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/98145023/173813582-9ea72447-8b38-4134-8b66-6860c6cbd754.png)<br><br>

**4)Develop a program to convert color string to RGB color values**<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("brown")<br>
print(img2)<br>

**OUTPUT:**<br>
(255, 255, 0)<br>
(165, 42, 42)<br><br>

**5)Write a program to create image using color**<br>
from PIL import Image<br>
img=Image.new('RGB',(200,40),(255,196,0))<br>
img.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/98145023/173815051-8a0d2d36-f78a-46ae-a854-3b391466c711.png)<br><br>

**6)Develop a program to visualize the image using varoius color spaces**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('b1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/98145023/173815502-14d7f5a5-4369-4414-9b04-5baa5a0b37a6.png)<br><br>

**7)Display the image attributes**<br>
from PIL import Image<br>
image=Image.open('b1.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>

**OUTPUT:**<br>
Filename: b1.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
Size: (1080, 1920)<br>
Width: 1080<br>
Height: 1920<br>
import cv2<br><br>

**8)Convert the original image to gray scale and then to binary**<br>
import cv2<br>
img=cv2.imread('p2.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br><br>
img=cv2.imread('p2.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br><br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/98145023/174037732-9d5a42af-9a0b-4b65-8487-47d263e3543f.png)<br>
![image](https://user-images.githubusercontent.com/98145023/174037840-09af38de-119b-4730-bf6d-f7e044309251.png)<br>
![image](https://user-images.githubusercontent.com/98145023/174037923-e35b050f-43d0-4f74-928c-bd87f74b5e7d.png)<br><br>


**9)Resize the original image**<br>
import cv2<br>
img=cv2.imread('l5.jpg')<br>
print('Original image length width',img.shape)<br>
cv2.imshow('Origianl image',img)<br>
cv2.waitKey(0)<br>

imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized imzge',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>

**OUTPUT:**<br>
original image length width (1500, 1000, 3)<br>
Resized image length width (160, 150, 3)<br>

![image](https://user-images.githubusercontent.com/98145023/174048116-7816da97-de2b-4364-aacb-e9208910f03c.png)<br>
![image](https://user-images.githubusercontent.com/98145023/174048256-91e805b8-6a9b-4e89-9f3c-c96a97948a11.png)<br><br>


**LAB EXERCISE**<br>
**10)Develop a program to readimage using URL**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://images.unsplash.com/photo-1522069169874-c58ec4b76be5?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8MXx8fGVufDB8fHx8&w=1000&q=80'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/175004843-72e5a221-7734-4aa4-a59d-9b1d02a7fa46.png)<br><br>

