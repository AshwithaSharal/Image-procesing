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


**LAB EXERCISE**<br><br>
**1)Develop a program to readimage using URL**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://images.unsplash.com/photo-1522069169874-c58ec4b76be5?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8MXx8fGVufDB8fHx8&w=1000&q=80'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/175004843-72e5a221-7734-4aa4-a59d-9b1d02a7fa46.png)<br><br><br>


**2)Develop a program to readiamge using URL**<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('fish1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/175014696-c3fca2e7-a812-468f-8ab4-642b8c4c1970.png)<br><br>

hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/175015016-5f81e112-16db-4449-8c82-13eff269e112.png)<br><br>

light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/175015187-a606f1bb-4e38-4242-988b-be444186301b.png)<br><br>

blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/175015383-58bc7480-ab13-48a3-9369-62d9428b0218.png)<br><br>


**3)Write a program to perform arithmatic operations on images**<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>

img1=cv2.imread('ff1.jpg')<br>
img2=cv2.imread('ff2.jpg')<br>

fimg1 = img1 + img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>

cv2.imwrite('output.jpeg',fimg1)<br>
fimg2 = img1 -img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>

cv2.imwrite('output.jpeg',fimg2)<br>
fimg3 = img1 * img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>

cv2.imwrite('output.jpeg',fimg3)<br>
fimg4 = img1 / img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
cv2.imwrite('output.jpeg',fimg4)<br><br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/175022261-2f9d4754-031b-4ef1-84a6-43f244a14e0a.png)<br>
![image](https://user-images.githubusercontent.com/98145023/175022304-5fa9b1b9-6a58-457f-a11d-f55492599d81.png)<br>
![image](https://user-images.githubusercontent.com/98145023/175022398-874dedcd-89e9-4ad7-b004-46bd8a77eb97.png)<br>


**4)Develop the program to change the image to different color spaces.**<br>
import cv2<br>
img=cv2.imread('E:\\f1.jpg')<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br><br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/175274944-ad3a26dd-ce05-4c03-bae9-8c625e9204f6.png)<br>
![image](https://user-images.githubusercontent.com/98145023/175274041-a761b56e-7665-4985-bead-eb5720737818.png)<br>
![image](https://user-images.githubusercontent.com/98145023/175274155-438f43cf-dc00-4585-bb8b-fcb838efd085.png)<br>
![image](https://user-images.githubusercontent.com/98145023/175274271-22cd6c5c-4be6-4841-9313-ee5d52d65ae3.png)<br><br>


**5)Program to create an image using 2D array**<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[25,13,0]<br>
array[:,100:]=[120,0,255]<br>
img=Image.fromarray(array)<br>
img.save('image1.png')<br>
img.show()<br>
cv2.waitKey(0)<br><br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/175280876-1b5dbfcd-ff19-4f36-a169-af7206f20688.png)<br><br>

**6)Program on images using binary operation**<br>
**With different images**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
img1=cv2.imread('f2n.jpg')<br>
img2=cv2.imread('f3n.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(img1,img2)<br>
bitwiseOr=cv2.bitwise_or(img1,img2)<br>
bitwiseXor=cv2.bitwise_xor(img1,img2)<br>
bitwiseNot_img1=cv2.bitwise_not(img1)<br>
bitwiseNot_img2=cv2.bitwise_not(img2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br><br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/176405160-ab95d63e-c1f9-4381-9245-0f11b3c9295e.png)<br><br>

**With same image**<br>
import cv2import cv2<br>
import matplotlib.pyplot as plt<br>
img1=cv2.imread('f2n.jpg',1)<br>
img2=cv2.imread('f2n.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(img1,img2)<br>
bitwiseOr=cv2.bitwise_or(img1,img2)<br>
bitwiseXor=cv2.bitwise_xor(img1,img2)<br>
bitwiseNot_img1=cv2.bitwise_not(img1)<br>
bitwiseNot_img2=cv2.bitwise_not(img2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/176405805-1ba3e267-36b8-42fe-80c2-b61372cdaa4f.png)<br><br>

**7)Blurring image**<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('b1.jpg')<br>
cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>
median=cv2.medianBlur(image,39)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/176421466-d17d9dd1-d4de-4dce-8fd6-e08c0328364b.png)<br>
![image](https://user-images.githubusercontent.com/98145023/176421597-72d3059f-2eb1-430d-8f75-2402c7508dd0.png)<br>
![image](https://user-images.githubusercontent.com/98145023/176421644-029a171f-e0a4-49ec-aeab-e0743f66cc17.png)<br>
![image](https://user-images.githubusercontent.com/98145023/176421703-c7d66080-1e9e-423b-97d9-9993124bd5d4.png)<br><br>


**Morphology**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('f1.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br><br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/176425914-12e7f603-cab3-4b2c-9f48-04935f688413.png)<br><br>

**Image Enhancement**<br>
  from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('f1.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br><br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/178703124-6cedcc73-cca9-4d6c-999a-1c8527cad944.png)
![image](https://user-images.githubusercontent.com/98145023/178703397-189e823e-1e36-4e6c-acfc-09e5a8fd8ca2.png)
![image](https://user-images.githubusercontent.com/98145023/178703533-e8fae6e7-4096-4a55-9ccc-23c0b09263ba.png)
![image](https://user-images.githubusercontent.com/98145023/178703690-76007080-5d41-4ab8-befd-f0e4e3055d62.png)
![image](https://user-images.githubusercontent.com/98145023/178703828-82d4afe5-8dba-495a-bd8f-b84f606ddfcf.png)<br><br>

**GrayScale operation**<br>
import cv2<br>
OriginalImg=cv2.imread('nnnn1.jpg')<br>
GrayImg=cv2.imread('nnnn1.jpg',0)<br>
isSaved=cv2.imwrite('‪‪E:/i.jpg',GrayImg)<br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('The image is successfully saved.')<br><br>
    
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/178706649-bf6d7e8e-6f40-43e8-bdef-51ed60d89ead.png)
![image](https://user-images.githubusercontent.com/98145023/178706776-dae9876a-deb3-4a65-9b57-cffbd6c67db1.png)<br><br>

**Slicing without bachground**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('nnnn1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if (image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
                z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing w/o bachground')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/178710159-7edfcb70-3fa0-4dd2-977e-e35420583c02.png)<br><br>

    





