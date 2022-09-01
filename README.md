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


**8)Morphology**<br>
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

**9)Image Enhancement**<br>
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

**10)GrayScale operation**<br>
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
![image](https://user-images.githubusercontent.com/98145023/178717997-70748699-0d54-4a80-9bde-518747187857.png)<br>
![image](https://user-images.githubusercontent.com/98145023/178706649-bf6d7e8e-6f40-43e8-bdef-51ed60d89ead.png)<br>
![image](https://user-images.githubusercontent.com/98145023/178706776-dae9876a-deb3-4a65-9b57-cffbd6c67db1.png)<br>
![image](https://user-images.githubusercontent.com/98145023/178718134-2969b7d5-3da8-4d46-942a-43003433cb05.png)<br><br>

**11)Slicing without bachground**<br>
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

**12)Slicing without bachground**<br>
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
                z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing w/o bachground')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>    

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/178711993-faa7aee7-9b5a-422e-85e2-c0d38eff53c7.png)<br><br>


**13)Histogram**<br>
**Numpy**<br>
import numpy as np<br>
import cv2 as cv<br>
from matplotlib import pyplot as plt<br>
img = cv.imread('f2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img = cv.imread('f2.jpg',0)<br>
plt.hist(img.ravel(),256,[0,256]);<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/178962658-fd9de21e-6736-4fb8-990d-147dfd67b42a.png)<br><br>

**Skimage**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
img = io.imread('f2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count') <br>
image = io.imread('f2.jpg')<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/178963570-016d9e1e-1296-4879-8c36-739eefdf8624.png)<br><br>

**RGB**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('f2.jpg')<br>

ax= plt.hist(image.ravel(), bins = 256, color = 'orange', )<br>
ax= plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)<br>
_= plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)<br>
ax = plt.xlabel('Intensity Value')<br>
ax = plt.ylabel('Count')<br>
_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/178964462-fabd114f-49c7-49eb-a99a-48535c351819.png)<br><br>

from matplotlib import pyplot as plt<br>
import numpy as np<br>
fig,ax = plt.subplots(1,1)<br>
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])<br>
ax.hist(a, bins = [0,25,50,75,100])<br>
ax.set_title("histogram of result")<br>
ax.set_xticks([0,25,50,75,100])<br>
ax.set_xlabel('marks')<br>
ax.set_ylabel('no. of students')<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/178965952-028834d7-c1d7-4828-8fcc-5ffbabd9e335.png)<br><br>

**14)Program to perform basic image data analysis using intensity transformation**<br>
**a)Image negative**<br>
**b)Log transformation**<br>
**c)Gamma correction**<br>

%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('f2.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/179960486-a9580f5c-5746-4b78-887e-f77edc82b4f7.png)<br><br>

negative=255-pic<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/179960583-7f1afb49-3a88-4fc5-bbd5-675e7b5e64b5.png)<br><br>

%matplotlib inline<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('f2.jpg')<br>
gray=lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>
max_=np.max(gray)<br>
def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name="gray"))<br>
plt.axis("off");<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/179960756-cc1fd8c5-b640-4b65-b9a7-f4e02c963b2c.png)<br><br>


import imageio<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('f2.jpg')<br>
gamma=0.2<br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5,))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/179960876-12d6e967-c95e-4084-8749-44924d925fe6.png)<br><br>

**15)Program to perform basic image manipulation**<br>
**a)Sharpness**<br>
**b)Flipping**<br>
**c)Cropping**<br>

from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
my_image=Image.open('f2.jpg')<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
sharp.save('D:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/179965170-95bddceb-a977-4df0-8842-c91982df0cf2.png)<br><br>


import matplotlib.pyplot as plt<br>
img=Image.open('f2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>
flip.save('D:/image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/179965247-6d8f4256-eb99-459d-9d03-aec14c8aee66.png)<br><br>


from PIL import Image<br>
import matplotlib.pyplot as plt<br>
im=Image.open('f2.jpg')<br>
width,height=im.size<br>
im1=im.crop((280,100,800,600))<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/179965346-fab4d3ce-cddb-4c93-b96b-46787fbbe57d.png)<br><br>

**JUST FOR REFERENCE**<br>
![image](https://user-images.githubusercontent.com/98145023/179965994-afc93db2-e48a-4a42-a9ec-35607f8b3a7b.png)<br><br>


**ASSIGNMENT**<br><br>
**IMAGE MATRIX**<br>
from PIL import Image<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
w, h = 100, 100<br>
data = np. zeros((h, w, 3), dtype=np. uint8)<br>
data[0:50,0:50] = [66, 66, 125]<br>
data[20:60,20:60]=[25,55,100]<br>
data[40:70,40:70]=[56,78,89]<br>
data[60:80,60:80]=[22,52,29]<br>
data[70:80,70:80]=[22,16,13]<br>
img = Image. fromarray(data, 'RGB')<br>
img. save('my.png')<br>
img. show()<br>
plt.imshow(img)<br>
plt.show()<br>
**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/98145023/180202239-6147136c-d0bc-418d-bb03-e6a1adf1f88b.png)<br><br>

**MATRIX 0...1....2**<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (0,0,0)<br>
outerColor = (255, 255, 255)<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        #Find the distance to the center<br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>
        #Make it on a scale from 0 to 1innerColor<br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>
        #Calculate r, g, and b values<br>
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        # print r, g, b<br>
        arr[y, x] = (int(r), int(g), int(b))<br>
plt.imshow(arr, cmap='gray')<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/180202688-83fa697d-d2b5-4abe-948b-a5f5b341e949.png)<br><br>




**#Average**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread("f2.jpg")<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
np.average(img)<br>
![image](https://user-images.githubusercontent.com/98145023/181449982-7fbff3d3-a62d-4521-ba71-1b717754d676.png)<br><br>

**#SD**<br>
from PIL import Image,ImageStat<br>
import matplotlib.pyplot as plt<br>
im=Image.open('f2.jpg')<br>
plt.imshow(im)<br>
plt.show()<br>
stat=ImageStat.Stat(im)<br>
print(stat.stddev)<br>
![image](https://user-images.githubusercontent.com/98145023/181450243-88d02351-e57b-41f3-9959-0e3a680d295b.png)<br><br>

**#Max**<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('f2.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>
print(max_channels)<br>
![image](https://user-images.githubusercontent.com/98145023/181450444-447cbbd6-9a7b-4b1a-ace4-339197637177.png)<br><br>

**#Min**<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('f2.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
min_channels = np.amin([np.min(img[:,:,0]), np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>
print(min_channels)<br><br>
![image](https://user-images.githubusercontent.com/98145023/181450792-4aab9251-74d4-4ac9-928a-16eab4b99f15.png)<br><br>

#Python3 program for printing<br>
#the rectangular pattern<br>
#Function to print the pattern<br>
def printPattern(n):<br>
     arraySize = n * 2 - 1;<br>
    result = [[0 for x in range(arraySize)]<br>
                 for y in range(arraySize)];<br>
 #Fill the values<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            if(abs(i - (arraySize // 2)) ><br>
               abs(j - (arraySize // 2))):<br>
                result[i][j] = abs(i - (arraySize // 2));<br>
            else:<br>
                result[i][j] = abs(j - (arraySize // 2));<br>
  #Print the array<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            print(result[i][j], end = " ");<br>
        print("");<br>
 
#Driver Code<br>
n = 4;<br>
printPattern(n);<br>
![image](https://user-images.githubusercontent.com/98145023/181451251-bd62758b-34cc-4651-af1b-f91278ea4529.png)<br><br>

import matplotlib.pyplot as plt<br>
M =    ([2, 2, 2, 2, 2],  <br>
        [2, 1, 1, 1, 2],  <br>
        [2, 1, 0, 1, 2],  <br>
        [2, 1, 1, 1, 2],<br>
        [2, 2, 2, 2, 2])  <br>
plt.imshow(M,cmap='Blues')<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/98145023/181451544-9ac69639-4a53-4091-8e0b-10416060fe11.png)<br><br>

**EDGE DETECTION**<br>
import cv2<br>
# Read the original image<br>
img = cv2.imread('g1.jpg')<br>
# Display original image<br>
cv2.imshow('Original', img)<br>
cv2.waitKey(0)<br>

# Convert to graycsale<br>
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)<br>
# Blur the image for better edge detection<br>
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)<br>

# Sobel Edge Detection<br>
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis<br>
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis<br>
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection<br>

# Display Sobel Edge Detection Images<br>
cv2.imshow('Sobel X', sobelx)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel Y', sobely)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)<br>
cv2.waitKey(0)<br>

# Canny Edge Detection<br>
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection<br>
# Display Canny Edge Detection Image<br>
cv2.imshow('Canny Edge Detection', edges)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186405908-920ad2c8-7b6d-48d3-82a1-14fad4346e9e.png)<br>
![image](https://user-images.githubusercontent.com/98145023/186406019-4e25faed-8c94-4b3c-9eea-8eb99eb9348b.png)<br>
![image](https://user-images.githubusercontent.com/98145023/186406123-1aa4f620-c3e4-411f-9a66-1871eaa10e56.png)<br>
![image](https://user-images.githubusercontent.com/98145023/186406177-763fd9fd-048d-4802-8f87-f91bf36df179.png)<br>
![image](https://user-images.githubusercontent.com/98145023/186406240-1af601de-eaef-402a-b1cf-bceb253b8b40.png)<br><br>


**USING PILLOW FUNCTIONS**<br><br>

from PIL import Image, ImageChops, ImageFilter <br>
from matplotlib import pyplot as plt<br>
#Create a PIL Image objects<br>
x = Image.open("x.png")<br>
o = Image.open("o.png")<br>
#Find out attributes of Image Objects<br>
print('size of the image: ', x.size, ' colour mode:', x.mode) <br>
print('size of the image: ', o.size, ' colour mode:', o.mode)<br>
#plot 2 images one besides the other <br>
plt.subplot(121), plt.imshow(x)<br>
plt.axis('off')<br>
plt.subplot(122), plt.imshow(o)<br>
plt.axis('off')<br>
#multiply images<br>
merged = ImageChops.multiply(x,o)<br>
#adding 2 images <br>
add = ImageChops.add(x,o)<br>
#convert colour mode <br>
greyscale = merged.convert('L') <br>
greyscale<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186651615-0a50295a-4119-4130-bea7-0285032f63dd.png)<br><br>

image = merged<br>
print('image size: ', <br>image.size,<br>
'\ncolor mode: ', image.mode,<br>
'\nimage width: ', image.width, '| also represented by: ',image.size[0], <br>
'\nimage height: ',image.height, '| also represented by: ',image.size[1],)<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186651961-94684c42-97c7-46f5-926d-2a115f3da769.png)<br><br>


#mapping the pixels of the image so we can use them as coordinates<br>
pixel = greyscale.load()<br>

#a nested Loop to parse through all the pixels in the image<br>
for row in range (greyscale.size[0]): <br>
 for column in range(greyscale.size[1]):<br>
    if pixel[row, column] != (255): <br>
      pixel[row, column] = (0)<br>
greyscale<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186652717-5006c52b-e169-4f9f-ad7f-4695cdc048d9.png)<br><br>

invert = ImageChops.invert(greyscale)<br>

#2.invert by subtraction<br>
bg = Image.new('L', (256, 256), color=(255)) #create a new image with a solid white background <br>
subt = ImageChops.subtract(bg, greyscale) #subtract image from background<br>

#3. rotate<br>
rotate =subt.rotate(45) <br>
rotate<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186652911-220f5d46-134a-4138-8f41-647dddfdab9e.png)<br><br>

#gaussian blur<br>
blur = greyscale.filter(ImageFilter.GaussianBlur (radius=1))<br>

#edge detection<br>
edge = blur.filter(ImageFilter.FIND_EDGES)<br>
edge<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186653038-a567f27e-65b2-40a1-8ae9-c5621669edda.png)<br><br>

#change edge colours<br>
edge = edge.convert('RGB')<br>
bg_red = Image.new('RGB', (256,256), color=(255,0,0))<br>

filled_edge = ImageChops.darker(bg_red, edge)<br>
filled_edge<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186653134-70cab8dd-3fec-4b15-ab42-8f3dc576990b.png)<br><br>

edge.save('processed.png')<br>

**IMAGE RESTORATION**<br>

import numpy as np<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
#Open the image.<br>
img = cv2.imread('dimage_damaged.png')<br>
plt.imshow(img)<br>
plt.show()<br>
#Load the mask.<br>
mask= cv2.imread('dimage_mask.png', 0)<br>
plt.imshow(mask)<br>
plt.show()<br>
#Inpaint.<br>
dst = cv2.inpaint (img, mask, 3, cv2.INPAINT_TELEA)<br>
#write the output.<br>
cv2.imwrite('dimage_inpainted.png', dst)<br>
plt.imshow(dst)<br>
plt.show()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186654017-7590c35e-d2ee-48ac-a9b3-10c7fdba55f1.png)<br><br>

import numpy as np<br>
import pandas as pd<br>
import matplotlib.pyplot as plt<br>
plt.rcParams['figure.figsize'] = (10, 8)<br><br>

def show_image(image, title='<br>Image', cmap_type='gray'):<br>
    plt.imshow(image, cmap=cmap_type)<br>
    plt.title(title)<br>
    plt.axis('off')<br>
def plot_comparison (img_original, img_filtered, img_title_filtered):<br>
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True) <br>
    ax1.imshow(img_original, cmap=plt.cm.gray)<br>
    ax1.set_title('Original')<br>
    ax1.axis('off')<br>
    ax2.imshow(img_filtered, cmap=plt.cm.gray)<br>
    ax2.set_title(img_title_filtered)<br>
    ax2.axis('off')<br><br>
    
from skimage.restoration import inpaint <br>
from skimage.transform import resize <br>
from skimage import color<br><br>

image_with_logo= plt.imread('imlogo.png')<br>

#Initialize the mask <br>
mask= np.zeros(image_with_logo.shape[:-1])<br>

#Set the pixels where the Logo is to 1 <br>
mask [210:272, 360:425] = 1<br>

#Apply inpainting to remove the Logo <br>
image_logo_removed =inpaint.inpaint_biharmonic (image_with_logo,<br>
mask, multichannel=True)<br>

#Show the original and Logo removed images <br>
plot_comparison (image_with_logo, image_logo_removed, 'Image with logo removed')<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186654280-8aedcb37-5d63-4126-a1e8-901ed71f1714.png)<br><br>

from skimage.util import random_noise<br>
fruit_image = plt.imread('fruitts.jpeg')<br>

#Add noise to the image<br>
noisy_image = random_noise (fruit_image)<br>

#Show th original and resulting image<br>
plot_comparison (fruit_image, noisy_image, 'Noisy image')<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186655736-7134954d-5259-47c0-8ad9-7af12d7873cd.png)<br><br>

from skimage.restoration import denoise_tv_chambolle<br>
noisy_image = plt.imread('noisy.jpg')<br>

#Apply total variation filter denoising <br>
denoised_image = denoise_tv_chambolle (noisy_image, multichannel=True)<br>

#Show the noisy and denoised image<br>
plot_comparison (noisy_image, denoised_image, 'Denoised Image')<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186655955-70826cf4-baa4-4b94-b526-51d09dd337f6.png)<br><br>

from skimage.restoration import denoise_bilateral<br>
landscape_image = plt.imread('noisy.jpg')<br>

#Apply bilateral filter denoising<br>
denoised_image = denoise_bilateral (landscape_image, multichannel=True)<br>

#Show original and resulting images<br>
plot_comparison (landscape_image, denoised_image, 'Denoised Image')<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186656159-4259057c-e412-4c23-9c84-8ec9868655e6.png)<br><br>

from skimage.segmentation import slic<br>
from skimage.color import label2rgb<br>
face_image=plt.imread('face.jpg')<br>
#obtain the segmentation with 400 regions <br>
segments = slic(face_image, n_segments=400)<br>
#Put segments on top of original image to compare<br> 
segmented_image = label2rgb(segments, face_image, kind="avg")<br>
#Show the segmented image<br>
plot_comparison (face_image, segmented_image, 'Segmented image, 400 superpixels')<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186656418-093c0470-e408-4ca7-9469-ee6c69ea640e.png)<br><br>

def show_image_contour (image, contours): <br>
    plt.figure()<br>
    for n, contour in enumerate (contours):<br>
        plt.plot(contour[:, 1], contour[:,0], linewidth=3)<br>
        plt.imshow(image, interpolation='nearest', cmap='gray_r') <br>
        plt.title('Contours')<br>
        plt.axis('off')<br>
        
        from skimage import measure, data<br>

#Obtain the horse image <br>
horse_image = data.horse()<br>

#Find the contours with a constant Level value of 0.8 <br>
contours = measure.find_contours (horse_image, level=0.8)<br>

#Shows the image with contours found <br>
show_image_contour (horse_image, contours)<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186656555-f2c9f783-50fc-41ce-bdad-47b350e12fc2.png)<br>

from skimage.io import imread<br>
from skimage.filters import threshold_otsu<br>

image_dices = imread('diceimg.png')<br>
                     
#Make the image grayscale <br>
image_dices = color.rgb2gray(image_dices)<br>

#Obtain the optimal thresh value <br>
thresh = threshold_otsu (image_dices)<br>

#Apply thresholding <br>
binary = image_dices > thresh<br>

#Find contours at a constant value of 0.8 <br>
contours = measure.find_contours(binary, level=0.8)<br>

#Show the image<br>
show_image_contour(image_dices, contours)<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186656619-01cb1456-a3a0-4f18-8cdb-4dfe6933550b.png)<br>

#Create List with the shape of each contour <br>
shape_contours = [cnt.shape[0] for cnt in contours]<br>

#set se as the maximum size of the dots shape<br>
max_dots_shape = 50<br>

#Count dots in contours excluding bigger than dots size<br>
dots_contours = [cnt for cnt in contours if np.shape (cnt) [0] < max_dots_shape]<br>

#Shows all contours found<br>
show_image_contour (binary, contours)<br>

#Print the dice's number <br>
print('Dices dots number: {}'.format(len(dots_contours)))<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/186656680-1543fb22-d51d-4200-8d72-8543915196c6.png)<br><br>

**EDGE DETECTION TECHNIQUES**<br>
**a)Canny Edge Detection**<br>
#Canny Edge detection <br>
import cv2<br>
import numpy as np <br>
import matplotlib.pyplot as plt<br>
plt.style.use('seaborn')<br>
loaded_image = cv2.imread("kenny.jpeg")<br>
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)<br>
gray_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)<br>
edged_image = cv2.Canny(gray_image, threshold1=30, threshold2=100)<br>
plt.figure(figsize=(20,20))<br>
plt.subplot(1,3,1)<br>
plt.imshow(loaded_image, cmap="gray")<br>
plt.title("Original Image")<br>
plt.axis("off")<br>
plt.subplot(1,3,2)<br>
plt.imshow(gray_image, cmap="gray")<br>
plt.axis("off")<br>
plt.title("GrayScale Image")<br>
plt.subplot(1,3,3)<br>
plt.imshow(edged_image,cmap="gray")<br>
plt.axis("off")<br>
plt.title("Canny Edge Detected Image")<br>
plt.show()<br>
**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/98145023/187894531-6c37374a-4fd6-4813-af36-58630f41b7a4.png)<br><br>


b) **Edge detection schemes - the gradient (Sobel - first order derivatives) based edge detector and the Laplacian (2nd order derivative, so it is extremely sensitive to noise) based edge detector.**<br><br>
#Laplacian and Sobel Edge detecting methods<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
# Loading image<br>
#imge = cv2.imread('SanFrancisco.jpg',) <br>
img0 = cv2.imread('kenny.jpeg',)<br>
#converting to gray scale <br>
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)<br>
# remove noise<br>
img = cv2.GaussianBlur (gray, (3,3),0)<br>
#convolute with proper kernels<br>
laplacian= cv2. Laplacian (img,cv2.CV_64F)<br> 
sobelx= cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) # x <br>
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #y<br>
plt.subplot(2,2,1), plt.imshow(img, cmap = 'gray')<br>
plt.title('Original'), plt.xticks([]), plt.yticks([]) <br>
plt.subplot(2,2,2), plt.imshow(laplacian,cmap = 'gray')<br>
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,3), plt.imshow(sobelx, cmap = 'gray') <br>
plt.title('Sobel x'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,4), plt.imshow(sobely,cmap = 'gray')<br>
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/98145023/187895010-6e43b2b0-0e51-4e1c-9aff-b96d890f1887.png)<br><br>

**c)Edge detection using Prewitt operation**<br><br>
#Edge detection using Prewitt operator<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
img = cv2.imread('kenny.jpeg')<br>
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) <br>
img_gaussian = cv2.GaussianBlur (gray, (3,3),0)<br>
#prewitt<br>
kernelx = np.array([[1,1,1], [0,0,0],[-1,-1,-1]])<br>
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])<br>
img_prewittx = cv2.filter2D (img_gaussian, -1, kernelx) <br>
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)<br>
cv2.imshow("Original Image", img)<br>
cv2.imshow("Prewitt X", img_prewittx)<br>
cv2.imshow("Prewitt y", img_prewitty)<br>
cv2.imshow("Prewitt", img_prewittx + img_prewitty)<br>
cv2.waitKey()<br>
cv2.destroyAllWindows()<br>

![image](https://user-images.githubusercontent.com/98145023/187895378-1996ad02-4c53-4bb5-aa4e-51ac74a9c27b.png)<br>
![image](https://user-images.githubusercontent.com/98145023/187895467-c263e2ef-21e2-4f2b-ae58-f083ca21261d.png)<br>
![image](https://user-images.githubusercontent.com/98145023/187895545-fe5827f0-716e-4fca-a82f-c0c7884bdd3a.png)<br>
![image](https://user-images.githubusercontent.com/98145023/187895578-37341c34-8f3b-41ac-86ce-8a87c6ea80d4.png)<br><br>


**d) Roberts Edge Detection- Roberts cross operator**<br>
#Roberts Edge Detection- Roberts cross operator<br>
import cv2<br>
import numpy as np<br>
from scipy import ndimage<br>
from matplotlib import pyplot as plt <br>
roberts_cross_v = np.array([[1, 0 ],
                            [0,-1]])<br>
roberts_cross_h= np.array([[0, 1], 
                           [-1, 0]] )<br>
img = cv2.imread("kenny.jpeg",0).astype('float64')<br>
img/=255.0 <br>
vertical= ndimage.convolve( img, roberts_cross_v) <br>
horizontal = ndimage.convolve( img, roberts_cross_h)<br>
edged_img = np.sqrt( np.square (horizontal) + np.square (vertical))<br>
edged_img*=255<br>
cv2.imwrite("output.jpg",edged_img)<br>
cv2.imshow("OutputImage", edged_img)<br>
cv2.waitKey()<br>
cv2.destroyAllWindows()<br>

![image](https://user-images.githubusercontent.com/98145023/187895837-2df77deb-2de6-45d2-8a00-c8f8a94e1866.png)<br>






