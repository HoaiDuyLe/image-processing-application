#CAU1
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

def noise_gaussian(img, variance = 100):
    gaussian_noise = img.copy()
    cv2.randn(gaussian_noise, 0, variance**0.5)
    return cv2.add(img,gaussian_noise)

def noise_uniform(img, low=0, high=100):
    uniform_noise = img.copy()
    cv2.randu(uniform_noise,low,high)
    return cv2.add(img,uniform_noise)

def noise_salt_pepper(img, noise_type, prob=0.1):
    if noise_type == "salt":
        salt_noise = img.copy()
        for i in range (int(img.shape[0]*img.shape[1]*prob)):
            y_coord=random.randint(0, img.shape[0] - 1)
            x_coord=random.randint(0, img.shape[1] - 1)
            salt_noise[y_coord][x_coord] = 255
        return salt_noise
    elif noise_type == "pepper":
        pepper_noise = img.copy()
        for i in range (int(img.shape[0]*img.shape[1]*prob)):
            y_coord=random.randint(0, img.shape[0] - 1)
            x_coord=random.randint(0, img.shape[1] - 1)
            pepper_noise[y_coord][x_coord] = 0
        return pepper_noise

# if __name__ == "__main__":
image = cv2.imread('./tb1.png',cv2.IMREAD_GRAYSCALE)
gaussian1 = noise_gaussian(image,200)
gaussian2 = noise_gaussian(image,500)
uniform1 = noise_uniform(image,10,50)
uniform2 = noise_uniform(image,20,100)
salt1 = noise_salt_pepper(image,"salt",0.1)
salt2 = noise_salt_pepper(image,"salt",0.3)
pepper1 = noise_salt_pepper(image,"pepper",0.1)
pepper2 = noise_salt_pepper(image,"pepper",0.3)

fig, axes = plt.subplots(ncols=2, nrows=2,figsize=(20, 20))
ax0, ax1, ax2, ax3 = axes.flat
ax0.imshow(gaussian1, cmap = 'gray')
ax0.set_title('Image corrupted by AWGN with variance = 1000', fontsize=24)
ax0.axis('off')

ax1.imshow(gaussian2, cmap = 'gray')
ax1.set_title('Image corrupted by AWGN with variance = 2000', fontsize=24)
ax1.axis('off')

ax2.imshow(uniform1, cmap = 'gray')
ax2.set_title('Image corrupted by uniform noise with \n low_intensity = 20, high_intensity = 100', fontsize=24)
ax2.axis('off')

ax3.imshow(uniform2, cmap = 'gray')
ax3.set_title('Image corrupted by uniform noise with \n low_intensity = 50, high_intensity = 200', fontsize=24)
ax3.axis('off')
fig.savefig("D:/result_HW7_1.png",bbox_inches='tight')

fig, axes = plt.subplots(ncols=2, nrows=2,figsize=(20, 20))
ax0, ax1, ax2, ax3 = axes.flat
ax0.imshow(salt1, cmap = 'gray')
ax0.set_title('Image corrupted by salt noise with prob. = 0.1', fontsize=24)
ax0.axis('off')

ax1.imshow(salt2, cmap = 'gray')
ax1.set_title('Image corrupted by salt noise with prob. = 0.3', fontsize=24)
ax1.axis('off')

ax2.imshow(pepper1, cmap = 'gray')
ax2.set_title('Image corrupted by pepper noise with prob. = 0.1', fontsize=24)
ax2.axis('off')

ax3.imshow(pepper2, cmap = 'gray')
ax3.set_title('Image corrupted by pepper noise with prob. = 0.3', fontsize=24)
ax3.axis('off')
fig.savefig("D:/result_HW7_2.png",bbox_inches='tight')

#Geometric mean filter:
def G_mean(img,kernel_size): 
    G_mean_img = np.ones(img.shape)
    k = int((kernel_size-1)/2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i <k or i>(img.shape[0]-k-1) or j <k or j>(img.shape[1]-k-1):
                G_mean_img[i][j]=img[i][j]
            else:
                for n in range(kernel_size):
                    for m in range(kernel_size):
                        G_mean_img[i][j] *=np.float(img[i-k+n][j-k+m])
                G_mean_img[i][j] = pow(G_mean_img[i][j],1/(kernel_size*kernel_size))
    G_mean_img = np.uint8(G_mean_img)
    return G_mean_img

#Contraharmonic mean filter filter:
def HT_mean(img,kernel_size,Q):
    G_mean_img = np.zeros(img.shape)
    k = int((kernel_size-1)/2) 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i <k or i>(img.shape[0]-k-1) or j <k or j>(img.shape[1]-k-1):
                G_mean_img[i][j]=img[i][j]
            else:
                result_top = 0
                result_down = 0
                for n in range(kernel_size):
                    for m in range(kernel_size):
                        if Q>0:
                            result_top +=pow(np.float(img[i-k+n][j-k+m]),Q+1)
                            result_down +=pow(np.float(img[i-k+n][j-k+m]),Q)
                        else:
                            if img[i-k+n][j-k+m]==0:
                                G_mean_img[i][j] = 0
                                break
                            else:
                                result_top +=pow(np.float(img[i-k+n][j-k+m]),Q+1)
                                result_down +=pow(np.float(img[i-k+n][j-k+m]),Q)
                    else:
                        continue
                    break 
                else:
                    if result_down !=0:
                        G_mean_img[i][j] = result_top/result_down
    G_mean_img = np.uint8(G_mean_img)
    return G_mean_img

G_mean_img_3 = G_mean(gaussian1,kernel_size = 3)
fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(20, 10))
ax0, ax1 = axes.flat
ax0.imshow(gaussian1, cmap = 'gray')
ax0.set_title('Image corrupted by AWGN with variance = 200', fontsize=24)
ax0.axis('off')

ax1.imshow(G_mean_img_3, cmap = 'gray')
ax1.set_title('Image restored by 3x3 geometric \n mean filter', fontsize=24)
ax1.axis('off')
fig.savefig("D:/result_HW7_3.png",bbox_inches='tight')

salt_restore = HT_mean(salt1,kernel_size = 3, Q = -1.5)
fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(20, 10))
ax0, ax1 = axes.flat
ax0.imshow(salt1, cmap = 'gray')
ax0.set_title('Image corrupted by salt noise with prob. = 0.1', fontsize=24)
ax0.axis('off')

ax1.imshow(salt_restore, cmap = 'gray')
ax1.set_title('Image restored by 3x3 contraharmonic \n mean filter with Q = -1.5', fontsize=24)
ax1.axis('off')
fig.savefig("D:/result_HW7_4.png",bbox_inches='tight')

pepper_restore = HT_mean(pepper1,kernel_size = 3, Q = 1.5)
fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(20, 10))
ax0, ax1 = axes.flat
ax0.imshow(pepper1, cmap = 'gray')
ax0.set_title('Image corrupted by pepper noise with prob. = 0.1', fontsize=24)
ax0.axis('off')

ax1.imshow(pepper_restore, cmap = 'gray')
ax1.set_title('Image restored by 3x3 contraharmonic \n mean filter with Q = 1.5', fontsize=24)
ax1.axis('off')
fig.savefig("D:/result_HW7_5.png",bbox_inches='tight')

#CAU2
import os
import cv2
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
import math
def fft_distances(m, n):
    u = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v = np.array([i if i <= n / 2 else n - i for i in range(n)],
                 dtype=np.float32)
    u.shape = m, 1
    ret = u * u + v * v
    return np.fft.fftshift(ret)
def turbulance_model(img,k = 0.0025):    
    #compute fft and fft_shift
    img_fft = np.fft.fft2(img, axes=(0,1))
    img_fftshift = np.fft.fftshift(img_fft)
    
    #compute mask filter
    duv = fft_distances(*img_fftshift.shape[:2])
    filter_mat =np.exp(-k * (duv**(5/6)))
    
    #apply filter by multiple filter_mat and fftshift
    dst_fftshift = np.multiply(img_fftshift,filter_mat)
    
    #compute inverse fft
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    
    #combine complex real and imaginary components to form (the magnitude for) the original image again
    dst = np.abs(dst_ifft)
    turbulance_img = cv2.normalize(dst, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return turbulance_img

def motion_process(image_size, motion_angle):
    psf=np.zeros(image_size)
    center_position=(image_size [0] -1)/2
    slope_tan=math.tan (motion_angle * math.pi/180)
    slope_cot=1/slope_tan
    if slope_tan<= 1:
        for i in range (15):
            offset=round (i * slope_tan) #((center_position-i) * slope_tan)
            psf [int(center_position + offset), int (center_position-offset)]=1
        return psf/psf.sum () #normalize the brightness of the point spread function
    else:
        for i in range (15):
            offset=round (i * slope_cot)
            psf [int(center_position-offset), int(center_position + offset)]=1
    return psf/psf.sum ()
#Motion blur the picture
def make_blurred(input, psf, eps):
    input_fft=fft.fft2 (input) #Fourier transform a two-dimensional array
    psf_fft=fft.fft2(psf) + eps
    blurred=fft.ifft2(input_fft * psf_fft)
    blurred=np.abs(fft.fftshift (blurred))
    return blurred

img = cv2.imread('./tb3.png',cv2.IMREAD_GRAYSCALE)
turbulance_img1 = turbulance_model(img, k=0.0025)
turbulance_img2 = turbulance_model(img, k=0.001)
turbulance_img3 = turbulance_model(img, k=0.00025)
fig, axes = plt.subplots(ncols=2, nrows=2,figsize=(20, 20))
ax0, ax1, ax2, ax3 = axes.flat
ax0.imshow(img, cmap = 'gray')
ax0.set_title('Original image', fontsize=24)
ax0.axis('off')

ax1.imshow(turbulance_img1, cmap = 'gray')
ax1.set_title('Severe turbulence k = 0.0025', fontsize=24)
ax1.axis('off')

ax2.imshow(turbulance_img2, cmap = 'gray')
ax2.set_title('Mild turbulence k = 0.001', fontsize=24)
ax2.axis('off')

ax3.imshow(turbulance_img3, cmap = 'gray')
ax3.set_title('Low turbulence k = 0.00025', fontsize=24)
ax3.axis('off')
fig.savefig("D:/result_HW7_6.png",bbox_inches='tight')

image=cv2.imread ("./tb4.png",0)
img_h=image.shape [0]
img_w=image.shape [1]

#Perform motion blur
psf=motion_process((img_h, img_w), 300)
blurred=np.abs(make_blurred (image, psf, 1e-2))

fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(20, 10))
ax0, ax1 = axes.flat
ax0.imshow(image, cmap = 'gray')
ax0.set_title('Original image', fontsize=24)
ax0.axis('off')

ax1.imshow(blurred, cmap = 'gray')
ax1.set_title('Motion blurred image', fontsize=24)
ax1.axis('off')
fig.savefig("D:/result_HW7_7.png",bbox_inches='tight')

#CAU3
import numpy as np
from numpy import fft
import math
import cv2
#Simulation motion blur
def motion_process(image_size, motion_angle):
    psf=np.zeros(image_size)
    center_position=(image_size [0] -1)/2
    slope_tan=math.tan (motion_angle * math.pi/180)
    slope_cot=1/slope_tan
    if slope_tan<= 1:
        for i in range (15):
            offset=round (i * slope_tan) #((center_position-i) * slope_tan)
            psf [int(center_position + offset), int (center_position-offset)]=1
        return psf/psf.sum () #normalize the brightness of the point spread function
    else:
        for i in range (15):
            offset=round (i * slope_cot)
            psf [int(center_position-offset), int(center_position + offset)]=1
    return psf/psf.sum ()
#Motion blur the picture
def make_blurred(input, psf, eps):
    input_fft=fft.fft2 (input) #Fourier transform a two-dimensional array
    psf_fft=fft.fft2(psf) + eps
    blurred=fft.ifft2(input_fft * psf_fft)
    blurred=np.abs(fft.fftshift (blurred))
    return blurred
#Wiener filtering
def wiener(input, psf, eps, k=0.01):
    input_fft=fft.fft2(input)
    psf_fft=fft.fft2(psf) + eps
    psf_fft_1=np.conj(psf_fft) /(np.abs(psf_fft)**2 + k)
    result=fft.ifft2(input_fft * psf_fft_1)
    result=np.abs(fft.fftshift (result))
    return result 

image=cv2.imread ("./tb4.png",0)
img_h=image.shape [0]
img_w=image.shape [1]
#Perform motion blur
psf=motion_process((img_h, img_w), 300)
blurred=np.abs(make_blurred (image, psf, 1e-3))
#Wiener filtering
result=wiener(blurred, psf, 1e-2)

fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(20, 10))
ax0, ax1 = axes.flat
ax0.imshow(blurred, cmap = 'gray')
ax0.set_title('Motion blurred image', fontsize=24)
ax0.axis('off')
ax1.imshow(result, cmap = 'gray')
ax1.set_title('Wiener filtered image with K = 0.01', fontsize=24)
ax1.axis('off')
fig.savefig("D:/result_HW7_8.png",bbox_inches='tight')