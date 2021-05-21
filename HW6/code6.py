# CAU 1
import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def fft_distances(m, n):
    #calculate the distance of each point of the m, n matrix from the center
    u = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v = np.array([i if i <= n / 2 else n - i for i in range(n)],
                 dtype=np.float32)
    u.shape = m, 1
    
    # The distance from each point to the upper left corner of the matrix
    ret = np.sqrt(u * u + v * v)
    
    # The distance of each point from the center of the matrix
    return np.fft.fftshift(ret)
def homomorphic_filter(img, rl=0.4, rh=2, c=5, d0=20):
    # compute ln(img)
    img = np.log1p(np.float64(img), dtype=np.float64)
    
    #compute fft and fft_shift
    img_fft = np.fft.fft2(img, axes=(0,1))
    img_fftshift = np.fft.fftshift(img_fft)
    
    #compute mask filter
    duv = fft_distances(*img_fftshift.shape[:2])
    filter_mat = (rh - rl) * (1 - np.exp(-c * (duv * duv) / (d0 * d0))) + rl
    
    #apply filter by multiple filter_mat and fftshift
    dst_fftshift = np.multiply(img_fftshift,filter_mat)
    
    #compute inverse fft
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    
    #combine complex real and imaginary components to form (the magnitude for) the original image again
    dst = np.real(dst_ifft)
    
    #compute exponential 
    dst = np.exp(dst, dtype=np.float64)
    img_homomorphic = cv2.normalize(dst, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_homomorphic

if __name__ == "__main__":
     path = "./CTI.png"
     img = np.array(Image.open(path).convert('L'))
     img_new = homomorphic_filter(img, rl=0.25, rh=1.5, c=5, d0=20)

     fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(12, 10))
     axes[0].imshow(img, cmap = 'gray')
     axes[0].set_title('Input Image', fontsize=24)
     axes[0].axis('off')

     axes[1].imshow(img_new, cmap = 'gray')
     axes[1].set_title('Homomorphic filtered image', fontsize=24)
     axes[1].axis('off')
     fig.savefig("D:/result_HW6_1.png",bbox_inches='tight')

#CAU2
import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math

def fft_distances(m, n):
    #calculate the distance of each point of the m, n matrix from the center
    u = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v = np.array([i if i <= n / 2 else n - i for i in range(n)],
                 dtype=np.float32)
    u.shape = m, 1
    
    # The distance from each point to the upper left corner of the matrix
    ret = np.sqrt(u * u + v * v)
    
    # The distance of each point from the center of the matrix
    return np.fft.fftshift(ret)

def laplacian_filter(img):    
    #compute fft and fft_shift
    img_fft = np.fft.fft2(img, axes=(0,1))
    img_fftshift = np.fft.fftshift(img_fft)
    
    #compute mask filter
    duv = fft_distances(*img_fftshift.shape[:2])
    filter_mat = 4*(math.pi*math.pi)*(duv * duv)
    
    #apply filter by multiple filter_mat and fftshift
    dst_fftshift = np.multiply(img_fftshift,filter_mat)
    
    #compute inverse fft
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    
    #combine complex real and imaginary components to form (the magnitude for) the original image again
    dst = np.abs(dst_ifft)
     
    delta = cv2.normalize(dst, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    laplacian = img - delta
    return laplacian

def unsharp_filter(img,k):    
    #compute fft and fft_shift
    img_fft = np.fft.fft2(img, axes=(0,1))
    img_fftshift = np.fft.fftshift(img_fft)
    
    #compute mask filter
    duv = fft_distances(*img_fftshift.shape[:2])
    filter_mat = 1 + k*(1 - np.exp(-(duv * duv) / (2 * d0 * d0)))
    
    #apply filter by multiple filter_mat and fftshift
    dst_fftshift = np.multiply(img_fftshift,filter_mat)
    
    #compute inverse fft
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    
    #combine complex real and imaginary components to form (the magnitude for) the original image again
    dst = np.abs(dst_ifft)
     
    img_unsharp = cv2.normalize(dst, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_unsharp

if __name__ == "__main__":
     path = "./moon.jpg"     
     img = np.array(Image.open(path).convert('L'))
     #Laplacian
     img_new_laplacian = laplacian_filter(img)
     #unsharp
     img_new_unsharp = unsharp_filter(img,1)
     #high-boot
     img_new_highboot = unsharp_filter(img,3)

     fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(20, 10))
     axes[0].imshow(img, cmap = 'gray')
     axes[0].set_title('Input Image', fontsize=24)
     axes[0].axis('off')
     axes[1].imshow(img_new_laplacian, cmap = 'gray')
     axes[1].set_title('Laplacian filtered image', fontsize=24)
     axes[1].axis('off')
     fig.savefig("D:/result_HW6_4.png",bbox_inches='tight')     

     fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(20, 10))
     axes[0].imshow(img, cmap = 'gray')
     axes[0].set_title('Input Image', fontsize=24)
     axes[0].axis('off')
     axes[1].imshow(img_new_unsharp, cmap = 'gray')
     axes[1].set_title('Unsharp filtered image', fontsize=24)
     axes[1].axis('off')
     fig.savefig("D:/result_HW6_5.png",bbox_inches='tight')     

     fig, axes = plt.subplots(ncols=2, nrows=1,figsize=(20, 10))
     axes[0].imshow(img, cmap = 'gray')
     axes[0].set_title('Input Image', fontsize=24)
     axes[0].axis('off')
     axes[1].imshow(img_new_highboot, cmap = 'gray')
     axes[1].set_title('High-boot filtered image', fontsize=24)
     axes[1].axis('off')
     fig.savefig("D:/result_HW6_6.png",bbox_inches='tight')