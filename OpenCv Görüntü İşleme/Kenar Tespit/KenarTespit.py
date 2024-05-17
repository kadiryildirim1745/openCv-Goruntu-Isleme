import cv2
import numpy as np

# Görüntüyü yükle
image = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel operatörüyle kenarları tespit et
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # X yönünde türev
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Y yönünde türev

# Kenarların gücünü ve yönünü hesapla
magnitude = np.sqrt(sobelx**2 + sobely**2)
#angle = np.arctan2(sobely, sobelx) * (180 / np.pi)

# Görüntüyü göster
cv2.imshow('Original Image', image)
cv2.imshow('Sobel X', cv2.convertScaleAbs(sobelx))
cv2.imshow('Sobel Y', cv2.convertScaleAbs(sobely))
cv2.imshow('Magnitude', cv2.convertScaleAbs(magnitude))
#cv2.imshow('Angle', angle.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()