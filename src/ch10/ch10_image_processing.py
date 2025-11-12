from skimage import io
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fname = askopenfilename()
image = io.imread(fname)
plt.imshow(image)
plt.show()

# type of image object
type(image)

# image shape
print(image.shape)

# image data
print(image[:,:,1]) # green channel

# color to gray
from skimage import color

gray_image = color.rgb2gray(image)
print(gray_image.shape)
plt.imshow(gray_image, cmap='gray')
plt.show()

# resize
from skimage import transform

new_shape = (image.shape[0]//2, image.shape[1]//2, image.shape[2])
small = transform.resize(image=image, output_shape=new_shape)

print(small.shape)
plt.imshow(small)
plt.show()

from skimage.transform import rotate
import numpy as np

plt.imshow(np.flip(image))              # flip
plt.show()
plt.imshow(rotate(image, angle=45))     # rotate
plt.show()

# filters
from skimage.filters import sobel_h

plt.imshow(sobel_h(gray_image))
plt.show()

# save image
io.imsave('/home/shpark/Downloads/test.png', image)