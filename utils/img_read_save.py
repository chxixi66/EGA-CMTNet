import numpy as np
import cv2
import os
from skimage.io import imsave

def image_read_cv2(path, mode='RGB'):
    #img_BGR = cv2.imread(path).astype('float32')
    img_BGR = cv2.imread(path)
    if img_BGR is None:
        raise FileNotFoundError(f"Error: Could not read image file '{path}'. The file may not exist or may be in an unsupported format.")
    img_BGR = img_BGR.astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb' or mode == 'YUV', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    elif mode == 'YUV':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV)
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # Convert float image to uint8 format for PNG saving
    
    if np.issubdtype(image.dtype, np.floating):
        # Normalize to 0-255 range if needed
        if image.max() > 1.0:
            image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
    
    # Save as PNG
    imsave(os.path.join(savepath, "{}.png".format(imagename)), image)