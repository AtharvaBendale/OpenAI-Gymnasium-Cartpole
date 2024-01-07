import numpy as np

def rgb_to_gray(img : np.ndarray) -> np.ndarray:
    img = img[35:-15:2,::2,:]
    img = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
    img = (img-np.mean(img))/(np.std(img))
    return img