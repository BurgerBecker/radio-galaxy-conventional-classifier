import numpy as np
import cv2 as cv
import pyfits as pf

def read_fits(name):
    fh = pf.open(name)
    image = fh[0].data
    return image

def preprocess(image, sig):
    image2 = image.copy()
    image2 = image2/np.linalg.norm(image2)
    n,m = image.shape
    mean = image.mean()
    for i in range(n):
        for j in range(m):      
            if (image[i][j] < (mean + sig)):
                image2[i][j] = 0
    return image2

def get_centre(label):    
    Mb = cv.moments(label)    
    Bx = int(Mb["m10"] / Mb["m00"])
    By = int(Mb["m01"] / Mb["m00"])
    return Bx, By

def preprocess_reverse(image, sigmaM = 0):
    """
    Binary thresholds out all parts of the images with a value greater than the value of sigmaM from the mean.
    
    Keyword arguments:
    image -- fits image you want thresholded.
    sigmaM -- threshold value (default 0)
    
    Returns:
    image2 -- binary thresholded image with value set to 5
    """
    image2 = image.copy()
    image2 = image2/np.linalg.norm(image2)
    n,m = image.shape
    mean = image.mean()
    hi,hj = np.unravel_index(image.argmax(), image.shape)
    for i in range(n):
        for j in range(m):
            if (image[i][j] > (mean + sigmaM)):
                image2[i][j] = 5
            else:
                image2[i][j] = 0
    return image2

def preproc_svd(image,k):
    U, sigma, V = np.linalg.svd(preprocess(image,2*np.std(image)))

    reconstimg = np.matrix(U[:, :k]) * np.diag(sigma[:k]) * np.matrix(V[:k, :])

    ret,thresh = cv.threshold(reconstimg,np.std(reconstimg),np.max(reconstimg),cv.THRESH_BINARY)
    
    return thresh

def get_lobes_svd(image, lw, thresh):
    """Takes in an image, it's labeled counter-part and the binary threshold of it's SVD preprocessed result. Returns an array of the labels and an image of the thresholded area."""
    l,w = thresh.shape
    img = np.zeros([l,w])
    labels = np.array([])
    for i in range(0, l):
        for j in range(0, w):
            if thresh[i][j] != 0:
                img[i][j] = image[i][j]
                label = lw[i][j]                
                if not label in labels and label != 0:
                    labels = np.append(labels,label)
    return labels, img

def get_lobes(image, lw, label, disp=False):
        """Takes in an image and its labeled counter-part and isolates the labels set out in label. Returns an image and label counter-part of the specified labels."""
        l, w = lw.shape
        imcopy = np.zeros([l, w])
        labeledLobes = np.zeros([l,w])
        hotspots = []
        for k in label:
            temp = np.zeros([l, w])
            for i in range(0, l):
                for j in range(0, w):
                    if lw[i, j] == k and k != 0:
                        temp[i, j] = image[i, j]
                        labeledLobes[i, j] = k
            imcopy = temp + imcopy
        if disp == True:
            plt.figure(figsize=(12,12))
            plt.set_cmap("gist_stern")
            plt.imshow(imcopy)
            plt.show()
        return imcopy, labeledLobes
