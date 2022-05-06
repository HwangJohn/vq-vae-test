import matplotlib.pyplot as plt
import numpy as np

def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

def get_img(img):
    return img.permute(1,2,0)
    # npimg = img.numpy()
    # return np.transpose(npimg, (1,2,0))