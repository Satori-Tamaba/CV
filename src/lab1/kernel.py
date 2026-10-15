import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
import matplotlib.pyplot as plt

class Kernel(ABC):

    @abstractmethod
    def __call__(self, X):
        pass



class GaussianKernel(Kernel):

    def __init__(self, sigma: float, size: int):
        self.sigma = sigma
        if size % 2 == 0:
            raise  ValueError("Только нечетное число")

        self.size = size
        x, y = np.mgrid[:self.size, :self.size]
        self.kernel = (1/(2*np.pi*self.sigma**2)) * np.exp(-(x**2 + y**2)/ (2*self.sigma**2))
        print(self.kernel)
        self.kernel = self.kernel / self.kernel.sum()


    def __call__(self, X:np.ndarray):
        X = np.asarray(X)
        X = np.atleast_3d(X)
        X = np.moveaxis(X, -1, 0)

        c_axis,img_height, img_width =  X.shape
        pad_size = self.size // 2
        pad_mats = [np.pad(X[i], pad_width= ((pad_size, pad_size), (pad_size, pad_size)) , mode= "edge") for i in range(c_axis)]


        for k, ax in enumerate(pad_mats):

            for i in range(img_height):

                for j in range(img_width):
                    region = ax[i: i+self.size,j: j+self.size]
                    X[k,i,j] = np.sum(region * self.kernel)
        return  np.moveaxis(X, 0, -1 )


