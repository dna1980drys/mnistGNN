import gzip
import numpy as np

class MNIST():
    def __init__(self,normalize=False,twod=False):
        with gzip.open('./train-images-idx3-ubyte.gz', 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            if normalize:
                data = data.reshape(-1,784) / 255
            else:
                data = data.reshape(-1, 784)
            self.train_image = data
        with gzip.open('./train-labels-idx1-ubyte.gz', 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            self.train_label = data
