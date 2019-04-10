import numpy as np

if __name__ == '__main__':
    a = np.array([[1, 2, 4], [4, 5, 6]])
    print(a.argmax())
    print(np.unravel_index(a.argmax(), a.shape))
