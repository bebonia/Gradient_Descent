from multiprocessing.sharedctypes import Value
import numpy as np

def gradient_descent(gradient,x,y, start, learn_rate, n_iter=50,tolerance=1e-6,
dtype='float64'):

    # Check if gradient is callable
    if not callable(gradient):
        raise TypeError("'gradient' must be callable")

    # Setting up the data type for NumPy Arrays
    dtype_ = np.dtype(dtype)

    # Converting x and y to NumPy Arrays
    x, y = np.array(x, dtype=dtype_), np.array(y,dtype=dtype_)
    if x.shape[0] != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")
    
    # Initializing the values of the  variables
    vector = np.array(start,dtype=dtype_)
 
    # Setting up and checking the learning rate
    learn_rate = np.array(learn_rate, dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")

    # Setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")

    # Setting up and checking the tolerance
    tolerance = np.array(tolerance, dtype=dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")

    # Performing the gradient descent loop
    for _ in range(n_iter):
        # Recalculating the difference
        diff = -learn_rate * np.array(gradient(x, y, vector), dtype_)

        # Checking if the absolute difference is small enough
        if np.all(np.abs(diff) <= tolerance):
            break

        # Updating the values of the variables
        vector += diff

    return vector if vector.shape else vector.item()

def ssr_gradient(x,y,b):
    res = b[0] + b[1] * x - y
    return res.mean(), (res * x).mean()

if __name__ == "__main__":  


    train = np.array([5,10,15,20,25,30])
    test= np.array([1,2,3,4,5,6])

    ssr = gradient_descent(
        gradient = ssr_gradient,
        x = train,
        y = test,
        start = [0.5,0.5],
        learn_rate = 0.0008,
        n_iter = 100_000
    )

    print(ssr)

    print(ssr[0] + ssr[1]*train[5])

