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

def sgd(
    gradient,x,y,n_vars = None,start=None,decay_rate = 0.0,
    learn_rate=0.1,batch_size=1,n_iter=50,tolerance=1e-06
    ,dtype="float64",random_state=None
):
    #Check if gradient is callable
    if not callable(gradient):
        raise TypeError("'gradient' must be callable")
    
    #setting up the data type for numpy arrays
    dtype_ = np.dtype(dtype)

    # converting the x and y to numpy arrays
    x, y  = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    n_obs = x.shape[0]

    if n_obs != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")

    xy = np.c_[x.reshape(n_obs,-1), y.reshape(n_obs,1)]

    print(y.reshape(n_obs,-1))
    print(xy)

    #initialize the RNG
    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    # initializing the values
    vector = (
        rng.normal(size=int(n_vars)).astype(dtype_)
        if start is None else
        np.array(start, dtype=dtype_)
    )
    #setting up and checking the learning rate
    learn_rate = np.array(learn_rate,dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn rate' must be greater than zero")

    #setting up and checking the batches
    batch_size = int(batch_size)
    if not 0 < batch_size <= n_obs:
        raise ValueError(
            "'batch size' must be greater than zero and less than or equal to number of observations"
        )

    #setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")

    #setting up and checking the tolerance
    tolerance = np.array(tolerance, dtype = dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")


    decay_rate = np.array(decay_rate, dtype=dtype_)
    if np.any(decay_rate < 0) or np.any(decay_rate > 1):
        raise ValueError("'decay_rate' must be between zero and one")

    # Setting the difference to zero for the first iteration
    diff = 0
    
    #Performing the gradient descent loop
    for _ in range(n_iter):
        rng.shuffle(xy)

        for start in range(0,n_obs,batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            #recalculate the difference
            grad = np.array(gradient(x_batch,y_batch,vector),dtype_)
            diff = decay_rate * diff - learn_rate * grad

            if np.all(np.abs(diff) <= tolerance):
                break

            vector  += diff

    return vector if vector.shape  else vector.item()


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

    x = np.array([5, 15, 25, 35, 45, 55])
    y = np.array([5, 20, 14, 32, 22, 38])

    a =sgd(
     ssr_gradient, x, y, n_vars=2, learn_rate=0.0001,
     decay_rate=0.8, batch_size=3, n_iter=100_000, random_state=0
     )
    print(a)


