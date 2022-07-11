import numpy as np

def gradient_descent(gradient,x,y, start, learn_rate, n_iter=50,tolerance=1e-6):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * np.array(gradient(x,y,vector))
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    
    return vector

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

