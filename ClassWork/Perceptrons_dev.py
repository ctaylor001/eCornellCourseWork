import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt


def perceptron_update(x,y,w):
    """
    function w=perceptron_update(x,y,w);
    
    Implementation of Perceptron weights updating
    Input:
    x : input vector of d dimensions (d)
    y : corresponding label (-1 or +1)
    w : weight vector of d dimensions
    
    Output:
    w : weight vector after updating (d)

  From email 
	    Part one and two finds out w and b with the training data 
	    Part one - perception_update(x, y, w) return w : one time update on w ( += y*x)
	    Part two - perception(xs, ys) return (w,b)  : Iteration of Part one - multiple updates by the number of misclassification
#     """

    w += np.dot(np.transpose(x),y)


    return w


N = 100;
d = 10;   
#     # YOUR CODE HERE
# #    raise NotImplementedError()
## Created my own array for the excercise 
# # little test
#x = np.array([9,6],[7,7],[8,9],[9,8],[-2,2],[-1,-3],[-3,-4],[-4,-6])
x = np.array([9,6,7,7,8,9,9,8,-2,2,-1,-3,-3,-4,-4,-6]).reshape(8,2)
#x = np.array([0,1,]) 
#x = np.random.rand(8,2)
print(x)
y = -1
w = 0 #np.array([0,0])
#w = np.array([1,1])
print(w)
w1 = perceptron_update(x,y,w)
print(w1)


def perceptron(xs,ys):
    """
    function w=perceptron(xs,ys);
    
    Implementation of a Perceptron classifier
    Input:
    xs : n input vectors of d dimensions (nxd)
    ys : n labels (-1 or +1)
    
    Output:
    w : weight vector (1xd)
    b : bias term
    """

    n, d = xs.shape     # so we have n input vectors, of d dimensions each
    w = np.zeros(d)
    b = 0.0
    
    M = 8
    i = 0
    
    while (i < M):
        misclass = 0
        # Randomize the order in the training data
        for i in np.random.permutation(n): 
            if ys[i]*(np.dot(w, xs[i]) + b) <= 0:
                perceptron_update(xs[i], ys[i], w)
                b += ys[i]
                misclass += 1
        if (misclass == 0):
            break
        i += 1
    return (w,b)

N = 8;
d = 2;
x = np.array([9,6,7,7,8,9,9,8,-2,2,-1,-3,-3,-4,-4,-6]).reshape(8,2)
#x = np.random.rand(N,d)
#w = np.array([0,0])
w = np.random.rand(1,2)
y = np.sign(w.dot(x.T))[0]
w, b = perceptron(x,y)
print(w, b)

# np.array([9,6,7,7,8,9,9,8,-2,2,-1,-3,-3,-4,-4,-6]).reshape(4,4)


def classify_linear(xs,w,b = None):
    w = w.flatten()
    predictions = np.zeros(xs.shape[0])
    
    # YOUR CODE HERE
    if b == None:
        b = 0
    p = np.matmul(xs,w) + b
    for i in range(len(p)):
        if p[i] >=0:
            predictions[i] = 1
        else:
            predictions[i] = -1
    return predictions
  

xs = np.random.rand(50000,20)-0.5 # draw random data 
w0 = np.random.rand(20)
b0 =- 0.1 # with bias -0.1
ys = classify_linear(xs,w0,b0)
print(ys)
#uniquepredictions=np.unique(ys) # check if predictions are only -1 or 1
#return set(uniquepredictions)==set([-1,1])