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


    
#     # YOUR CODE HERE
# #    raise NotImplementedError()
## Created my own array for the excercise 
# # little test
#x = np.array([0.19183646, 0.48121646, 0.41110048, 0.96940754, 0.22901644, 0.85920568, 0.70751275, 0.27379106, 0.66048124, 0.66006196]) 
#x = np.array([0,1,]) 
# print(x)
#y = -1
#w = np.array([0.61776051, 0.14132546, 0.31739066, 0.16899387, 0.64821564, 0.52378534, 0.72528514, 0.06154031, 0.6576074,  0.04163673])
#w = np.array([1,1])
#print(w)
#w1 = perceptron_update(x,y,w)
#print(w1)


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
    
    M = 100
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

N = 100;
d = 10;
#x = np.array([1,3,-1,4]).reshape(2,2)
x = np.random.rand(N,d)
#w = np.array([0,0])
w = np.random.rand(1,d)
y = np.sign(w.dot(x.T))[0]
w, b = perceptron(x,y)
print(w, b)





  
