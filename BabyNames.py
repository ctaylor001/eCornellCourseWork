import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

def hashfeatures(baby, B, FIX):
    """
    Input:
        baby : a string representing the baby's name to be hashed
        B: the number of dimensions to be in the feature vector
        FIX the number of chunks to extract and hash for each string

    Output:
        v: a feature vector representing the input string
    """
# B = 128
# FIX = 3
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "Suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v
 #   xtest = name2features(yourname,B=DIMS,LoadFile=False)
def name2features(filename, B, 3 Loadfile):
#def name2features(filename, B=128, LoadFile):
    
    if loadfile:
        with open(filename) as f:
             babynames = [x.rstrip() for x in f.readlines() if len(x) >0]
    else:
        babynames =  filename.split('\n') 
    n = len(babynames) 
    X = np.zeros((n, 128))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], 128, 3)
    return X

def genTrainFeatures(dimension=128):
    """
    Input: 
        dimension: desired dimension of the features
    Output: 
        X: n feature vectors of dimensionality d (nxd)
        Y: n labels (-1 = girl, +1 = boy) (n)
    """
    
    # Load in the data
    Xgirls = name2features("girls.train", B=dimension)
    Xboys = name2features("boys.train", B=dimension)
    X = np.concatenate([Xgirls, Xboys])
    
    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])
    
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    
    return X[ii, :], Y[ii]

X, Y = genTrainFeatures(128)
print(X),print(Y)

def naivebayesPY(X, Y):
    """
    naivebayesPY(Y) returns [pos,neg]

    Computation of P(Y)
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (nx1)

    Output:
        pos: probability p(y=1)
        neg: probability p(y=-1)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    Y = np.concatenate([Y, [-1,1]])
    n = len(Y)
    # YOUR CODE HERE
    pos = np.mean(Y == 1)
    neg = np.mean(Y == -1)
    
    return pos,neg

pos, neg = naivebayesPY(X,Y)

def naivebayesPXY(X,Y):
    """
    naivebayesPXY(X, Y) returns [posprob,negprob]
    
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (n)
    
    Output:
        posprob: probability vector of p(x_alpha = 1|y=1)  (d)
        negprob: probability vector of p(x_alpha = 1|y=-1) (d)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = X.shape
    X = np.concatenate([X, np.ones((2,d)), np.zeros((2,d))])
    Y = np.concatenate([Y, [-1,1,-1,1]])
    n, d = X.shape
    
    posprob = np.mean(X[Y == 1], axis=0)
    negprob = np.mean(X[Y == -1], axis=0)
    return posprob, negprob
    
    
    
posprob, negprob = naivebayesPXY(X,Y)

def naivebayesPXY_test1():
    x = np.array([[0,1],[1,0]])
    y = np.array([-1,1])
    pos, neg = naivebayesPXY(x,y)
    return pos, neg
 
    naivebayesPXY_test1()

def loglikelihood(posprob, negprob, X_test, Y_test):
    """
    loglikelihood(posprob, negprob, X_test, Y_test) returns loglikelihood of each point in X_test
    
    Input:
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
        Y_test : labels (-1 or +1) (n)
    
    Output:
        loglikelihood of each point in X_test (n)
    """
    n, d = X_test.shape
    loglikelihood = np.zeros(n)
    
    # YOUR CODE HERE
    
    positive = (Y_test == 1)
    negative = (Y_test == -1)
    
    loglikelihood[positive] = X_test[positive]@np.log(posprob) + (1 - X_test[positive])@np.log(1 - posprob)
    loglikelihood[negative] = X_test[negative]@np.log(negprob) + (1 - X_test[negative])@np.log(1 - negprob)
    

    return loglikelihood

# compute the loglikelihood of the training set
posprob, negprob = naivebayesPXY(X,Y)
loglikelihood(posprob,negprob,X,Y) 
print(loglikelihood(posprob,negprob,X,Y) )

def naivebayes_pred(pos, neg, posprob, negprob, X_test):
    """
    naivebayes_pred(pos, neg, posprob, negprob, X_test) returns the prediction of each point in X_test
    
    Input:
        pos: class probability for the negative class
        neg: class probability for the positive class
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
    
    Output:
        prediction of each point in X_test (n)
    """
    n, d = X_test.shape
    ratio1 = loglikelihood(posprob, negprob, X_test, np.ones(n)) - loglikelihood(posprob, negprob, X_test, -np.ones(n))
    ratio2 = np.log(pos) - np.log(neg)
    
    loglikelihood_ratio = ratio1 + ratio2
    
        
    prediction = - np.ones(n)
    prediction[loglikelihood_ratio > 0] =1

    
    return prediction 



DIMS = 128
print('Loading data ...')
X,Y = genTrainFeatures(DIMS)
print('Training classifier ...')
pos, neg = naivebayesPY(X, Y)
posprob, negprob = naivebayesPXY(X, Y)
error = np.mean(naivebayes_pred(pos, neg, posprob, negprob, X) != Y)
print('Training error: %.2f%%' % (100 * error))

while True:
    print('Please enter a baby name>')
    yourname = input()
    if len(yourname) < 1:
        break
    xtest = name2features(yourname,B=DIMS,LoadFile=False)
    pred = naivebayes_pred(pos, neg, posprob, negprob, xtest)
    if pred > 0:
        print("%s, I am sure you are a baby boy.\n" % yourname)
    else:
        print("%s, I am sure you are a baby girl.\n" % yourname)