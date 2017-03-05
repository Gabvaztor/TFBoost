def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x),axis=0)


scores = [1.0, 2.0]
print (softmax(scores))
