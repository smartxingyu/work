import numpy as np
import random
file=open('dataset1-a9a-training.txt','r')
file1=open('dataset1-a9a-testing.txt','r')
# file=open('covtype-training.txt','r')
# file1=open('covtype-testing.txt','r')
def read_file(file):
    matrix = []
    dataMatrix = []
    classLabels = []
    for line in file:
        matrix.append(line.strip().split(','))
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            temp = float(matrix[i][j])
            matrix[i][j] = temp
        dataMatrix.append(matrix[i][0:len(matrix[0])-1])
        classLabels.append([matrix[i][len(matrix[0])-1]])
    return np.mat(dataMatrix),np.mat(classLabels)
def SGD(dataMatrix, classLabels):
    c = dataMatrix.shape
    weights=np.zeros((1,c[1]))
    print(weights.shape)
    d = [x for x in range(c[0])]
    for j in range(2):
        random.shuffle(d)
        for i in d:
            alpha = 0.01
            weights=(1-4*alpha)*weights+2*alpha*dataMatrix[i]*(int(classLabels[i])-float(np.dot(dataMatrix[i],weights.T)))
    return weights
a,b=read_file(file)
c,d=read_file(file1)
B=SGD(a,b)
print(B)
def error_rate(B,c1,d1):
    e=c1.shape
    error=0
    count=0
    count1=0
    for i in range(e[0]):
        h=float(np.dot(c1[i],B.T))
        if d1[i]*h<=0:
            error=error+1
        if h<0:
            count=count+1
        if d1[i]<0:
            count1=count1+1
    print(error/e[0])
    print(count/e[0])
    # print(count1/e[0])
error_rate(B,c,d)


