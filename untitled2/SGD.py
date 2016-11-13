import numpy as np
import matplotlib.pyplot as plt
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
def error_rate(B,c1,d1):
    e=c1.shape
    error=0
    for i in range(e[0]):
        h=float(np.dot(c1[i],B.T))
        if d1[i]*h<=0:
            error=error+1
    return (error/e[0])
def SGD(dataMatrix, classLabels):
    print(1)
    c = dataMatrix.shape
    error_list=[]
    weights=np.zeros((1,c[1]))
    d = [x for x in range(c[0])]
    m=int(10*c[0]/100)
    for j in range(10):
        random.shuffle(d)
        for i in d:
            alpha = 0.01
            lam=0.001
            weights=(1-2*lam*alpha)*weights+2*alpha*dataMatrix[i]*(int(classLabels[i])-float(np.dot(dataMatrix[i],weights.T)))
            m=m-1
            if m==0:
                error_list.append(error_rate(weights,dataMatrix,classLabels))
                m=int(10*c[0]/100)
    return error_list
if __name__ == '__main__':
    a,b=read_file(file)
    c,d=read_file(file1)
    error_table=SGD(a,b)
    print(len(error_table))
    frequency=[x*0.01*10 for x in range(0,100)]
    plt.plot(frequency, error_table, 'b*')
    plt.plot(frequency, error_table, 'r')
    plt.xlabel('frequency')
    plt.ylabel('error rate')
    plt.ylim(0, 1)
    plt.title('line_regression & gradient decrease')
    plt.legend()
    plt.show()


