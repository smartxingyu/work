import numpy as np
import matplotlib.pyplot as plt
import random
# file=open('dataset1-a9a-training.txt','r')
# file1=open('dataset1-a9a-testing.txt','r')

def read_file(file_3):
    matrix = []
    dataMatrix = []
    classLabels = []
    for line in file_3:
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
    return error/e[0]
def SGD(dataMatrix, classLabels):
    c = dataMatrix.shape
    weights=np.zeros((1,c[1]))
    d = [x for x in range(c[0])]
    m=int(10*c[0]/100)
    error_list=[]
    for j in range(10):
        count=0
        random.shuffle(d)
        for i in d:
            count=count+1
            alpha = 0.01
            f=[]
            for ii in range(c[1]):
                if weights[0,ii]<=0:
                    f.append(-1)
                else:
                    f.append(1)
            # print(type(classLabels[i]))
            # print(type(np.dot(dataMatrix[i],weights.T)))
            # print(np.e**(-float(classLabels[i])*float(np.dot(dataMatrix[i],weights.T))))
            weights=weights-alpha*((-classLabels[i]*dataMatrix[i]*np.e**(-float(classLabels[i])*float(np.dot(dataMatrix[i],weights.T))))/(1+np.e**(-float(classLabels[i])*float(np.dot(dataMatrix[i],weights.T))))+2*np.mat([f]))
            m=m-1
            if m==0:
                m=int(10*c[0]/100)
                error_list.append(error_rate(weights,dataMatrix,classLabels))
    return error_list
if __name__ == '__main__':
    file_1=open('covtype-training.txt','r')
    file_2=open('covtype-testing.txt','r')
    a,b=read_file(file_1)
    c,d=read_file(file_2)
    error_table=SGD(a,b)
    frequency=[x*0.01*10 for x in range(0,100)]
    plt.plot(frequency, error_table, 'b*')
    plt.plot(frequency, error_table, 'r')
    plt.xlabel('frequency')
    plt.ylabel('error rate')
    plt.ylim(0, 1)
    plt.title('error——rate')
    plt.legend()
    plt.show()
