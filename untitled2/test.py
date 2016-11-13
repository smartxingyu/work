import numpy as np
import matplotlib.pyplot as plt
import random
c=[[1,2,3,4],[5,9,7,3],[9,1,8,4]]
b=np.mat(c)
d=range(0,10)
print(d)
print(b)
random.shuffle(c)
print(np.e**2)
if __name__ == '__main__':
    years=[1,2,3,4,5]
    price=[1,1,2,2,3]
#     file = open(E:machine_learningdatasetshousing_datahousing_data_ages.txt, 'r')
#     linesList = file.readlines()
# #     print(linesList)
#     linesList = [line.strip().split(,) for line in linesList]
#     file.close()
#     print(linesList:)
#     print(linesList)
# #     years = [string.atof(x[0]) for x in linesList]
#     years = [x[0] for x in linesList]
#     print(years)
#     price = [x[1] for x in linesList]
#     print(price)
    plt.plot(years, price, 'b*')#,label=$cos(x^2)$)
    plt.plot(years, price, 'r')
    plt.xlabel('years')
    plt.ylabel('price')
    plt.ylim(0, 5)
    plt.title('line_regression & gradient decrease')
    plt.legend()
    plt.show()
