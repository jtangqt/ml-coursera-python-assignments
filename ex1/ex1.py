import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def showPlot(filename):
    data = pd.read_csv(filename, header=None)
    m = len(data)
    x = np.append(np.ones((m,1)), data.values[:,0].reshape(m,1), axis=1)
    y = data.values[:,1].reshape(m, 1)
    # plt.plot(x, y, 'ro')
    # plt.show()
    return x, y

def showPlot2(filename):
    data = pd.read_csv(filename, header=None)
    m = len(data)
    x = np.append(np.ones((m,1)), data[data.columns[:2]], axis=1)
    y = data.values[:,2].reshape(m, 1)
    return x, y


def gradientDescent(X, y, theta, alpha, m):
    h = X.dot(theta)
    theta = theta - alpha * (1 / m) * np.dot(X.T, h-y)
    return theta


def computeCost(X, y, theta, m):
    h = X.dot(theta)
    cost = 1 / (2 * m) * np.sum((h - y) ** 2)
    return cost


def batchGradientDescent(X, y, theta, iterations):
    alpha = 0.01
    m = len(X)
    y.reshape(m, 1)
    cost = computeCost(X, y, theta, m)
    for i in range(iterations):
        theta = gradientDescent(X, y, theta, alpha, m)
        cost = computeCost(X, y, theta, m)
    return cost, theta

def featureNormalization(data):
    mean = np.mean(data)
    variance = np.sqrt(np.var(data))
    return (data - mean)/variance

def batchGradientDescent2(X, y, theta, iterations, alpha):
    m = len(X)
    j = []
    for i in range(iterations):
        h = np.dot(X, theta)
        cost = 1/(2*m)*np.sum((h-y)**2)
        theta = theta - alpha / m * np.dot(X.T, (h-y))
        j.append(cost)
    print(theta)
    return j

def normalEquation(X, y):
    inside_inverse = np.dot(X.T, X)
    inverse = np.linalg.inv(inside_inverse)
    outside_inverse = np.dot(X.T, y)
    return np.dot(inverse, outside_inverse)


######## PART 1 ########
x1, y1 = showPlot('ex1data1.txt')
x1 = np.array(x1)
y1 = np.array(y1)
x0 = np.ones(len(x1))
theta = np.zeros((2, 1))
J, theta = batchGradientDescent(x1, y1, theta, 1500)
plt.plot(x1[:,1], y1, 'ro')
x2 = np.linspace(4, 24, 1000)
y2 = theta[1]*x2+theta[0]
plt.plot(x2, y2)
plt.show()

######## PART 2 ########
x2, y2 = showPlot2('ex1data2.txt')
x2[:,1] = featureNormalization(x2[:,1])
x2[:,2] = featureNormalization(x2[:,2])
theta = np.zeros((3, 1))
iterations = 50
alpha = 0.3
j = batchGradientDescent2(x2, y2, theta, iterations, alpha)
x = np.linspace(0, 50, 50)
plt.plot(x, j, "red")
plt.show()


######## PART 3 ########
theta = normalEquation(x2, y2)
print(theta)