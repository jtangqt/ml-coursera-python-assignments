import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import optimize
import matplotlib.cm as cm

def showPlot(filename):
    data = pd.read_csv(filename, header=None)
    # data_0 = data.loc[data[2] == 0]
    # data_1 = data.loc[data[2] == 1]
    # plt.plot(data_0[[0]], data_0[[1]], 'ro')
    # plt.plot(data_1[[0]], data_1[[1]], 'bo')
    return data

def plotDecisionBoundary(data, theta):
    data_0 = data.loc[data[2] == 0]
    data_1 = data.loc[data[2] == 1]
    plt.plot(data_0[[0]], data_0[[1]], 'ro')
    plt.plot(data_1[[0]], data_1[[1]], 'bo')
    x = np.linspace(0, 100, 1000)
    y = -(theta[0] + theta[1]*x)/theta[2]
    plt.plot(x, y, 'purple')
    plt.show()

def plotDecisionBoundaryRegularized(data, theta):
    data_0 = data.loc[data[2] == 0]
    data_1 = data.loc[data[2] == 1]
    plt.plot(data_0[[0]], data_0[[1]], 'ro')
    plt.plot(data_1[[0]], data_1[[1]], 'bo')
    x1 = np.linspace(-.8, 1.2, 100)
    x2 = x1.copy()
    z = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            features = pd.DataFrame([[x1[i], x2[j]]])
            z[i, j] = np.dot(mapFeatures(features),theta)
        print("{}/100".format(i))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cpf = ax.contourf(x1,x2,z, 20, cmap=cm.Greys_r)
    colours = ['w' if level<0 else 'k' for level in cpf.levels]
    cp = ax.contour(x1,x2,z, 20, colors=colours)
    ax.clabel(cp, fontsize=12, colors=colours)
    plt.show()

def sigmoidFunction(z):
    return 1/(1 + math.exp(-z))

def calculateHypothesis(data, theta):
    return sigmoidFunction(theta[0] + theta[1]*data[0] + theta[2]*data[1])

def costFunctionInnerSum(h, y):
    return -(1 - y)*math.log(1 - h) - y*math.log(h)

def costFunction(theta, data):
    m = len(data)
    data['h'] = data.apply(lambda x: calculateHypothesis(x, theta), axis = 1)
    inner_sum = data.apply(lambda x: costFunctionInnerSum(x['h'], x[2]), axis = 1)
    j = 1/m * np.sum(inner_sum)
    grad = gradientDescent(data, theta)
    return j, grad

def gradientDescent(data, theta):
    m = len(data)
    x = np.append(np.ones((m,1)), data[[0, 1]], axis=1)
    data['h'] = data.apply(lambda x: calculateHypothesis(x, theta), axis = 1)
    inner_sum = np.dot((data['h'] - data[2]), x)
    grad = 1/m * inner_sum
    return grad

def predict(data, theta):
    return calculateHypothesis(data, theta)

def mapFeatures(data):
    m = len(data)
    degree = 6
    mapped_features = pd.DataFrame(np.ones((m, 1)))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            power = str(i) + "," + str(j)
            mapped_features[power] = data[0]**(i - j)*data[1]**j
    return mapped_features

def gradientDescentRegularized(data, y, theta, l):
    m = len(data)
    z = np.dot(theta, data.T)
    h = np.array([sigmoidFunction(x) for x in z])
    inner_sum = np.dot((h - y), data)
    theta[0] = 0
    grad = 1/m * inner_sum + l/m * theta
    return grad

def costFunctionRegularized(theta, data, y, l):
    m = len(data)
    z = np.dot(theta, data.T)
    h = np.array([sigmoidFunction(x) for x in z])
    inner_sum = np.array([costFunctionInnerSum(h[i], y[i]) for i in range(m)])
    J = 1/m * np.sum(inner_sum) + l/(2*m) * np.sum(theta[1:]**2)
    grad = gradientDescentRegularized(data, y, theta, l)
    return J, grad

######## PART 1 ########
data = showPlot("ex2data1.txt")
plt.show()
theta = np.array([-24, 0.2, 0.2])
j, grad = costFunction(theta, data)
# print(j, grad)
options = {'maxiter': 400}
res = optimize.minimize(costFunction,
                        theta,
                        (data),
                        jac=True,
                        method='TNC',
                        options=options)

cost = res.fun
theta = res.x
# plotDecisionBoundary(data, theta)
is_accepted = predict([45, 85], theta)

######## PART 1 ########
data = showPlot("ex2data2.txt")
mapped_features = mapFeatures(data)
y = data[2]
l = 1
theta = np.zeros(len(mapped_features.columns))
j, grad = costFunctionRegularized(theta, mapped_features, y, l)

test_j, test_grad = costFunctionRegularized(np.ones(len(mapped_features.columns)), mapped_features, y, 10)

options= {'maxiter': 100}

res = optimize.minimize(costFunctionRegularized,
                        theta,
                        (mapped_features, y, l),
                        jac=True,
                        method='TNC',
                        options=options)

cost = res.fun
theta = res.x
print(theta)
plotDecisionBoundaryRegularized(data, theta)
