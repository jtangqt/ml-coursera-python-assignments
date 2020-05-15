from scipy import io
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

def get_values(x, y):
    plt.plot(x, y, 'ro')
    plt.show()

def linear_reg_cost_function(x, y, theta, lambda_ = 0):
    m = x.shape[0]
    h = x.dot(theta)
    inner_sum = h - y.T
    j = 1/(2*m) * np.sum(inner_sum**2) + lambda_ / (2*m) * np.sum(theta[1:])
    grad = 1/m * np.dot(inner_sum, x)+ lambda_ / m * np.concatenate([np.zeros(1), theta[1:]])
    return j, grad


def train_linear_reg(linear_reg_cost_function, x, y, lambda_=0.0, maxiter=200):
    # Initialize Theta
    initial_theta = np.zeros(x.shape[1])

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linear_reg_cost_function(x, y, t, lambda_)

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': maxiter}

    # Minimize using scipy
    res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
    return res.x

def learning_curves(x, y, x_val, y_val):
    m = x.shape[0]
    train_err = np.zeros(m)
    val_err = np.zeros(m)
    for i in range(m):
        theta = train_linear_reg(linear_reg_cost_function,x[:i+1],y[:i+1], lambda_=0)
        h = x[:i+1].dot(theta)
        train_err[i] = 1/(2*(i+1)) * np.sum((h - y[:i+1].T)**2)
        val_h = x_val.dot(theta)
        val_err[i] = 1/(2*m) * np.sum((val_h - y_val.T)**2)
    return train_err, val_err

def show_err_plot(x, train_err, val_err):
    plt.plot(x, train_err)
    plt.plot(x, val_err)
    plt.show()

def map_features(x, power):
    mapped_x = np.zeros((len(x), power + 1))
    for i in range(power + 1):
        if i is not 0:
            normalized_x = feature_normalization((x**i))
        else:
            normalized_x = x**i
        mapped_x[:, i] = normalized_x.T
    return mapped_x

def feature_normalization(data):
    mean = np.mean(data)
    variance = np.sqrt(np.var(data))
    return (data - mean)/variance

mat = io.loadmat('ex5data1.mat')
x = mat['X']
y = mat['y']
x_test = mat['Xtest']
y_test = mat['ytest']
x_val = mat['Xval']
y_val = mat['yval']
plt.plot(x, y, 'ro')
plt.show()
theta = np.array([1,1])
m = x.shape[0]
x1 = np.concatenate([np.ones((m, 1)), x], axis = 1)
x_val1 = np.concatenate([np.ones((x_val.shape[0], 1)), x_val], axis = 1)
j, grad = linear_reg_cost_function(x1, y, theta)
print(j, grad)
theta = train_linear_reg(linear_reg_cost_function, x1, y, lambda_=0)
print(theta)

###### Part 2 #####
train_err, val_err = learning_curves(x1, y, x_val1, y_val)
# print(train_err, val_err)
x_err = np.linspace(1, m, m)
show_err_plot(x_err, train_err, val_err)

###### Part 3 #####
power = 8
mapped_x = map_features(x, power)
mapped_val = map_features(x_val, 8)
theta = np.ones((power + 1, 1))
train_err, val_err = learning_curves(mapped_x, y, mapped_val, y_val)
# print(train_err, val_err)
show_err_plot(x_err, train_err, val_err)
