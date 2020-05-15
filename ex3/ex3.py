from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')
    plt.show()

def get_values(filename):
    mat = io.loadmat(filename)
    X = mat['X']
    y = mat['y']
    y[y == 10] = 0
    y = np.hstack(y)
    return X, y

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def costFunctionInnerSum(h, y):
    return -(1 - y)*np.log(1 - h) - y*np.log(h)


def lrCostFunction(theta, X, y, l):
    inner_product = np.dot(X, theta)
    h = np.array([sigmoid_function(x) for x in inner_product])
    m = y.size
    inner_sum = costFunctionInnerSum(h, y)
    theta[0] = 0
    J = 1/m * np.sum(inner_sum) + l / (2*m) * np.sum(theta**2)
    grad = 1/m * np.dot((h - y),X) + l/m * theta
    return J, grad

def one_vs_all(X, y, labels, l):
    m = y.size
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    initial_theta = np.zeros(X.shape[1])
    thetas = np.zeros((labels, X.shape[1]))
    options = {'maxiter': 500}
    for i in range(labels):
        y_i = y == i
        y_i = y_i*1
        res = optimize.minimize(lrCostFunction,
                                initial_theta,
                                (X, y_i, l),
                                jac=True,
                                method='TNC',
                                options=options)
        thetas[i,:] = res.x
    return thetas

def predict_one_vs_all(X, theta):
    m = X.shape[0]
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    p = np.dot(X, theta.T)
    p = np.argmax(p, 1)
    return p

def predict(theta_1, theta_2, X):
    m = X.shape[0]
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    z2 = np.dot(theta_1, X.T)
    z2 = sigmoid_function(z2)
    a2 = np.concatenate([np.ones((m, 1)), z2.T], axis = 1)
    z3 = np.dot(a2, theta_2.T)
    a3 = sigmoid_function(z3)
    p = np.argmax(a3, 1)
    return p

######## PART 1 ########
theta_t = np.array([-2, -1, 1, 2], dtype=float)
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
print(J, grad)

# ######## PART 2 ########
X, y = get_values('ex3data1.mat')
rand_indices = np.random.choice(y.size, 100, replace=False)
sel = X[rand_indices, :]
# displayData(sel)
l = 0.1
all_thetas = one_vs_all(X, y, 10, l)
p = predict_one_vs_all(X, all_thetas)
print("accuracy: {}".format(np.mean(p == y) * 100))

######## PART 2 ########
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

weights = io.loadmat('ex3weights.mat')

Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)
output = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(output == y) * 100))