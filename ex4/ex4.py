from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import math

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

def sigmoid_gradient(z):
    return sigmoid_function(z) * (1 - sigmoid_function(z))

def costFunctionInnerSum(h, y):
    return -(1 - y)*np.log(1 - h) - y*np.log(h)

def make_y(num_labels, y):
    vectorized_y = np.zeros((len(y), num_labels))
    for i in range(len(y)):
        vectorized_y[i][y[i]] = 1
    return vectorized_y

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, l):
    theta_1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    theta_2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    m = X.shape[0]
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    z2 = np.dot(theta_1, X.T)
    z2 = sigmoid_function(z2)
    a2 = np.concatenate([np.ones((m, 1)), z2.T], axis = 1)
    z3 = np.dot(a2, theta_2.T)
    h = sigmoid_function(z3)
    vectorized_y = make_y(num_labels, y)
    inner_sum = costFunctionInnerSum(h, vectorized_y)
    J = 1/m * np.sum(inner_sum) + l / (2*m) * (np.sum(theta_1[1:,1:]**2) + np.sum(theta_2[1:,1:]**2))
    grad = 1/m * np.dot((h - vectorized_y).T, X)

    return J, grad

def rand_init(l_in, l_out, epsilon):
    return np.random.rand(l_out, 1 + l_in) * 2 * epsilon - epsilon

def rand_matrix(l_in, l_out, epsilon=.12):
    matrix = np.zeros((l_out, 1 + l_in))
    matrix += rand_init(l_in, l_out, epsilon)
    return matrix

def nn_cost_function_with_backpropagation(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, l):
    theta_1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                         (hidden_layer_size, (input_layer_size + 1)))

    theta_2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                         (num_labels, (hidden_layer_size + 1)))
    m = X.shape[0]
    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    z2 = a1.dot(Theta1.T)
    a2 = np.concatenate([np.ones((m, 1)), sigmoid_function(z2)], axis = 1)
    z3 = a2.dot(theta_2.T)
    h = sigmoid_function(z3)
    vectorized_y = make_y(num_labels, y)
    inner_sum = costFunctionInnerSum(h, vectorized_y)
    J = 1/m * np.sum(inner_sum) + l / (2*m) * (np.sum(theta_1[1:,1:]**2) + np.sum(theta_2[1:,1:]**2))
    grad = 1/m * np.dot((h - vectorized_y).T, X)

    diroc_3 = h - vectorized_y
    diroc_2 = diroc_3.dot(Theta2)[:, 1:] * sigmoid_gradient(z2)
    delta_1 = diroc_2.T.dot(a1)
    delta_2 = diroc_3.T.dot(a2)

    Theta1_grad = delta_1 / m + l / m * np.sum(Theta1[:,1:])
    Theta2_grad = delta_2 / m + l / m * np.sum(Theta2[:,1:])

    return J, grad

######## PART 1 ########
X, y = get_values('ex4data1.mat')
rand_indices = np.random.choice(y.size, 100, replace=False)
sel = X[rand_indices, :]
# displayData(sel)

input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9
l = 0
weights = io.loadmat('ex4weights.mat')
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                        num_labels, X, y, l)
print(J)

######## PART 2 ########
z = np.array([-1, -0.5, 0, 0.5, 1])
g = sigmoid_gradient(z)
epsilon_init = 0.12
theta_1 = rand_matrix(input_layer_size, hidden_layer_size)
theta_2 = rand_matrix(hidden_layer_size, num_labels)
nn_params = np.concatenate([theta_1.ravel(), theta_2.ravel()], axis = 0)
l = 3
J, _ = nn_cost_function_with_backpropagation(nn_params, input_layer_size, hidden_layer_size,
                                             num_labels, X, y, l)
print(J)