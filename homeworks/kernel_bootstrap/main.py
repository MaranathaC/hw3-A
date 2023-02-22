from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return (np.outer(x_i, x_j) + 1) ** d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return np.exp(-gamma * np.add.outer(x_i, -x_j) ** 2)


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    n = len(x)
    K = kernel_function(x, x, kernel_param)
    return np.linalg.solve(K + _lambda * np.eye(n), y.reshape((n, 1)))


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    n = x.shape[0]

    mse = 0

    for i in range(num_folds):
        val_indices = range(i * fold_size, (i + 1) * fold_size)
        training_indices = list(set(range(n)) - set(val_indices))

        x_train, y_train = x[training_indices], y[training_indices]
        x_val, y_val = x[val_indices], y[val_indices]

        alpha = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        y_pred = kernel_function(x_val, x_train, kernel_param) @ alpha

        mse += np.mean((y_pred.T - y_val) ** 2)

    return mse / num_folds


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2 for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    gamma = 1 / np.median(np.outer(x, x).reshape(len(x) ** 2)) ** 2
    i = np.linspace(-5, -1)
    err = np.zeros(len(i))

    for j, _lambda in enumerate(10**i):
        err[j] = cross_validation(x, y, rbf_kernel, gamma, _lambda, num_folds)

    return 10 ** i[np.argmin(err)], gamma


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You do not really need to search over gamma. 1 / median((x_i - x_j) for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """
    i = np.linspace(-5, -1)
    ds = np.arange(5, 26, 1)
    err = np.zeros((len(i), len(ds)))
    for j, _lambda in enumerate(10 ** i):
        for k, d in enumerate(ds):
            err[j][k] = cross_validation(x, y, poly_kernel, d, _lambda, num_folds)
    j, k = np.unravel_index(np.argmin(err), err.shape)
    return 10 ** i[j], ds[k]


@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid
        C. Repeat A, B with x_300, y_300

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    _lambda_rbf, gamma = rbf_param_search(x_30, y_30, len(x_30))
    print("RBF30 Lambda: ", _lambda_rbf, " Gamma: ", gamma)

    _lambda_poly, d = poly_param_search(x_30, y_30, len(x_30))
    print("Poly30 Lambda: ", _lambda_poly, " D: ", d)

    # RBF Kernel Graph 30 data points
    x_plot = np.linspace(0, 1, 100)
    y_true = f_true(x_plot)

    alpha_rbf = train(x_30, y_30, rbf_kernel, gamma, _lambda_rbf)
    y_rbf = rbf_kernel(x_plot, x_30, gamma) @ alpha_rbf

    fig, ax = plt.subplots()
    ax.plot(x_plot, y_true, label='True Function')
    ax.scatter(x_30, y_30, label='Data Points')
    ax.plot(x_plot, y_rbf, label='RBF Kernel')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()

    # Poly kernel Graph 30 data points
    alpha_poly = train(x_30, y_30, poly_kernel, d, _lambda_poly)
    y_poly = poly_kernel(x_plot, x_30, d) @ alpha_poly

    fig, ax = plt.subplots()
    ax.plot(x_plot, y_true, label='True Function')
    ax.scatter(x_30, y_30, label='Data Points')
    ax.plot(x_plot, y_poly, label='Poly Kernel')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()

    _lambda_rbf, gamma = rbf_param_search(x_300, y_300, 10)
    print("RBF300 Lambda: ", _lambda_rbf, " Gamma: ", gamma)
    _lambda_poly, d = poly_param_search(x_300, y_300, 10)
    print("Poly300 Lambda: ", _lambda_poly, " D: ", d)

    # RBF Graph 300 data points
    alpha_rbf = train(x_300, y_300, rbf_kernel, gamma, _lambda_rbf)
    y_rbf = rbf_kernel(x_plot, x_300, gamma) @ alpha_rbf

    fig, ax = plt.subplots()
    ax.plot(x_plot, y_true, label='True Function')
    ax.scatter(x_300, y_300, label='Data Points')
    ax.plot(x_plot, y_rbf, label='RBF Kernel')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()

    # Poly kernel Graph 300 data points
    alpha_poly = train(x_300, y_300, poly_kernel, d, _lambda_poly)
    y_poly = poly_kernel(x_plot, x_300, d) @ alpha_poly

    fig, ax = plt.subplots()
    ax.plot(x_plot, y_true, label='True Function')
    ax.scatter(x_300, y_300, label='Data Points')
    ax.plot(x_plot, y_poly, label='Poly Kernel')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
