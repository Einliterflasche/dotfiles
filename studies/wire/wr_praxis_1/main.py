import numpy as np

from lib import timedcall, plot_2d


def matrix_multiplication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate product of two matrices a * b.

    Arguments:
    a : first matrix
    b : second matrix

    Return:
    c : matrix product a * b

    Raised Exceptions:
    ValueError: if matrix sizes are incompatible

    Side Effects:
    -

    Forbidden: numpy.dot, numpy.matrix, numpy.einsum
    """

    m, n_a = a.shape
    n_b, p = b.shape

    if not n_a == n_b:
        raise ValueError(f"incompatible matrix shapes: ({m}x{n_a})*({n_b}x{p})")

    # Initialize result matrix with zeros
    c = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            for k in range(n_a):
                c[i][j] += a[i][k] * b[k][j] 
    
    return c


def compare_multiplication(nmax: int, n: int) -> dict:
    """
    Compare performance of numpy matrix multiplication (np.dot()) and matrix_multiplication.

    Arguments:
    nmax : maximum matrix size to be tested
    n : step size for matrix sizes

    Return:
    tr_dict : numpy and matrix_multiplication timings and results {"timing_numpy": [numpy_timings],
    "timing_mat_mult": [mat_mult_timings], "results_numpy": [numpy_results], "results_mat_mult": [mat_mult_results]}

    Raised Exceptions:
    -

    Side effects:
    Generates performance plots.
    """

    x, y_mat_mult, y_numpy, r_mat_mult, r_numpy = [], [], [], [], []
    tr_dict = dict(timing_numpy=y_numpy, timing_mat_mult=y_mat_mult, results_numpy=r_numpy, results_mat_mult=r_mat_mult)

    for m in range(2, nmax, n):

        # Create random mxm matrices a and b
        a = np.random.rand(m, m)
        b = np.random.rand(m, m)

        # Execute functions and measure the execution time
        time_mat_mult, result_mat_mult = timedcall(matrix_multiplication, a, b)
        time_numpy, result_numpy = timedcall(np.dot, a, b)

        # Add calculated values to lists
        x.append(m)
        y_numpy.append(time_numpy)
        y_mat_mult.append(time_mat_mult)
        r_numpy.append(result_numpy)
        r_mat_mult.append(result_mat_mult)

    # Plot the computed data
    plot_2d(x_data=x, y_data=[y_mat_mult, y_numpy], labels=["matrix_mult", "numpy"],
            title="NumPy vs. for-loop matrix multiplication",
            x_axis="Matrix size", y_axis="Time", x_range=[2, nmax])

    return tr_dict


def machine_epsilon(fp_format: np.dtype) -> np.number:
    """
    Calculate the machine precision for the given floating point type.

    Arguments:
    fp_format : floating point format, e.g. float32 or float64

    Return:
    eps : calculated machine precision

    Raised Exceptions:
    -

    Side Effects:
    Prints out iteration values.

    Forbidden: numpy.finfo
    """

    # TODO: create epsilon element with correct initial value and data format fp_format
    eps = fp_format.type(1.0)


    # Create necessary variables for iteration
    one = fp_format.type(1.0)
    two = fp_format.type(2.0)
    i = 0

    print('  i  |       2^(-i)        |  1 + 2^(-i)  ')
    print('  ----------------------------------------')
    
    while True:
        middle = (one + two) / fp_format.type(2.0)
        print(f"middle: {middle}")

        if middle > one:
            two = middle
        elif middle == one:
            eps = (two - 1)
            break
        else:
            print(f"MAkdsakdjas wWHAT? {middle} < 1!!!")
            break


    print('{0:4.0f} |  {1:16.8e}   | equal 1'.format(i, eps))
    return eps



def close(a: np.ndarray, b: np.ndarray, eps: np.number=1e-08) -> bool:
    """
    Compare two floating point matrices. 

    Arguments:
    a : first matrix
    b : second matrix
    eps: tolerance

    Return:
    c : if a is close to b (within the tolerance)

    Raised Exceptions:
    ValueError: if matrix sizes are incompatible

    Side Effects:
    -

    Forbidden: numpy.isclose, numpy.allclose
    """
    if a.shape != b.shape:
        raise ValueError(f"incompatible matrix sizes: ({a.shape[0]}x{a.shape[1]}) and ({b.shape[0]}x{b.shape[1]})")

    isclose = True

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if abs(a[i][j] - b[i][j]) > eps:
                isclose = False

    return isclose


def rotation_matrix(theta: float) -> np.ndarray:
    """
    Create 2x2 rotation matrix around angle theta.

    Arguments:
    theta : rotation angle (in degrees)

    Return:
    r : rotation matrix

    Raised Exceptions:
    -

    Side Effects:
    -
    """

    # create empty matrix
    r = np.zeros((2, 2))

    # convert angle to radians
    theta_rad = theta * np.pi / 180

    # calculate diagonal terms of matrix
    r[0][0] = r[1][1] = np.cos(theta_rad)

    # off-diagonal terms of matrix
    r[1][0] = np.sin(theta_rad)
    r[0][1] = -np.sin(theta_rad)

    return r


def inverse_rotation(theta: float) -> np.ndarray:
    """
    Compute inverse of the 2d rotation matrix that rotates a 
    given vector by theta.
    
    Arguments:
    theta: rotation angle
    
    Return:
    Inverse of the rotation matrix

    Forbidden: numpy.linalg.inv, numpy.linalg.solve
    """

    # compute inverse rotation matrix

    # just create a rotation matrix which rotates by minus theta degrees

    m = np.zeros((2, 2))

    # convert angle to radians
    theta_rad = theta * np.pi / 180

    # calculate diagonal terms of matrix
    m[0][0] = m[1][1] = np.cos(-theta_rad)

    # off-diagonal terms of matrix
    m[1][0] = np.sin(-theta_rad)
    m[0][1] = -np.sin(-theta_rad)
    
    return m


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")

