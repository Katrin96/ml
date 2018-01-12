import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def ridge(X, y, cov_matrix, n, tau=0.01):
    cov_matrix_ridge = cov_matrix + tau * np.eye(n)
    print(cov_matrix_ridge)
    eigenval_ridge, eigenvect_ridge = np.linalg.eigh(cov_matrix_ridge)
    print("new eigen values:")
    print(", ".join("%.2f" % f for f in eigenval_ridge))

    return np.linalg.inv(cov_matrix_ridge).dot(X.T).dot(y)


def SSE(y_pred, y_real):
    return np.linalg.norm(y_pred - y_real) ** 2


def generate_wave_set(n_support=1000, n_train=250):
    data = {}
    data['support'] = np.linspace(0, 2 * np.pi, num=n_support)
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.sin(data['x_train']).ravel()
    data['y_train'] += 0.5 * (0.55 - np.random.rand(data['y_train'].size))
    return data


def plot_error(x, y):
    error = []
    weights = []
    tau_val = np.logspace(-6, 6, 200)
    # weight = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)  # initial weights

    for a in tau_val:
        weight_ridge = ridge(fX, y, cov_matrix_standart, n=5, tau=a)
        weights.append(weight_ridge)
        # error.append(mean_squared_error(weight_ridge, weight))
        error.append(mean_squared_error(y, x.dot(weight_ridge)))

    plt.figure(figsize=(20, 6))

    plt.subplot(121)
    ax = plt.gca()
    ax.plot(tau_val, weights)
    ax.set_xscale('log')
    plt.xlabel('tau')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')

    plt.subplot(122)
    ax = plt.gca()
    ax.plot(tau_val, error)
    ax.set_xscale('log')
    plt.xlabel('tau')
    plt.ylabel('error')
    plt.title('Dependence of the error on the regularization parameter')
    plt.axis('tight')

    plt.show()


if __name__ == '__main__':
    n = 20
    data = generate_wave_set(100, n)
    X = data['x_train']
    y = data['y_train']

    fX = np.array([X]).T
    fX = np.concatenate((fX, np.power(fX, 2), np.power(fX, 3), 2 * fX, 3 * fX), axis=1)

    cov_matrix_standart = fX.T.dot(fX)
    print(cov_matrix_standart)
    # get the eigenvalues and eigenvectors
    eigenval, eigenvect = np.linalg.eigh(cov_matrix_standart)
    print("initial eigen values:")
    print(", ".join("%.2f" % f for f in eigenval))

    ans = ridge(fX, y, cov_matrix_standart, n=5, tau=0.3)
    print("weight vector:")
    print(ans)

    plot_error(fX, y)
    print(ans)
    print(mean_squared_error(y, fX.dot(ans)))
