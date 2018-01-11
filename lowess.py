import numpy as np
from scipy.spatial import distance
import pylab as plt
from sklearn.metrics import mean_squared_error


# Gauss kernel
def Gauss(z):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(- 0.5 * z ** 2)


# Kvartich kernel
def Kvartich(z):
    if abs(z) <= 1:
        return (1 - z ** 2) ** 2
    else:
        return 0


def stability(arr, eps):
    for el in arr:
        if el > eps:
            return True
    return False


def nadaray_watson(x, y, kernel="Gauss", h=0.5):
    possibles = globals().copy()
    possibles.update(locals())
    K = possibles.get(kernel)

    n = len(x)
    w = []
    for t in range(n):
        w.append([])
        for i in range(n):
            w[t].append(K(distance.euclidean(x[t], x[i]) / h))
    w = np.array(w)
    yest = (w * y[:, None]).sum(axis=0) / w.sum(axis=0)
    return yest


def lowess(x, y, kernel="Gauss", kernel1="Kvartich", h=0.5, eps=1e-5):
    possibles = globals().copy()
    possibles.update(locals())
    K = possibles.get(kernel)
    K1 = possibles.get(kernel1)
    n = len(x)

    gamma = np.ones(n)
    gamma_old = np.zeros(n)
    yest = np.zeros(n)
    cnt = 0

    while stability(np.abs(gamma - gamma_old), eps):
        cnt += 1
        # find weights as multiplication of gammas and kernel function on dist
        w = []
        for t in range(n):
            w.append([])
            for i in range(n):
                w[t].append(K(distance.euclidean(x[t], x[i]) / h) * gamma[t])
        w = np.array(w)
        yest = (w * y[:, None]).sum(axis=0) / w.sum(axis=0)

        # calc new gammas as Kernel function(error)
        err = np.abs(yest - y)
        gamma = [K1(err[j]) for j in range(n)]
        if cnt > 5:
            break
    return yest


def SSE(y_pred, y_real):
    res = y_pred - y_real
    res *= res
    return np.sum(res)


def generate_wave_set(n_support=1000, n_train=250):
    data = {'support': np.linspace(0, 2 * np.pi, num=n_support)}
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.sin(data['x_train']).ravel()
    data['y_train'] += 0.5 * (0.55 - np.random.rand(data['y_train'].size))
    return data


if __name__ == '__main__':
    x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]
    x = np.array(x)
    y = [1, 2, 4.3, 3, 2, 2, 1.5, 1.3, 1.5, 1.7, 1.8, 2]
    y = np.array(y)

    yest_nadaray = nadaray_watson(x, y)
    yest_lowess = lowess(x, y)
    print("MSE for Nadaray-Watson:")
    print(mean_squared_error(y, yest_nadaray))
    print("MSE for Lowess:")
    print(mean_squared_error(y, yest_lowess))

    plt.clf()
    plt.scatter(x, y, label='data', color="black")
    plt.plot(x, yest_nadaray, label='y nadaray-watson', color="green")
    plt.plot(x, yest_lowess, label='y lowess', color="yellow")
    plt.title('Nadaray Watson vs Lowess')
    plt.legend()
    plt.show()
