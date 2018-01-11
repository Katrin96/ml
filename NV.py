import numpy as np
from scipy.spatial import distance
from pylab import plot, legend, scatter, show, clf, title

from sklearn import datasets

# Gauss kernel
def Gauss(z):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(- 0.5 * z ** 2)

# Kvartich kernel
def Kvartich(z):
    if abs(z) <= 1:
        return (1 - z ** 2) ** 2
    else:
        return 0

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
    data = generate_wave_set(100, 80)
    x = data['x_train']
    y = data['y_train']
    boston = datasets.load_boston()

    yest_nadaray_g = nadaray_watson(x, y, kernel="Gauss", h=0.1)
    yest_nadaray_k = nadaray_watson(x, y, kernel="Kvartich", h=0.1)
    print("SSE for Gauss kernel:")
    print(SSE(yest_nadaray_g, y))
    print("SSE for Kvartich kernel:")
    print(SSE(yest_nadaray_k, y))

    clf()
    scatter(x, y, label='dat', color="black")
    plot(x, yest_nadaray_g, label='y nadaray-watson with gauss kernel', color="red")
    plot(x, yest_nadaray_k, label='y nadaray-watson with kvartich kernel', color="blue")
    title('Nadaray Watson with Gauss and Kvartich kernel')
    legend()
    show()
