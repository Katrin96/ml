import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def func(x, val):
    c0, c1 = val
    return c0 + np.log(x)*c1


def derivative(x, y, val):
    c0, c1 = val
    f = c0 + c1*np.log(x) - y
    dc0 = f
    dc1 = f*np.log(x)
    return np.array([dc0, dc1])

def hessian(x, y, val):
    c0, c1 = val
    val_cnt = 2
    hes = np.empty((val_cnt, val_cnt))
    grad = derivative(x, y, val)
    f = func(x, val)

    hes[0, 0] = 0
    hes[0, 1] = 0
    hes[1, 0] = 0
    hes[1, 1] = 0

    for i in range(val_cnt):
        for j in range(val_cnt):
            hes[i, j] = grad[i] * grad[j] + (f - y) * hes[i, j]
    return hes


def SSE(x, y, val):
    res = 0
    for i in range(len(x)):
        res += 0.5*(func(x[i], val) - y[i])**2
    return res


def stability(arr, eps):
    for el in arr:
        if el > eps:
            return True
    return False


def newton_raphson(x, y, maxcnt=100, eps=1e-5):
    params_cnt = 2
    params = np.random.rand(params_cnt)
    params_old = params+10
    cnt = 0
    while (stability(np.abs(params-params_old), eps) and cnt < maxcnt):
        grad = np.zeros(params_cnt)
        hes = np.zeros((params_cnt, params_cnt))
        for i in range(len(X)):
            grad += derivative(X[i], y[i], params)
            hes += hessian(X[i], y[i], params)
        # find best step gamma
        f = lambda val: SSE(x, y, params-val*np.linalg.pinv(hes).dot(grad))
        gamma = optimize.brent(f)

        params_old = params.copy()
        # make iteration as difference of old params value and
        #  step * gradient * inverse(hessian)
        params -= np.linalg.pinv(hes).dot(grad) * gamma
        cnt += 1
    print("iteration cnt: ", cnt)
    return params

def gradient_descent(X, y, maxcnt = 100, eps=1e-5):
    params_cnt = 2
    params = np.random.randn(params_cnt)
    params_old = params+10
    cnt = 0
    while (stability(np.abs(params-params_old), eps) and cnt < maxcnt):
        grad = np.zeros(params_cnt)
        for i in range(len(X)):
            grad += derivative(X[i], y[i], params)
        # find best step gamma
        f = lambda val: SSE(X, y, params - val * grad)
        gamma = optimize.brent(f)

        params_old = params.copy()
        params -= grad * gamma
        cnt += 1
    print("iteration cnt: ", cnt)
    return params

def generate_set(n = 100):
    data = {}
    data['x'] = np.linspace(1, n, num=n)
    data['y'] = -2*np.random.rand(data['x'].size) + np.log(data['x'])*3.3
    return data

if __name__ == '__main__':
    n = 20
    data = generate_set(n)
    X = data['x']
    y = data['y']

    params_vec = gradient_descent(X, y)
    print(params_vec)
    print("SSE = %.3f" % (SSE(X, y, params_vec)))

    yest = [func(X[i], params_vec) for i in range(n)]

    plt.scatter(X, y, label='data', color="black")
    plt.plot(X, yest, label='y prediction (Gradient descent)', color="blue")
    plt.title('Nonlinear regression')
    plt.legend()

    params_vec = newton_raphson(X, y)
    print(params_vec)
    print("SSE = %.3f" % (SSE(X, y, params_vec)))

    yest = [func(X[i], params_vec) for i in range(n)]

    plt.plot(X, yest, label='y prediction (Newton-Raphson)', color="red")
    plt.legend()

    plt.show()