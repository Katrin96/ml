import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


def generate_wave_set(n_support=1000, n_train=250):
    data = {}
    data['support'] = np.linspace(0, 2 * np.pi, num=n_support)
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.sin(data['x_train']).ravel()
    data['y_train'] += 0.5 * (0.55 - np.random.rand(data['y_train'].size))
    return data

def PCA(X, y, eps=1):
    (m, n) = X.shape
    X_std = StandardScaler().fit_transform(X)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    print('Covariance matrix \n%s' %cov_mat)

    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print('Eigenvectors \n%s' %eig_vecs)
    print('Eigenvalues \n%s' %eig_vals)

    u,s,v = np.linalg.svd(X_std.T)

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_vals_sorted = []
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])
        eig_vals_sorted = np.append(eig_vals_sorted, i[0])

    tot = sum(eig_vals)
    var_exp = [(eig_vals_sorted[i] / tot)*100 for i in range(n)]
    cum_var_exp = np.cumsum(var_exp)

    var_i = np.array([np.sum(eig_vals_sorted[: i + 1])/ tot * 100.0 for i in range(n)])
    print("% of saved information with different component cnt", var_i)
    k = 2
    print('%.2f %% variance retained in %d dimensions' % (var_i[k-1], k))

    matrix_w = np.zeros((n, k))
    for i in range(k):
        ar = np.asarray(eig_pairs[i][1])
        for j in range(n):
            matrix_w[j][i] = ar[j]
    print('Matrix W:\n', matrix_w)

    Y = X_std.dot(matrix_w)
    return Y


if __name__ == '__main__':
    data = datasets.load_wine()
    X = data.data
    y = data.target
    PCA(X, y)

