import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

iris = datasets.load_iris()

# take two features
X = iris.data[:, [2, 3]]
y = iris.target


n = len(X)  # data length
xl = np.arange(n)  # data indexes

eps = 1e-5
alpha = 0.9
tetta = 0.2


# substraction of two sets
def sets_diff(a1, a2):
    a1 = np.atleast_1d(a1)
    a2 = np.atleast_1d(a2)
    if (not a1.size):
        return a1
    if (not a2.size):
        return a1
    return np.setdiff1d(a1, a2)


# union of two sets
def sets_union(a1, a2):
    a1 = np.atleast_1d(a1)
    a2 = np.atleast_1d(a2)
    if (not a1.size):
        return a2
    if (not a2.size):
        return a1
    return np.union1d(a1, a2)


# fris-function
def fris(a, b, x):
    return (distance.euclidean(a, x) - distance.euclidean(a, b)) / (distance.euclidean(a, x) +
                                                                    distance.euclidean(a, b) + eps)


# returns nearest to u object from U
def nearest_neighbor(u, U):
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(U)
    return U[nbrs.kneighbors(u, return_distance=False)]


# return new etalon (index) for Y class based on existing etalons 'etalons' and set of Xy elements of Y class
def find_etalon(Xy, etalons):
    length = len(Xy)
    if length == 1:
        return Xy[0]
    etalonsVal = []
    for i in etalons:
        etalonsVal.append(X[i])
    etalonsVal = np.asarray(etalonsVal)
    D, T, E = [], [], []
    i = 0
    arrDiff = sets_diff(xl, Xy)  # X/Xy
    for x in range(length):
        # defence
        sum = 0
        for u in range(length):
            if u == x:
                continue
            sum = sum + fris(X[Xy[u]], X[Xy[x]], nearest_neighbor(np.reshape(X[Xy[u]], (1, -1)), etalonsVal))
        # defence
        D.append(sum/(len(Xy) - 1))

        # tolerance
        sum = 0
        for v in range(len(arrDiff)):
            sum = sum + fris(X[arrDiff[v]], X[Xy[x]], nearest_neighbor(np.reshape(X[arrDiff[v]], (1,-1)), etalonsVal))
        # tolerance
        T.append(sum/(len(arrDiff)))

        # efficiency
        E.append(alpha*D[i] + (1-alpha)*T[i])

        i = i + 1
    return Xy[np.argmax((E))]  # index from Xy


def fris_stolp():
    xByClass = []
    for i in np.unique(y):
        xByClass.append(np.arange(n)[y == i])

    # step1: finding etalon0
    etalon0 = []
    for i in (np.unique(y)):
        etalon0.append(find_etalon(xByClass[i], sets_diff(xl, xByClass[i])))
    etalon0 = np.asarray(etalon0)
    print("Initial etalons:")
    print(etalon0)

    # step2: initializing etalons for all classes
    etalonsUnion = []
    for i in (np.unique(y)):
        etalonsUnion = sets_union(etalonsUnion, etalon0[i])

    etalons = []
    for i in (np.unique(y)):
        etalons.append(find_etalon(xByClass[i], sets_diff(etalonsUnion, etalon0[i])))
    print("etalons:")
    print(etalons)

    # step3: repeat steps4-6 untill X is not empty
    etalonsList = []  # use this trick for comfortable array substaction
    for i in (np.unique(y)):
        etalonsList = sets_union(etalonsList, etalons[i])

    xIndexes = xl
    while (len(xIndexes)):
        # step4: initialize correct obj
        correct = []
        for i in range(len(xIndexes)):
            index = xIndexes[i]  # elemet index
            x = X[index]
            yClass = y[index]

            etalonsYVal = []
            for j in np.atleast_1d(etalons[yClass]):
                etalonsYVal.append(X[j])
            etalonsYVal = np.asarray(etalonsYVal)

            etalonsDif = sets_diff(etalonsList, etalons[yClass])
            etalonsVal = []
            for j in np.atleast_1d(etalonsDif):
                etalonsVal.append(X[j])
            etalonsVal = np.asarray(etalonsVal)

            val = fris(x, nearest_neighbor(np.reshape(x, (1, -1)), etalonsYVal),
                       nearest_neighbor(np.reshape(x, (1, -1)), etalonsVal))
            if (val > tetta):
                correct.append(index)
        print("correct")
        print(correct)
        if (not len(correct)):
            break

        # step5: delete correct from xByClass and xIndexes
        for i in np.unique(y):
            xByClass[i] = sets_diff(xByClass[i], correct)
        xIndexes = sets_diff(xIndexes, correct)

        # step6: add new etalon for each class
        for i in np.unique(y):
            if (len(xByClass[i])):
                etalons[i] = sets_union(etalons[i], find_etalon(xByClass[i], sets_diff(etalonsList, etalons[i])))

        etalonsList = []
        for i in np.unique(y):
            etalonsList = sets_union(etalonsList, etalons[i])

        print(etalons)

    return etalons


ans = fris_stolp()
print("final etalons:")
print(ans)
etalons = []
for i in np.unique(y):
    etalons = sets_union(etalons, ans[i])
colors = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colors, s=30)
plt.scatter(X[etalons, 0], X[etalons, 1], c=y[etalons], cmap=colors, s=300)
# plt.title("Etalons for iris flower data set with alpha = %f" % alpha)
plt.title("Etalons for wine data set with alpha = %f" % alpha)
plt.show()