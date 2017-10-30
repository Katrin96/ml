# sklearn.neighbors.KNeighborsClassifier
```
class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, 
algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)
```
Классификатор реализующий выбор к-ближайших соседей:

### Параметры:

**n_neighbors(n-соседей):** целое число, опциональный параметр (по умолчанию значение = 5)

**weights** - вызываемый, необязательный параметр (по умолчанию 'uniform')

*uniform* - все точки из каждого класса весят одинаково;
	
*distance* - взвешивает точки опираясь на дистанции;
	
*[callable]* -  принимает массив расстояний и возвращает массив такого же типа который уже содержит веса.
	
**algorithm** - опциональный параметр 

*auto* - попытается выбрать наиболее подходящий алгоритм, основанный на значениях, переданных fit-методу;
	
*ball_tree* - будет использовать BallTree;
	
*kd_tree* - будет использовать KDTree;
	
*brute* - будет использовать полный перебор.
	
**leaf_size** - целочисленный, необязательный (по умолчанию = 30)

Размер листа передается BallTree или KDTree. Оптимальное значение зависит от характера проблемы.
	
**p** - целочисленный , опциональный (по умолчанию = 2)
	
При его изменении меняется метрика Минковского.

p = 1 ~ manhattan_distance (l1) 

p = 2 ~ euclidean_distance (l2) 

Для любого p используется minkowski_distance (l_p).

**metric** - метрика, строка либо вызываемый параметр (по умолчанию “Минковский”)

Метрика расстояния используется для дерева
	
**metric_params** - необязательный (по умолчанию = None)

Дополнительные аргументы ключевых слов для метрической функции.

**n_jobs** - целочисленный ,  необязательный параметр (по умолчанию = 1)
	
Число параллельных вычислений для поиска соседей. Если данный параметр равен -1 то число вычислений устанавливается по количеству ядер процессора.

## Пример:
 > X = [[0], [1], [2], [3]]
 
 > y = [0, 0, 1, 1]
 
 > from sklearn.neighbors import KNeighborsClassifier
 
 > neigh = KNeighborsClassifier(n_neighbors=3)
 
 > neigh.fit(X, y) 
 
 KNeighborsClassifier(...)
 
 > print(neigh.predict([[1.1]]))
 
 [0]
 
 > print(neigh.predict_proba([[0.9]]))
 
 [[ 0.66666667  0.33333333]]

## Методы

`fit(X, y)` - подстраивает модель, использую Х как обучающую выборку , а y как целевые значения.

`get_params([deep])` - считывает параметры для оценки.

`kneighbors([X, n_neighbors, return_distance])` - осуществляет поиск к - соседей запрашиваемой точки.

`kneighbors_graph([X, n_neighbors, mode])` - просчитывает взвешенный граф для К - соседей точки Х.

`predict(X)` - прогнозирует для входящих данных принадлежность к классам.

`predict_proba(X)` - возвращает возможность оценить тестовую выборку Х.

`score(X, y[, sample_weight])` - возвращает среднюю точность определения меток принадлежности к классам для тестовой выборки.

`set_params(** PARAMS)` - устанавливает параметры для оценки.

```
__init__(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, 
metric_params=None, n_jobs=1, **kwargs)
```
```
fit(X, y)
```

Подстраивает модель используя Х как обучающую выборку и у как целевые значения.

### Параметры:  

**X** : *{array-like, sparse matrix, BallTree, KDTree}*. Данные обучения. Если массив или матрица, форма [n_samples, n_features] или [n_samples, n_samples], если metric = 'precomputed'. 

**y** : *{array-like, sparse matrix}*. Целевые значения формы = [n_samples] или [n_samples, n_outputs] 

```
get_params(deep=True)
```

### Параметры: 
**deep** : булевый , необязательный. Если истина, то вернет параметры для этой оценки и подобъекты…

### Возвращает: 

**params**: Имена параметров отображаются на их значения. 

```
kneighbors(X=None, n_neighbors=None, return_distance=True)
```
Ищет к - соседей точки. Возвращает индексы и расстояния до соседей каждой точки.

### Параметры: 
**X** : массив, состояние (n_query, n_features) или (n_query, n_indexed), если metric == 'precomputed'

Точка запроса или точки. Если не указано, возвращаются соседи каждой проиндексированной точки. В этом случае точка запроса не считается ее собственным соседом.

**n_neighbors** : целочисленный

Количество соседей для получения (значение по умолчанию - это значение, переданное конструктору).

**return_distance** : лонический тип, необязательное. По умолчанию используется значение True.

Если False, дистанции не будут возвращены

### Возвращает:

**dist** : массив

Массив, представляющий длины к точкам, присутствует только в том случае, если return_distance = True

**ind** : массив
Индексы ближайших точек в матрице совокупности.

## Примеры

В следующем примере мы строим класс NeighborsClassifier из массива, представляющего наш набор данных, и спрашиваем, кто ближайший к [1,1,1]

> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]

> from sklearn.neighbors import NearestNeighbors

> neigh = NearestNeighbors(n_neighbors=1)

> neigh.fit(samples) 

NearestNeighbors(algorithm='auto', leaf_size=30, ...)

> print(neigh.kneighbors([[1., 1., 1.]])) 

(array([[ 0.5]]), array([[2]]...))


 Как вы можете видеть, он возвращает [[0.5]] и [[2]], что означает, что элемент находится на расстоянии 0,5 и является третьим элементом образцов (индексы начинаются с 0). Вы также можете запросить несколько точек:
 
> X = [[0., 1., 0.], [1., 0., 1.]]

> neigh.kneighbors(X, return_distance=False) 

array([[1],
       [2]]...)

```
kneighbors_graph(X=None, n_neighbors=None, mode=’connectivity’)
```

Вычисляет (взвешенный) граф k-соседей для точек из X

### Параметры:	

**X** : массиы, состояние (n_query, n_features) или (n_query, n_indexed), если metric == 'precomputed'

Точка запроса или точки. Если не указано, возвращаются соседи каждой проиндексированной точки. В этом случае точка запроса не считается ее собственным соседом.

**n_neighbors** : целое

Количество соседей для каждого образца. 

**mode** : {'connectivity', 'distance'}, необязательный

Тип возвращаемой матрицы: ‘connectivity’ вернет матрицу смежности с единицами и нулями, на ‘distance’ ребра - евклидово расстояние между точками.

## Возвращает:

**A** : разреженная матрица в формате CSR, shape = [n_samples, n_samples_fit]

n_samples_fit - количество выборок в данных, которые были установлены в A [i, j], присваивается вес ребра, соединяющего i с j.

### Примеры:

> X = [[0], [3], [1]]
> from sklearn.neighbors import NearestNeighbors
> neigh = NearestNeighbors(n_neighbors=2)
> neigh.fit(X) 
NearestNeighbors(algorithm='auto', leaf_size=30, ...)
> A = neigh.kneighbors_graph(X)
> A.toarray()
array([[ 1.,  0.,  1.],
       [ 0.,  1.,  1.],
       [ 1.,  0.,  1.]])

```
predict( X )
```

Предсказывает метки классов для предоставленных данных

## Параметры:

**X** : массив, состояние (n_query, n_features) или (n_query, n_indexed), если metric == 'precomputed'

Испытательные образцы.

## Возвращает:

**y** : массив, состояние [n_samples] или [n_samples, n_outputs]

Классы для каждого образца данных.

```
predict_proba( X )
```

Оценки вероятности возврата для тестовых данных X.

## Параметры:

**X** : массив, состояние (n_query, n_features) или (n_query, n_indexed), если metric == 'precomputed'

Испытательные образцы.

## Возвращает:

**p** : массив, состояние = [n_samples, n_classes] или список n_outputs
таких массивов, если n_outputs> 1. Вероятности класса входных выборок. Классы упорядочиваются по лексикографическому порядку.

```
score( X , y , sample_weight = None )
```

Возвращает среднюю точность данных данных и меток.

В классификации с несколькими метками это точность подмножества, которая является жесткой метрикой, поскольку для каждого образца требуется, чтобы каждый набор меток был правильно предсказан.

## Параметры:	

**X** : массив, состояние = (n_samples, n_features)

Испытательные образцы.

**y** : массив, состояние = (n_samples) или (n_samples, n_outputs)

Истинные метки для X.

**sample_weight** :  массив, состояние = [n_samples], необязательный

Вес образца.

## Возвращает:	

**score** : float

Средняя точность self.predict (X) wrt. у.

```
set_params( ** params )
```
Задает параметры этой оценки.

Метод работает как с простыми оценками, так и с вложенными объектами. Последние имеют параметры формы, <component>__<parameter> так что можно обновлять каждый компонент вложенного объекта.

