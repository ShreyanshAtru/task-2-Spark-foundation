#        THE SPARK FOUNDATION INTERNSHIP

#        SHREYANSH JAIN
#        TASK 2 : Predict the optimum number of clusters from given 'iris' 
#                  data set and represt it visually


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import os

# Read the CSV file
os.chdir("E:/")

iris = pd.read_csv('Iris.csv')
iris.head()
iris.shape
iris.info()
iris.describe()
iris.isnull().sum()
iris.Species.value_counts()

iris.drop('Id', axis = 1 ,inplace = True)
iris['ID'] = iris.index + 100

iris.head()

plt.figure(figsize = (10,10))
f = iris.columns[:2]
for i in enumerate(f):
    plt.subplot(2,2,i[0]+1)
    sns.distplot(iris[i[1]])
    
plt.figure(figsize = (10,10))
f = iris.columns[:2]
for i in enumerate(f):
    plt.subplot(2,2,i[0]+1)
    sns.boxplot(x = i[1], data = iris)
    
#     outlier

a1 = iris['SepalWidthCm'].quantile(0.01)
a2 = iris['SepalWidthCm'].quantile(0.99)

iris['SepalWidthCm'][iris['SepalWidthCm']<= a1] = a1
iris['SepalWidthCm'][iris['SepalWidthCm']>= a2] = a2


# calculating Hopkins statics
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan

def hopkins(X):
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors = 1).fit(X.values)
    
    rand_X = sample(range(0,n,1), m)
    
    ujd =  []
    wjd = []
    for j in range(0,m):
        u_dist , _ = nbrs.kneighbors(uniform(np.amin(X,axis = 0),
                    np.amax(X,axis=0),d).reshape(1,-1),
                    2,return_distance = True)
    
        ujd.append(u_dist[0][1])
        w_dist , _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1,-1),2,
                                     return_distance = True)
        wjd.append(w_dist[0][1])
        
    H = sum(ujd) / (sum(ujd) +sum(wjd))
    if isnan(H):
        print(ujd,wjd)
        H= 0
    return H

hopkins(iris.drop(['ID', 'Species'],axis = 1))

#  Scaling   

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
df = s.fit_transform(iris.drop(['ID','Species'], axis = 1))
df


df = pd.DataFrame(df)
df.columns = iris.columns[:-2]
df.head()


# KMeans clustering 
#silhourette Analysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
ss = []
for k in range(2,11):
    kmean = KMeans(n_clusters = k).fit(df)
    ss.append([k, silhouette_score(df,kmean.labels_)])

sil = pd.DataFrame(ss)
plt.plot(sil[0], sil[1])
plt.show()

#   Elbow curve
# Finding the optiminum number of clusters for k-mean clsassification

x = iris.iloc[:,[0,1,2,3]].values

from sklearn.cluster import KMeans
wc= []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++',
                    max_iter = 300, n_init = 10 , random_state = 0)
    
    kmeans.fit(x)
    wc.append(kmeans.inertia_)
    
#plotting the results onto a line graph

plt.plot(range(1,11),wc)
plt.title('the elnow method')
plt.xlabel('number of clusters')
plt.ylabel('wc')
plt.show()
    

#KMean with K = 3
kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300,
                n_init = 10, random_state=0)
y_kmeans = kmeans.fit_predict(x)

# visualising the clusters - un the first two columns

plt.figure(figsize = (8,6))

plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1],s =100,
            c= 'purple', label = 'Iris-virginica')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1],s =100,
            c= 'red', label = 'Iris-virginica')
plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1],s = 100,
            c = 'blue', label = 'Iris-setosa')
# plotting the centroid of the clusters

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s =100 , c = 'yellow', label = 'Centroids' )

plt.legend()
plt.show()

    
    



####   Thank You ####
























