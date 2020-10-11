import numpy as np
import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from sklearn.model_selection import train_test_split
import pandas as pd
# X = [1335, 1345, 1355, 1365, 1375, 1385, 1395]
# y = [3134, 41785, 55481, 87063, 98544, 110643, 118564]

data_set = pd.read_csv('Anzali_Population.csv')

X = data_set.iloc[:, 0].values.reshape(-1, 1)
y = data_set.iloc[:, 1].values.reshape(-1, 1)

print(X)


lr = LinearRegression()
lr.fit(X, y)

print("Training set score: {:.2f}".format(lr.score(X, y)))

Y_pred = lr.predict(X)

plt.scatter(X, y)
plt.plot(X, Y_pred, color="red")
plt.xticks(rotation='vertical')
plt.title("Anzali population")
plt.xlabel('Year')
plt.ylabel('Total Population')

plt.show()  # display graph

