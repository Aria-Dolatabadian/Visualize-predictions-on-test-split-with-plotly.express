import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Load and split data
a = pd.read_csv("df1.csv")
b = pd.read_csv("df2.csv")

# Convert pandas.core.frame.DataFrame to numpy.ndarray
X = np.loadtxt('df1.csv', delimiter=',')
y = np.loadtxt('df2.csv', delimiter=',')

X, y = make_moons(noise=0.3, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y.astype(str), test_size=0.25, random_state=0)

# Fit the model on training data, predict on test data
clf = KNeighborsClassifier(15)
clf.fit(X_train, y_train)
y_score = clf.predict_proba(X_test)[:, 1]

fig = px.scatter(
    X_test, x=0, y=1,
    color=y_score, color_continuous_scale='RdBu',
    symbol=y_test, symbol_map={'0': 'square-dot', '1': 'circle-dot'},
    labels={'symbol': 'label', 'color': 'score of <br>first class'}
)
fig.update_traces(marker_size=12, marker_line_width=1.5)
fig.update_layout(legend_orientation='h')
fig.show()
