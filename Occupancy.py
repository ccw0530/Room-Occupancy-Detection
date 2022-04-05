import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, neighbors
from sklearn.metrics import roc_auc_score

df = pd.read_csv('datatraining.txt', index_col=0)
df = df.drop(['date'], axis=1)

df2 = pd.read_csv('datatest.txt', index_col=0)
df2 = df2.drop(['date'], axis=1)

corr_df = df.corr()
corr_df = corr_df.round(1)
fig, ax = plt.subplots()
im = ax.imshow(corr_df)
ax.set_xticks(np.arange(len(corr_df)), labels=corr_df.index)
ax.set_yticks(np.arange(len(corr_df)), labels=corr_df.index)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(corr_df)):
    for j in range(len(corr_df)):
        text = ax.text(j, i, corr_df.to_numpy()[i, j],
                       ha="center", va="center", color="w")
ax.set_title("Correlation of Features")
fig.tight_layout()
plt.show()

df = df.drop(['Humidity'], axis=1)
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]

df2 = df2.drop(['Humidity'], axis=1)
X_test = df2.iloc[:, :-1]
y_test = df2.iloc[:, -1]

model = Sequential()

model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
model.fit(X_train, y_train, batch_size=10, epochs=10, validation_data=(X_test,y_test), callbacks=[tensorboard_callback])

for k in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
    clf = LogisticRegression(random_state=0, solver=k).fit(X_train, y_train)
    print(k+' Logistic Regression Training: ', clf.score(X_train, y_train))
    print(k+' Logistic Regression Testing: ', clf.score(X_test, y_test))
    lg_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(k+' Logistic Regression AUC: ', lg_auc, '\n')

for k in ['linear', 'rbf', 'sigmoid']:
    clf = svm.SVC(kernel=k, gamma="auto", probability=True).fit(X_train, y_train)
    print(k+' SVC Training: ', clf.score(X_train, y_train))
    print(k+' SVC Testing: ', clf.score(X_test, y_test))
    svc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(k+' SVC AUC: ', svc_auc, '\n')

clf = neighbors.KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
print('K Nearest Neighbor Training: ', clf.score(X_train, y_train))
print('K Nearest Neighbor Testing: ', clf.score(X_test, y_test))
knn_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print('K Nearest Neighbor AUC: ', knn_auc, '\n')
