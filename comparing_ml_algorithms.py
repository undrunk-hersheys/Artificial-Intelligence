"""
=========================================================
The Digit Dataset
=========================================================

This dataset is made up of 1797 8x8 images. Each image,
like the one shown below, is of a hand-written digit.
In order to utilize an 8x8 figure like this, we'd have to
first transform it into a feature vector with length 64.

See `here
<https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits>`_
for more information about this dataset.

"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt

from sklearn import datasets

# Load the digits dataset
digits = datasets.load_digits()

# Display the last digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")
#plt.show()

#print(digits)

'''
import numpy as np

class sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)

class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)

class sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None)

'''

print("Data shape:", digits.data.shape)  # 데이터 배열의 형태
print("Target shape:", digits.target.shape)  # 타겟(레이블) 배열의 형태
print("Images shape:", digits.images.shape)  # 이미지 배열의 형태
#print("Dataset Description:\n", digits.DESCR)  # 데이터셋에 대한 설명

#print(digits.images[-2])
#print(digits.target[-2])

import numpy as np

# 숫자 이미지 데이터셋의 레이블에 대한 통계 정보
unique_labels, label_counts = np.unique(digits.target, return_counts=True)

print("number label:", unique_labels)
print("count per label:", label_counts)
print("total count:", len(digits.target))
print("max count label:", unique_labels[np.argmax(label_counts)], ":", np.max(label_counts))
print("min count label:", unique_labels[np.argmin(label_counts)], ":", np.min(label_counts))


import numpy as np
from sklearn.linear_model import LinearRegression
np.set_printoptions(threshold=np.inf)
##X=np.array(digits.data)
##print(X)
Y=np.array(digits.target)
print(Y)


import matplotlib.pyplot as plt

from sklearn import datasets

# Load the digits dataset
digits = datasets.load_digits()

# Display the last digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation="nearest")
#plt.show()

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 불러오기
digits = load_digits()

# 훈련 데이터와 타겟 설정
X = digits.data
y = digits.target


# 데이터를 무작위로 섞기 위해 인덱스 생성
indices = np.arange(X.shape[0])
np.random.seed(14)
np.random.shuffle(indices)

# 데이터를 무작위로 섞은 후, 새로운 순서로 데이터 재배열
X_shuffled = X[indices]
y_shuffled = y[indices]

# 새로운 순서로 재배열된 데이터를 훈련과 테스트 세트로 나눔
X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=14)

##LINEAR REGRESSION
# Linear Regression 모델 초기화
model = LinearRegression()

# 모델 학습
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print("LINEAR REGRESSION Accuracy:", accuracy)

##LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000000)  # max_iter를 증가시키면 반복 횟수를 늘릴 수 있습니다.
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("LOGISTIC REGRESSION Accuracy:", accuracy)

##SVM
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", accuracy)

##MLP
from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(random_state=14)
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100),random_state=14)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("MLP Accuracy:", accuracy)

##FDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

##FDA PLOT
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 데이터 로드
digits = load_digits()
X = digits.data
y = digits.target

# LinearDiscriminantAnalysis 모델 생성
lda = LinearDiscriminantAnalysis(n_components=2)  # 2개의 주요 구성 요소를 가진 LDA 모델
#qda = QuadraticDiscriminantAnalysis(store_covariance=True)
# 데이터를 2차원으로 변환하여 모델에 적합
X_r2 = lda.fit(X, y).transform(X)

# 클래스별로 산점도 플롯
colors = ['navy', 'turquoise', 'darkorange', 'green', 'red', 'blue', 'purple', 'yellow', 'pink', 'brown']
lw = 2

for color, i, target_name in zip(colors, range(10), digits.target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of Digits dataset')
plt.show()

##LDA QDA PLOT
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Load digits data
digits = load_digits()
X = digits.data
y = digits.target

# Apply PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# LDA and QDA models
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

# Fit models on reduced data
lda.fit(X_pca, y)
qda.fit(X_pca, y)

# Plot decision boundaries
def plot_decision_boundaries(model, title):
    h = .02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title(title)

# Plot decision boundaries for LDA and QDA
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_decision_boundaries(lda, "Linear Discriminant Analysis (PCA-reduced)")

plt.subplot(1, 2, 2)
plot_decision_boundaries(qda, "Quadratic Discriminant Analysis (PCA-reduced)")

plt.tight_layout()
plt.show()

