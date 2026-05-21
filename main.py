import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


print("SCRIPT STARTED")
load = pd.read_csv(r"Data\Skyserver_12_30_2019 4_49_58 PM.csv")
print(load.head())
print(load.shape)
print(load.columns)

print(load['class'].head(20))

load = load[load["class"] != 'STAR']
load['AGN'] = load['class'].apply(lambda x:1 if x=='QSO' else 0)
print(load['AGN'].value_counts())

features = ['u', 'g', 'r', 'i', 'z']
X = load[features]
Y = load['AGN']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, Y_train)

y_pred = log_model.predict(X_test)

print("Accuracy", accuracy_score(Y_test, y_pred))
print("Classificatoin Report", classification_report(Y_test, y_pred))
print("Confusion Matrix \n", confusion_matrix(Y_test, y_pred))
