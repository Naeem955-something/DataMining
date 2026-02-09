# STEP 1: Load Dataset from Colab Path
file_path = "/content/online_retail_II.xlsx"
df = pd.read_excel(file_path)

print("Dataset Loaded Successfully")
print("Shape of dataset:", df.shape)
df.head()


# STEP 2: Error Detection - Dataset Info
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())


# STEP 3: Error Fixing - Handle Missing Values
df['Customer ID'] = df['Customer ID'].fillna(df['Customer ID'].median())
df['Description'] = df['Description'].fillna("Unknown")

print("Missing values fixed")


# STEP 4: Noise / Outlier Detection using IQR
Q1 = df['Quantity'].quantile(0.25)
Q3 = df['Quantity'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['Quantity'] < Q1 - 1.5*IQR) | (df['Quantity'] > Q3 + 1.5*IQR)]
print("Total Outliers Detected:", outliers.shape[0])


# STEP 5: Remove Outliers
df = df[(df['Quantity'] >= Q1 - 1.5*IQR) & (df['Quantity'] <= Q3 + 1.5*IQR)]
print("Outliers Removed")


# STEP 6: Feature Engineering - Total Price
df['TotalPrice'] = df['Quantity'] * df['Price']
print("TotalPrice feature created")


# STEP 7: Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['Quantity']]
y = df['TotalPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Linear Regression R² Score:", lr.score(X_test, y_test))


# STEP 8: Multiple Linear Regression
X = df[['Quantity', 'Price']]
y = df['TotalPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlr = LinearRegression()
mlr.fit(X_train, y_train)

print("Multiple Linear Regression R² Score:", mlr.score(X_test, y_test))


# STEP 9: Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df[['Quantity']])

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

pr = LinearRegression()
pr.fit(X_train, y_train)

print("Polynomial Regression R² Score:", pr.score(X_test, y_test))


# STEP 10: Logistic Regression
from sklearn.linear_model import LogisticRegression

df['HighValue'] = (df['TotalPrice'] > df['TotalPrice'].median()).astype(int)

X = df[['Quantity', 'Price']]
y = df['HighValue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

print("Logistic Regression Accuracy:", log_reg.score(X_test, y_test))


# STEP 11: Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)

print("Decision Tree Accuracy:", dt.score(X_test, y_test))


# STEP 12: Confusion Matrix, Precision, Recall, F1, ROC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

y_pred = log_reg.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1]))


# STEP 13: K-Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("KNN Accuracy:", knn.score(X_test, y_test))


# STEP 14: K-Means Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Quantity', 'Price']])

print("K-Means Clustering Done")
df[['Quantity', 'Price', 'Cluster']].head()


# STEP 15: Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)

print("Naive Bayes Accuracy:", nb.score(X_test, y_test))


# STEP 16: Apriori Algorithm
install mlxtend

from mlxtend.frequent_patterns import apriori, association_rules

basket = df.groupby(['Invoice', 'Description'])['Quantity'] \
           .sum().unstack().fillna(0)

basket = basket.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

