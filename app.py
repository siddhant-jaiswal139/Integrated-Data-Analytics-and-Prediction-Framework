from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

@app.route("/upload", methods=["POST"])
def upload_file():

    file = request.files["file"]
    df = pd.read_csv(file)

    df = df.dropna()
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return jsonify({"error": "Need at least 2 numeric columns"})

    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    results = []

    # Ridge
    start = time.time()
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    ridge_acc = round(ridge.score(X_test, y_test), 2)
    ridge_time = round(time.time() - start, 2)
    results.append({"model": "Ridge Regression", "accuracy": ridge_acc, "time": ridge_time})

    # KNN
    start = time.time()
    try:
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_acc = round(knn.score(X_test, y_test), 2)
    except:
        knn_acc = 0
    knn_time = round(time.time() - start, 2)
    results.append({"model": "KNN", "accuracy": knn_acc, "time": knn_time})

    # Lasso ✅ NEW
    start = time.time()
    lasso = Lasso()
    lasso.fit(X_train, y_train)
    lasso_acc = round(lasso.score(X_test, y_test), 2)
    lasso_time = round(time.time() - start, 2)
    results.append({"model": "Lasso Regression", "accuracy": lasso_acc, "time": lasso_time})

    # KMeans
    start = time.time()
    kmeans = KMeans(n_clusters=3, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    kmeans_time = round(time.time() - start, 2)
    results.append({"model": "Clustering", "accuracy": "-", "time": kmeans_time})

    chart_data = {
        "first_column": numeric_df.iloc[:, 0].tolist()[:20],
        "second_column": numeric_df.iloc[:, 1].tolist()[:20],
        "cluster_labels": clusters.tolist()[:20]
    }

    best_model = max(
        [r for r in results if r["accuracy"] != "-"],
        key=lambda x: x["accuracy"]
    )

    return jsonify({
        "rows": df.shape[0],
        "columns": df.shape[1],
        "results": results,
        "best_model": best_model,
        "chart_data": chart_data
    })
if __name__ == "__main__":
    app.run(debug=True)