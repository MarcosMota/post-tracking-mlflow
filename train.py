import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import argparse

mlflow.set_tracking_uri("http://127.0.0.1:5000")

db = load_diabetes()
X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

 
mlflow.set_experiment('diabetes_regressor')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Diabetes regressor training')
    parser.add_argument('--n_estimators',type=int, default=100, help='The number of trees in the forest.')
    parser.add_argument('--max_depth', type=int,default=6, help='The maximum depth of the tree.')
    parser.add_argument('--max_features', type=int,default=3, help='The number of features.')

    args = parser.parse_args()


    with mlflow.start_run():
        
        # Salve os parametros
        mlflow.log_param("num_trees", args.n_estimators)
        mlflow.log_param("maxdepth", args.max_depth)
        mlflow.log_param("max_feat", args.max_features)

        rf = RandomForestRegressor(n_estimators = args.n_estimators, max_depth = args.max_depth, max_features = args.max_features)
        rf.fit(X_train, y_train)

        # Previsões
        predictions = rf.predict(X_test)
        
        # Salve o modelo
        mlflow.sklearn.log_model(rf, "random-forest-model")
        
        # Salve as métricas
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric("mse", mse)
