# Cell 1
# Import the libraries
import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import yfinance as yf


# Cell 2
# Set future_days
future_days = 30

# List of tickers to use for training
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SBUX', 'MCD', '600028.SS', 'CVS', 'V', 'MA', 'ADBE']


# Cell 3
# Download stock data
dataframes = []
for ticker in tickers:
    try:
        data = yf.download(ticker, period='5y')
        data['Ticker'] = ticker
        dataframes.append(data)
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

# Combine all dataframes
df = pd.concat(dataframes)
df = df[['Adj Close', 'Ticker']]
df.reset_index(inplace=True)

# Display the first few rows of the DataFrame
# Group by ticker and show the number of rows for each ticker
df.groupby('Ticker').size()


# Cell 4
# Prepare the data
df['Prediction'] = df.groupby('Ticker')['Adj Close'].shift(-future_days)
df.dropna(inplace=True)

# Display the first few rows of the prepared DataFrame
df.head()


# Cell 5
# Prepare feature and target arrays
X = df.drop(['Prediction', 'Date', 'Ticker'], axis=1).values  # Use 'Adj Close' as the feature
y = df['Prediction'].values  # Use shifted 'Adj Close' as the target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Display the shapes of the training and testing sets
print("Training set shape:", x_train.shape, y_train.shape)
print("Testing set shape:", x_test.shape, y_test.shape)


# Cell 6
# Create and train the support vector machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
svm_confidence = svr_rbf.score(x_test, y_test)

# Create and train the decision tree regressor model
tree = DecisionTreeRegressor()
tree.fit(x_train, y_train)
tree_confidence = tree.score(x_test, y_test)

# Create and train the linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)

# The best possible score is 1.0

# Evaluate models using RMSE, MAE, and R² metrics
def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2

# Evaluate models and store results in a dictionary
results = {
    "SVR": evaluate_model(svr_rbf, x_test, y_test),
    "Decision Tree": evaluate_model(tree, x_test, y_test),
    "Linear Regression": evaluate_model(lr, x_test, y_test)
}

# Print the results in a structured format
print("\nModel Evaluation Results:")
print("-" * 30)
for model_name, (rmse, mae, r2) in results.items():
    print(f"{model_name} Model:")
    print(f"  Confidence Score (R²): {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print("-" * 30)


# Cell 7
# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Save the models
joblib.dump(svr_rbf, 'models/svr_model.pkl')
joblib.dump(tree, 'models/tree_model.pkl')
joblib.dump(lr, 'models/lr_model.pkl')