from flask import Flask, request, render_template
import yfinance as yf
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Load models
svr_model = joblib.load('models/svr_model.pkl')
tree_model = joblib.load('models/tree_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker')
    future_days = int(request.form.get('future_days', 30))

    if not ticker:
        return render_template('index.html', error='No ticker provided')

    try:
        # Download stock data
        data = yf.download(ticker, period='1d')
        if data.empty:
            raise ValueError("No data found, symbol may be delisted")

        current_adj_close = data['Adj Close'].values[0]
        current_date = data.index[0].strftime('%Y-%m-%d')

        # Make prediction for future days
        predictions = []
        best_buy_date = None
        best_buy_price = float('inf')
        best_sell_date = None
        best_sell_price = float('-inf')

        for day in range(1, future_days + 1):
            features = np.array([[current_adj_close]])

            # Predict using each model
            svr_prediction = svr_model.predict(features)
            tree_prediction = tree_model.predict(features)
            lr_prediction = lr_model.predict(features)

            # Create future date
            future_date = pd.to_datetime(current_date) + pd.DateOffset(days=day)
            future_date_str = future_date.strftime('%Y-%m-%d')

            predictions.append({
                'Date': future_date_str,
                'SVR Prediction': svr_prediction[0],
                'Tree Prediction': tree_prediction[0],
                'LR Prediction': lr_prediction[0]
            })

            # Update current_adj_close to the latest SVR prediction
            current_adj_close = np.mean([svr_prediction, tree_prediction, lr_prediction])

            # Determine the best buy and sell dates based on Linear Regression prediction
            if lr_prediction[0] < best_buy_price:
                best_buy_price = lr_prediction[0]
                best_buy_date = future_date_str

            if lr_prediction[0] > best_sell_price:
                best_sell_price = lr_prediction[0]
                best_sell_date = future_date_str

        # Create Plotly graph for future predictions
        fig1 = go.Figure()

        # Line for SVR
        fig1.add_trace(go.Scatter(x=[pred['Date'] for pred in predictions],
                                   y=[pred['SVR Prediction'] for pred in predictions],
                                   mode='lines', name='SVR Prediction', visible='legendonly'))

        # Line for Tree
        fig1.add_trace(go.Scatter(x=[pred['Date'] for pred in predictions],
                                   y=[pred['Tree Prediction'] for pred in predictions],
                                   mode='lines', name='Tree Prediction', visible='legendonly'))

        # Line for Linear Regression
        fig1.add_trace(go.Scatter(x=[pred['Date'] for pred in predictions],
                                   y=[pred['LR Prediction'] for pred in predictions],
                                   mode='lines', name='Linear Regression Prediction'))

        fig1.update_layout(title='Predicted Prices for Future Days',
                           xaxis_title='Date',
                           yaxis_title='Price')

        # Convert Plotly graph to JSON format
        plot1_json = pio.to_json(fig1)

        # Render to index.html with all the data
        return render_template('index.html', data={
            'plot1': plot1_json,
            'best_buy_date': best_buy_date,
            'best_buy_price': best_buy_price,
            'best_sell_date': best_sell_date,
            'best_sell_price': best_sell_price
        })

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)