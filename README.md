📈 Cryptocurrency Price Prediction & Trading Recommendation System
A web-based application that predicts cryptocurrency prices, provides trading recommendations, and offers market insights using machine learning and real-time data from the Binance API. Despite being titled "Mutual Fund Portfolio Recommendation Solution - 2025," the project focuses on cryptocurrencies but is adaptable for mutual fund applications.

Features
Price Prediction: Forecasts the next hourly closing price using a pre-trained LSTM neural network.
Trading Recommendations: Generates Buy, Sell, or Hold signals based on predicted price changes and momentum indicators (SMAs).
Market Analysis:
Calculates 14-period annualized volatility to assess risk.
Detects bullish, bearish, or neutral market trends using SMA crossovers.
Simulates sentiment analysis from social media posts (placeholder for X API).


Portfolio Simulation: Estimates returns for a $1,000 investment based on predicted prices.
Price Alerts: Sets dynamic buy/sell thresholds based on volatility.
Interactive Web Interface: A responsive frontend built with HTML, CSS, and JavaScript for user input and result display.

Tech Stack
Frontend: HTML, CSS, JavaScript
Backend: Flask, Python, Pandas, NumPy
Machine Learning: Keras/TensorFlow (LSTM model), Scikit-learn (MinMaxScaler)
APIs: Binance API (real-time data), TextBlob (sentiment analysis)
Deployment: Docker, AWS EC2
Datasets: Real-time cryptocurrency data from Binance API

Project Structure
├── app.py                    # Flask backend with API routes
├── train_model.py            # Script to train and save the LSTM model
├── templates/
│   └── index.html            # Frontend UI
├── static/
│   ├── script.js             # Frontend JavaScript logic
│   ├── style.css             # Frontend styling
│   └── back.jpg              # Background image
├── crypto_model.h5           # Pre-trained LSTM model
├── requirements.txt          # Python dependencies
└── README.md                 # Project description

Installation & Run Locally
1. Clone the Repository
git clone https://github.com/your-username/crypto-price-prediction.git
cd crypto-price-prediction

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Ensure Model File
Ensure crypto_model.h5 is in the project root. If missing, run train_model.py to generate it (requires Binance API access).

5. Run the Flask App
python app.py
Visit http://127.0.0.1:8000 in your browser to access the application.

How It Works
Price Prediction: Fetches 100 hourly data points from the Binance API, preprocesses the last 60 closing prices, and uses the LSTM model to predict the next closing price.
Trading Recommendations: Combines predicted price changes (>2% for Buy, <-2% for Sell) with 5-period and 20-period SMAs for momentum-based recommendations.
Market Analysis: Computes volatility, market trends, and simulated sentiment to provide comprehensive insights.
Portfolio & Alerts: Simulates investment returns and sets volatility-based price thresholds.
Frontend: Users input a cryptocurrency symbol (e.g., BTCUSDT) and view results in a clean, responsive interface.

Deployment
Docker: The app is containerized for consistent environments.
AWS EC2: Deployed on an EC2 instance with port 8000 exposed for public access.
To deploy:
Build the Docker image: docker build -t crypto-prediction .
Run the container: docker run -p 8000:8000 crypto-prediction
Configure EC2 security groups to allow traffic on port 8000.

Future Enhancements
Integrate real-time X API for live sentiment analysis.
Enhance the LSTM model with additional features (e.g., volume, RSI).
Add interactive price trend charts using Chart.js.
Implement periodic model retraining with fresh data.
Adapt for mutual funds using financial APIs like Morningstar and risk-adjusted metrics.

Credits
Binance API: Real-time cryptocurrency data.
Keras/TensorFlow: LSTM model implementation.
TextBlob: Sentiment analysis simulation.
MovieLens Dataset: Inspiration for recommendation logic (not used directly).

License
This project is open-source under the MIT License.
