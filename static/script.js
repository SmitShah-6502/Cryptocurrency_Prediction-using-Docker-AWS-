function predictPrice() {
    let symbol = document.getElementById("cryptoSymbol").value.toUpperCase();

    if (symbol === "") {
        alert("Please enter a valid cryptocurrency symbol!");
        document.getElementById("result").innerHTML = "";
        document.getElementById("recommendation").innerHTML = "";
        document.getElementById("price_change").innerHTML = "";
        document.getElementById("volatility").innerHTML = "";
        document.getElementById("trend").innerHTML = "";
        document.getElementById("sentiment").innerHTML = "";
        document.getElementById("portfolio_return").innerHTML = "";
        document.getElementById("alerts").innerHTML = "";
        return;
    }

    fetch(`/predict?symbol=${symbol}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("API Response:", data); // Log response for debugging
            if (data.predicted_price) {
                document.getElementById("result").innerHTML = 
                    `Predicted Price: <b>$${parseFloat(data.predicted_price).toFixed(2)}</b>`;
                document.getElementById("recommendation").innerHTML = 
                    `Recommendation: <b>${data.recommendation}</b>`;
                document.getElementById("price_change").innerHTML = 
                    `Price Change: <b>${parseFloat(data.price_change_percent).toFixed(2)}%</b>`;
                document.getElementById("volatility").innerHTML = 
                    `Volatility (14-period): <b>${parseFloat(data.volatility).toFixed(2)}%</b>`;
                document.getElementById("trend").innerHTML = 
                    `Market Trend: <b>${data.trend}</b>`;
                document.getElementById("sentiment").innerHTML = 
                    `Market Sentiment: <b>${data.sentiment}</b>`;
                document.getElementById("portfolio_return").innerHTML = 
                    `Simulated Portfolio Return: <b>${parseFloat(data.portfolio_return).toFixed(2)}%</b>`;
                document.getElementById("alerts").innerHTML = 
                    `Buy Alert: <b>$${parseFloat(data.buy_alert).toFixed(2)}</b> | Sell Alert: <b>$${parseFloat(data.sell_alert).toFixed(2)}</b>`;
            } else {
                document.getElementById("result").innerHTML = 
                    `<span style="color:red;">Error: ${data.error}</span>`;
                document.getElementById("recommendation").innerHTML = "";
                document.getElementById("price_change").innerHTML = "";
                document.getElementById("volatility").innerHTML = "";
                document.getElementById("trend").innerHTML = "";
                document.getElementById("sentiment").innerHTML = "";
                document.getElementById("portfolio_return").innerHTML = "";
                document.getElementById("alerts").innerHTML = "";
            }
        })
        .catch(error => {
            console.error("Fetch Error:", error); // Log error for debugging
            document.getElementById("result").innerHTML = 
                `<span style="color:red;">API Error: ${error.message}</span>`;
            document.getElementById("recommendation").innerHTML = "";
            document.getElementById("price_change").innerHTML = "";
            document.getElementById("volatility").innerHTML = "";
            document.getElementById("trend").innerHTML = "";
            document.getElementById("sentiment").innerHTML = "";
            document.getElementById("portfolio_return").innerHTML = "";
            document.getElementById("alerts").innerHTML = "";
        });
}