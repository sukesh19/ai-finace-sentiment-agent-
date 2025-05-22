document.addEventListener('DOMContentLoaded', function () {
    const stockSelect = document.getElementById('stock-select');
    const analysisArea = document.getElementById('analysis-area');
    const analysisContent = document.getElementById('analysis-content');
    const predictionArea = document.getElementById('prediction-area');
    const predictedStockName = document.getElementById('predicted-stock-name');
    const predictedValue = document.getElementById('predicted-value');
    const loadingMessage = document.getElementById('loading-message');
    const errorMessage = document.getElementById('error-message');

    // Fetch stock names and populate dropdown
    fetch('/api/stocks')
        .then(response => response.json())
        .then(stocks => {
            stockSelect.innerHTML = '<option value="" disabled selected>-- Choose a stock --</option>';
            stocks.forEach(stock => {
                const option = document.createElement('option');
                option.value = stock;
                option.textContent = stock;
                stockSelect.appendChild(option);
            });
        })
        .catch(() => {
            alert('Failed to load stock list.');
        });

    stockSelect.addEventListener('change', function () {
        const stock = stockSelect.value;
        if (!stock) return;

        // Fetch analysis data
        fetch(`/api/analyze/${stock}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    analysisArea.style.display = 'none';
                    alert(data.error);
                    return;
                }

                // Display analysis
                analysisArea.style.display = 'block';
                analysisContent.innerHTML = `
                    <p><strong>Mean Close Price:</strong> ₹${data.mean_close.toFixed(2)}</p>
                    <p><strong>Median Close Price:</strong> ₹${data.median_close.toFixed(2)}</p>
                    <p><strong>Max Close Price:</strong> ₹${data.max_close.toFixed(2)}</p>
                    <p><strong>Min Close Price:</strong> ₹${data.min_close.toFixed(2)}</p>
                    <p><strong>Total Volume:</strong> ${data.total_volume}</p>
                `;
            })
            .catch(() => {
                alert('Failed to fetch analysis data.');
            });

        // Show loading message
        loadingMessage.style.display = 'block';
        predictionArea.style.display = 'none';
        errorMessage.style.display = 'none';

        // Fetch prediction data
        fetch(`/api/predict/${stock}`)
            .then(response => response.json())
            .then(data => {
                loadingMessage.style.display = 'none';
                if (data.error) {
                    errorMessage.style.display = 'block';
                    errorMessage.textContent = data.error;
                    return;
                }
                predictionArea.style.display = 'block';
                predictedStockName.textContent = stock;
                predictedValue.textContent = `₹${Number(data.predictedPrice).toFixed(2)}`;
                const accuracy = data.accuracy || "N/A";
                document.getElementById('prediction-accuracy').textContent = `Accuracy: ${accuracy}`;
            })
            .catch(() => {
                loadingMessage.style.display = 'none';
                errorMessage.style.display = 'block';
                errorMessage.textContent = 'Prediction failed.';
            });
    });
});