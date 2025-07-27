# 📈 Portfolio Manager Pro

A professional Streamlit application for managing stock portfolios with real-time performance tracking and analytics.

## 🌟 Features

### 📊 **Portfolio Management**
- ✅ Add stocks with purchase details (ticker, quantity, date, price)
- ✅ Remove individual positions
- ✅ Auto-fetch stock names from tickers
- ✅ CSV-based backend storage

### 📈 **Performance Analytics**
- ✅ Real-time portfolio performance tracking
- ✅ Interactive line charts showing stock returns over time
- ✅ Portfolio weight distribution pie chart
- ✅ Detailed holdings table with P&L calculations

### 💰 **Financial Metrics**
- ✅ Total invested amount
- ✅ Current portfolio value
- ✅ Total profit/loss ($ and %)
- ✅ Individual stock performance
- ✅ Return percentages

### 🎨 **Professional UI**
- ✅ Modern, responsive design
- ✅ Interactive charts with Plotly
- ✅ Color-coded profit/loss indicators
- ✅ Professional card-based layout

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install streamlit yfinance pandas plotly numpy
```

### 2. Run the Application
```bash
streamlit run portfolio_app.py
```

### 3. Add Your First Stock
1. Use the sidebar form to add stocks
2. Enter ticker symbol (e.g., AAPL, MSFT, TSLA)
3. Fill in quantity, buy date, and purchase price
4. Click "Add to Portfolio"

### 4. Demo Data (Optional)
```bash
# Create sample portfolio for testing
python create_demo_portfolio.py
```

## 📁 Data Storage

The app uses a simple CSV file (`portfolio.csv`) to store your portfolio data:

```csv
stock_name,ticker,quantity,buy_date,buy_price
Apple Inc.,AAPL,10,2024-01-15,185.50
Microsoft Corporation,MSFT,8,2024-02-10,412.30
```

## 🎯 Usage Examples

### Adding Stocks
1. **Enter Ticker**: Type the stock symbol (e.g., "AAPL")
2. **Stock Name**: Auto-filled or enter manually
3. **Quantity**: Number of shares purchased
4. **Buy Date**: Date of purchase
5. **Buy Price**: Price per share at purchase

### Viewing Performance
- **Performance Chart**: Shows percentage returns over selected time period
- **Weight Distribution**: Pie chart showing portfolio allocation
- **Detailed Holdings**: Complete table with current values and P&L

### Managing Portfolio
- **Remove Stocks**: Click the ❌ button next to any position
- **Export Data**: Download your portfolio as CSV
- **Real-time Updates**: Prices refresh automatically

## 📊 Charts & Analytics

### 📈 Performance Line Chart
- Compare multiple stocks' performance
- Adjustable time periods (1M, 3M, 6M, 1Y, 2Y)
- Percentage-based returns for easy comparison
- Interactive hover tooltips

### 🥧 Portfolio Weight Pie Chart
- Visual representation of portfolio allocation
- Shows percentage and dollar value for each stock
- Color-coded segments
- Interactive legend

### 📋 Holdings Table
Displays for each position:
- Stock name and ticker
- Quantity and buy date
- Purchase price vs current price
- Invested amount vs current value
- Profit/loss in dollars and percentage

## 🎨 Design Features

### Professional Styling
- **Gradient Cards**: Modern card-based layout
- **Color Coding**: Green for profits, red for losses
- **Responsive Design**: Works on desktop and mobile
- **Interactive Elements**: Hover effects and animations

### User Experience
- **Auto-completion**: Stock names fetched automatically
- **Form Validation**: Prevents invalid data entry
- **Real-time Updates**: Live price fetching
- **Error Handling**: Graceful error messages

## 🔧 Technical Details

### Dependencies
- `streamlit`: Web app framework
- `yfinance`: Stock data API
- `pandas`: Data manipulation
- `plotly`: Interactive charts
- `numpy`: Numerical operations

### Data Flow
1. User adds stock via sidebar form
2. Data saved to `portfolio.csv`
3. Current prices fetched from Yahoo Finance
4. Performance calculations performed
5. Charts and metrics updated in real-time

### Error Handling
- Invalid tickers are handled gracefully
- Network errors don't crash the app
- Missing data shows appropriate messages
- CSV corruption is detected and fixed

## 🚀 Advanced Features

### Portfolio Metrics
- **Total Return**: Overall portfolio performance
- **Individual Performance**: Per-stock returns
- **Weight Analysis**: Portfolio diversification
- **Time-based Analysis**: Performance over different periods

### Data Export
- **CSV Download**: Export complete portfolio
- **Timestamped Files**: Automatic date naming
- **Format Preservation**: Maintains data integrity

### Real-time Updates
- **Live Prices**: Fetched from Yahoo Finance
- **Auto-refresh**: Periodic data updates
- **Cache Management**: Efficient data loading

## 📱 Screenshots

The app features:
- **Dashboard**: Key metrics at a glance
- **Charts**: Interactive performance visualization
- **Holdings**: Detailed position management
- **Forms**: Clean, intuitive data entry

## 🔮 Future Enhancements

Potential additions:
- 📊 Dividend tracking
- 📈 Benchmark comparison (S&P 500)
- 🔔 Price alerts
- 📧 Email reports
- 🌐 Multi-currency support
- 📱 Mobile optimization
- 🔐 User authentication
- ☁️ Cloud storage integration

## 📞 Support

For issues or questions:
1. Check the CSV file format
2. Verify ticker symbols are correct
3. Ensure internet connection for price data
4. Review error messages in the app

## 📝 Notes

- Stock prices are fetched from Yahoo Finance
- Historical data may have delays
- Weekend/holiday data might be stale
- Free tier limitations may apply to data sources

---

**Built with ❤️ using Streamlit and Python**
