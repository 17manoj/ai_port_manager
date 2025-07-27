"""
Professional Stock Portfolio Management App
A comprehensive Streamlit application for managing stock portfolios with CSV backend
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ai_agent import *
import datetime

import os
from typing import Dict, List, Optional
import numpy as np
import time

# Global variable to track API calls for rate limiting
_last_api_call = 0
_min_api_interval = 0.5  # Minimum 500ms between API calls

def rate_limited_api_call(func, *args, **kwargs):
    """Rate limited wrapper for Yahoo Finance API calls"""
    global _last_api_call
    
    # Ensure minimum time between API calls
    current_time = time.time()
    time_since_last = current_time - _last_api_call
    
    if time_since_last < _min_api_interval:
        sleep_time = _min_api_interval - time_since_last
        time.sleep(sleep_time)
    
    try:
        result = func(*args, **kwargs)
        _last_api_call = time.time()
        return result
    except Exception as e:
        st.warning(f"API call failed: {str(e)}")
        _last_api_call = time.time()
        return None

# Page configuration
st.set_page_config(
    page_title="📈 AI - Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(145deg, #d4edda, #ffffff);
        border-left-color: #28a745;
    }
    .warning-card {
        background: linear-gradient(145deg, #fff3cd, #ffffff);
        border-left-color: #ffc107;
    }
    .danger-card {
        background: linear-gradient(145deg, #f8d7da, #ffffff);
        border-left-color: #dc3545;
    }
    .stock-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #17a2b8);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class PortfolioManager:
    """Main portfolio management class"""
    
    def __init__(self, csv_file: str = "portfolio.csv"):
        self.csv_file = csv_file
        self.ensure_csv_exists()
    
    def ensure_csv_exists(self):
        """Create CSV file if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=['stock_name', 'ticker', 'quantity', 'buy_date', 'buy_price'])
            df.to_csv(self.csv_file, index=False)
    
    def load_portfolio(self) -> pd.DataFrame:
        """Load portfolio from CSV"""
        try:
            df = pd.read_csv(self.csv_file)
            if df.empty:
                return pd.DataFrame(columns=['stock_name', 'ticker', 'quantity', 'buy_date', 'buy_price'])
            return df
        except Exception as e:
            st.error(f"Error loading portfolio: {str(e)}")
            return pd.DataFrame(columns=['stock_name', 'ticker', 'quantity', 'buy_date', 'buy_price'])
    
    def save_portfolio(self, df: pd.DataFrame):
        """Save portfolio to CSV"""
        try:
            df.to_csv(self.csv_file, index=False)
            return True
        except Exception as e:
            st.error(f"Error saving portfolio: {str(e)}")
            return False
    
    def add_stock(self, stock_name: str, ticker: str, quantity: float, buy_date: datetime.date, buy_price: float):
        """Add a new stock to portfolio"""
        df = self.load_portfolio()
        new_row = {
            'stock_name': stock_name,
            'ticker': ticker.upper(),
            'quantity': quantity,
            'buy_date': buy_date.strftime('%Y-%m-%d'),
            'buy_price': buy_price
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        return self.save_portfolio(df)
    
    def remove_stock(self, index: int):
        """Remove a stock from portfolio"""
        df = self.load_portfolio()
        if 0 <= index < len(df):
            df = df.drop(index).reset_index(drop=True)
            return self.save_portfolio(df)
        return False
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices for list of tickers with rate limiting"""
        prices = {}
        for ticker in tickers:
            try:
                def get_price():
                    stock = yf.Ticker(ticker)
                    
                    info = stock.info
                    
                  
                    return info.get('currentPrice', info.get('regularMarketPrice', 0))
                
                price = rate_limited_api_call(get_price)
                prices[ticker] = price if price is not None else 0
                
                # Show progress for multiple stocks
                if len(tickers) > 1:
                    st.write(f"✅ Fetched price for {ticker}")
                    
            except Exception as e:
                st.warning(f"Failed to get price for {ticker}: {str(e)}")
                prices[ticker] = 0
        return prices
    
    def get_stock_info(self, ticker: str) -> Dict:
        """Get detailed stock information with rate limiting"""
        try:
            def get_info():
                stock = yf.Ticker(ticker)
                return stock.info
            
            info = rate_limited_api_call(get_info)
            
            if info is None:
                return {'name': ticker, 'currentPrice': 0}
                
            return {
                'name': info.get('longName', ticker),
                'currentPrice': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'marketCap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'peRatio': info.get('trailingPE', 0),
                'dividendYield': info.get('dividendYield', 0)
            }
        except Exception as e:
            st.warning(f"Failed to get stock info for {ticker}: {str(e)}")
            return {'name': ticker, 'currentPrice': 0}
    
    def get_historical_price(self, ticker: str, date: datetime.date) -> float:
        """Get historical close price for a specific date"""
        try:
            def get_hist_price():
                stock = yf.Ticker(ticker)
                # Get data from the date and a few days before/after to handle weekends/holidays
                start_date = date - datetime.timedelta(days=5)
                end_date = date + datetime.timedelta(days=5)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Try to get exact date first
                    target_date = pd.Timestamp(date).tz_localize(hist.index.tz)
                    if target_date in hist.index:
                        return float(hist.loc[target_date]['Close'])
                    else:
                        # Get closest available date
                        closest_idx = hist.index.get_indexer([target_date], method='nearest')[0]
                        return float(hist.iloc[closest_idx]['Close'])
                return 0
            
            price = rate_limited_api_call(get_hist_price)
            return price if price is not None else 0
            
        except Exception as e:
            st.warning(f"Failed to get historical price for {ticker} on {date}: {str(e)}")
            return 0

@st.cache_data
def get_stock_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Get stock price history with caching and rate limiting"""
    try:
        def get_history():
            stock = yf.Ticker(ticker)
            return stock.history(period=period)
        
        hist = rate_limited_api_call(get_history)
        
        if hist is None or hist.empty:
            st.warning(f"No historical data available for {ticker}")
            return pd.DataFrame()
            
        return hist
        
    except Exception as e:
        st.warning(f"Failed to get history for {ticker}: {str(e)}")
        return pd.DataFrame()

def create_performance_chart(portfolio_df: pd.DataFrame, period: str = "6mo"):
    """Create performance line chart for all stocks"""
    if portfolio_df.empty:
        return None
    
    tickers = portfolio_df['ticker'].unique()
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    valid_tickers = []
    
    with st.spinner("Fetching stock data..."):
        for i, ticker in enumerate(tickers):
            hist = get_stock_history(ticker, period)
            if not hist.empty and len(hist) > 1:
                # Calculate percentage change from first day
                first_price = hist['Close'].iloc[0]
                if first_price > 0:  # Avoid division by zero
                    pct_change = ((hist['Close'] - first_price) / first_price) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=pct_change,
                        mode='lines',
                        name=ticker,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f'<b>{ticker}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Return: %{y:.2f}%<extra></extra>'
                    ))
                    valid_tickers.append(ticker)
                else:
                    st.warning(f"Invalid price data for {ticker}")
            else:
                st.warning(f"No sufficient data available for {ticker} in {period} period")
    
    if not valid_tickers:
        st.error("No valid stock data available for the selected period. Please try a different time period or check your stock symbols.")
        return None
    
    fig.update_layout(
        title=dict(
            text=f"📈 Portfolio Performance - {period.upper()}",
            font=dict(size=24, color='#1f77b4'),
            x=0.5
        ),
        xaxis_title="Date",
        yaxis_title="Return (%)",
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_portfolio_pie_chart(portfolio_df: pd.DataFrame, current_prices: Dict[str, float]):
    """Create pie chart showing portfolio weight distribution"""
    if portfolio_df.empty:
        return None
    
    # Calculate current values
    portfolio_df = portfolio_df.copy()
    portfolio_df['current_price'] = portfolio_df['ticker'].map(current_prices)
    portfolio_df['current_value'] = portfolio_df['quantity'] * portfolio_df['current_price']
    
    # Group by ticker and sum values
    portfolio_summary = portfolio_df.groupby('ticker').agg({
        'current_value': 'sum',
        'stock_name': 'first'
    }).reset_index()
    
    # Calculate weights
    total_value = portfolio_summary['current_value'].sum()
    portfolio_summary['weight'] = (portfolio_summary['current_value'] / total_value) * 100
    
    fig = go.Figure(data=[go.Pie(
        labels=portfolio_summary['ticker'],
        values=portfolio_summary['weight'],
        hole=0.4,
        texttemplate='<b>%{label}</b><br>%{percent}<br>$%{value:,.0f}',
        textposition="auto",
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='#FFFFFF', width=2)
        ),
        hovertemplate='<b>%{label}</b><br>' +
                      'Weight: %{percent}<br>' +
                      'Value: $%{value:,.2f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text="🥧 Portfolio Weight Distribution",
            font=dict(size=24, color='#1f77b4'),
            x=0.5
        ),
        font=dict(size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def display_portfolio_metrics(portfolio_df: pd.DataFrame, current_prices: Dict[str, float]):
    """Display key portfolio metrics"""
    if portfolio_df.empty:
        st.info("📊 Add stocks to see portfolio metrics")
        return
    
    # Calculate metrics
    portfolio_df = portfolio_df.copy()
    portfolio_df['current_price'] = portfolio_df['ticker'].map(current_prices)
    portfolio_df['current_value'] = portfolio_df['quantity'] * portfolio_df['current_price']
    portfolio_df['invested_value'] = portfolio_df['quantity'] * portfolio_df['buy_price']
    portfolio_df['profit_loss'] = portfolio_df['current_value'] - portfolio_df['invested_value']
    portfolio_df['profit_loss_pct'] = (portfolio_df['profit_loss'] / portfolio_df['invested_value']) * 100
    
    total_invested = portfolio_df['invested_value'].sum()
    total_current = portfolio_df['current_value'].sum()
    total_pl = total_current - total_invested
    total_pl_pct = (total_pl / total_invested) * 100 if total_invested > 0 else 0
    
    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        card_class = "metric-card"
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="color: #1f77b4; margin: 0;">💰 Total Invested</h3>
            <h2 style="margin: 0.5rem 0;">${total_invested:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        card_class = "metric-card"
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="color: #1f77b4; margin: 0;">📈 Current Value</h3>
            <h2 style="margin: 0.5rem 0;">${total_current:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pl_color = "#28a745" if total_pl >= 0 else "#dc3545"
        card_class = "metric-card success-card" if total_pl >= 0 else "metric-card danger-card"
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="color: {pl_color}; margin: 0;">💹 Total P&L</h3>
            <h2 style="margin: 0.5rem 0; color: {pl_color};">${total_pl:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        pl_color = "#28a745" if total_pl_pct >= 0 else "#dc3545"
        card_class = "metric-card success-card" if total_pl_pct >= 0 else "metric-card danger-card"
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="color: {pl_color}; margin: 0;">📊 Return %</h3>
            <h2 style="margin: 0.5rem 0; color: {pl_color};">{total_pl_pct:+.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

def display_detailed_holdings(portfolio_df: pd.DataFrame, current_prices: Dict[str, float]):
    """Display detailed holdings table"""
    if portfolio_df.empty:
        return
    
    st.subheader("📋 Detailed Holdings")
    
    # Prepare detailed data
    detailed_df = portfolio_df.copy()
    detailed_df['current_price'] = detailed_df['ticker'].map(current_prices)
    detailed_df['current_value'] = detailed_df['quantity'] * detailed_df['current_price']
    detailed_df['invested_value'] = detailed_df['quantity'] * detailed_df['buy_price']
    detailed_df['profit_loss'] = detailed_df['current_value'] - detailed_df['invested_value']
    detailed_df['return_pct'] = (detailed_df['profit_loss'] / detailed_df['invested_value']) * 100
    
    # Format for display
    display_df = detailed_df.copy()
    display_df['buy_price'] = display_df['buy_price'].apply(lambda x: f"${x:.2f}")
    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
    display_df['invested_value'] = display_df['invested_value'].apply(lambda x: f"${x:,.2f}")
    display_df['current_value'] = display_df['current_value'].apply(lambda x: f"${x:,.2f}")
    display_df['profit_loss'] = display_df['profit_loss'].apply(lambda x: f"${x:+,.2f}")
    display_df['return_pct'] = display_df['return_pct'].apply(lambda x: f"{x:+.2f}%")
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        'stock_name': 'Stock Name',
        'ticker': 'Ticker',
        'quantity': 'Qty',
        'buy_date': 'Buy Date',
        'buy_price': 'Buy Price',
        'current_price': 'Current Price',
        'invested_value': 'Invested',
        'current_value': 'Current Value',
        'profit_loss': 'P&L',
        'return_pct': 'Return %'
    })
    
    st.dataframe(
        display_df[['Stock Name', 'Ticker', 'Qty', 'Buy Date', 'Buy Price', 
                   'Current Price', 'Invested', 'Current Value', 'P&L', 'Return %']],
        use_container_width=True
    )

def main():
    """Main application function"""
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager()
    
    # Header
    st.markdown('<h1 class="main-header">📈  AI - Portfolio Manager </h1>', unsafe_allow_html=True)
    
    # Sidebar for adding stocks
    with st.sidebar:
        st.header("🎯 Add New Stock")
        
        with st.form("add_stock_form"):
            ticker = st.text_input("Stock Ticker", placeholder="e.g., AAPL, GOOGL", help="Enter the stock symbol")
            
            # Auto-fetch stock name if ticker is provided
            stock_name = st.text_input("Stock Name", placeholder="e.g., Apple Inc.")
            
            if ticker and not stock_name:
                try:
                    info = portfolio_manager.get_stock_info(ticker.upper())
                    stock_name = info.get('name', ticker.upper())
                    st.info(f"Auto-detected: {stock_name}")
                except:
                    pass
            
            quantity = st.number_input("Quantity", min_value=0.0, step=0.1, format="%.2f")
            buy_date = st.date_input("Buy Date", value=datetime.date.today())
            
            # Auto-fetch buy price based on ticker and date
            auto_fetch_price = st.checkbox("Auto-fetch historical price", value=True)
            
            if auto_fetch_price and ticker and buy_date:
                try:
                    historical_price = portfolio_manager.get_historical_price(ticker.upper(), buy_date)
                    if historical_price > 0:
                        st.info(f"Historical close price on {buy_date}: ${historical_price:.2f}")
                        buy_price = st.number_input("Buy Price ($)", value=historical_price, min_value=0.0, step=0.01, format="%.2f")
                    else:
                        buy_price = st.number_input("Buy Price ($)", min_value=0.0, step=0.01, format="%.2f")
                        if ticker:
                            st.warning(f"Could not fetch historical price for {ticker.upper()} on {buy_date}")
                except:
                    buy_price = st.number_input("Buy Price ($)", min_value=0.0, step=0.01, format="%.2f")
            else:
                buy_price = st.number_input("Buy Price ($)", min_value=0.0, step=0.01, format="%.2f")
            
            submitted = st.form_submit_button("🎉 Add to Portfolio", type="primary")
            
            if submitted:
                if ticker and stock_name and quantity > 0 and buy_price > 0:
                    success = portfolio_manager.add_stock(
                        stock_name, ticker.upper(), quantity, buy_date, buy_price
                    )
                    if success:
                        st.success(f"✅ Added {ticker.upper()} to portfolio!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add stock")
                else:
                    st.error("❌ Please fill all fields correctly")
        
        # Portfolio summary in sidebar
        portfolio_df = portfolio_manager.load_portfolio()
        if not portfolio_df.empty:
            st.subheader("📊 Quick Stats")
            unique_stocks = portfolio_df['ticker'].nunique()
            total_positions = len(portfolio_df)
            st.metric("Unique Stocks", unique_stocks)
            st.metric("Total Positions", total_positions)
    
    # Main content area
    portfolio_df = portfolio_manager.load_portfolio()
    
    if portfolio_df.empty:
        st.info("🚀 Welcome! Add your first stock using the sidebar to get started.")
        st.markdown("""
        ### Getting Started:
        1. **Enter Stock Ticker** (e.g., AAPL, TSLA, GOOGL)
        2. **Add Stock Details** (quantity, buy date, price)
        3. **View Performance** with interactive charts
        4. **Monitor Portfolio** with real-time metrics
        """)
    else:
        # Get current prices
        tickers = portfolio_df['ticker'].unique().tolist()
        with st.spinner("🔄 Fetching current prices..."):
            current_prices = portfolio_manager.get_current_prices(tickers)
        
        # Display portfolio metrics
        display_portfolio_metrics(portfolio_df, current_prices)
        
        # Charts section
        st.subheader("📊 Portfolio Analytics")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Performance chart
            period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
            perf_chart = create_performance_chart(portfolio_df, period)
            if perf_chart:
                st.plotly_chart(perf_chart, use_container_width=True)
        
        with chart_col2:
            # Portfolio pie chart
            pie_chart = create_portfolio_pie_chart(portfolio_df, current_prices)
            if pie_chart:
                st.plotly_chart(pie_chart, use_container_width=True)
        
        # Detailed holdings
        display_detailed_holdings(portfolio_df, current_prices)
        
        # Stock removal section
        st.subheader("🗑️ Manage Holdings")
        
        if len(portfolio_df) > 0:
            # Group by ticker for easier management
            grouped_df = portfolio_df.groupby(['ticker', 'stock_name']).agg({
                'quantity': 'sum',
                'buy_date': 'min',
                'buy_price': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Select positions to remove:**")
                
                for idx, row in portfolio_df.iterrows():
                    col_a, col_b, col_c = st.columns([3, 2, 1])
                    
                    with col_a:
                        st.write(f"**{row['stock_name']} ({row['ticker']})**")
                        st.write(f"Qty: {row['quantity']} | Date: {row['buy_date']} | Price: ${row['buy_price']:.2f}")
                    
                    with col_b:
                        current_price = current_prices.get(row['ticker'], 0)
                        current_value = row['quantity'] * current_price
                        invested_value = row['quantity'] * row['buy_price']
                        pl = current_value - invested_value
                        pl_color = "green" if pl >= 0 else "red"
                        st.write(f"Current: ${current_price:.2f}")
                        st.markdown(f"P&L: <span style='color: {pl_color}'>${pl:+.2f}</span>", unsafe_allow_html=True)
                    
                    with col_c:
                        if st.button(f"❌", key=f"remove_{idx}", help="Remove this position"):
                            if portfolio_manager.remove_stock(idx):
                                st.success("✅ Stock removed!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to remove stock")
        
        # Export option
        st.subheader("🤖 AI Portfolio Analysis")
        
        col1, col2 = st.columns([1, 1])

        with col1:
            openai_key = st.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API key")
            tavily_key = st.text_input("Tavily API Key", type="password", placeholder="Enter your Tavily API key")
            if st.button("🔍 AI Analysis", key="ai_analysis_btn", type="primary", help="Get AI-powered insights on your portfolio"):
                if openai_key and tavily_key:
                    with st.spinner("🧠 AI is analyzing your portfolio..."):
                        time.sleep(125)
                        agent = StockAnalysisAgent(
                                    openai_api_key=openai_key,
                                    tavily_api_key=tavily_key
                            )
                        stock_data = pd.read_csv('portfolio.csv')
                        result = agent.analyze_stocks(stock_data, current_date=datetime.datetime.now())
                        st.success("✅ AI analysis complete!")
                        st.write(result['analysis'])
                else:
                    st.warning("Sharing my OpenAI key sounded like a good idea—until I realized my account was running more queries than Google on a Monday morning. Please… save my bank account. Get your own key!")

if __name__ == "__main__":
    main()
