import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="SmartPortfolio Risk Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (for professional styling)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e86de);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class PortfolioRiskAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.risk_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_stock_data(_self, symbols, period="2y"):
        """Fetch stock data from Yahoo Finance with caching"""
        try:
            data = {}
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, symbol in enumerate(symbols):
                status_text.text(f'Fetching data for {symbol}...')
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period=period)
                    if not hist.empty:
                        data[symbol] = hist
                    else:
                        st.warning(f"No data found for {symbol}")
                except Exception as symbol_error:
                    st.warning(f"Failed to fetch {symbol}: {str(symbol_error)}")
                    continue

                progress_bar.progress((i + 1) / len(symbols))

            status_text.text('Data fetching completed!')
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            if not data:
                st.error("No stock data could be fetched. Please check your internet connection and try again.")
                return {}

            return data
        except Exception as e:
            st.error(f"Critical error fetching data: {str(e)}")
            return {}
    
    def calculate_returns(self, price_data):
        """Calculate daily returns"""
        returns = {}
        for symbol, data in price_data.items():
            returns[symbol] = data['Close'].pct_change().dropna()
        return pd.DataFrame(returns)
    
    def calculate_risk_metrics(self, returns, weights):
        """Calculate various risk metrics"""
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Basic metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Maximum Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Beta calculation (using SPY as market proxy)
        try:
            spy_data = yf.Ticker('SPY').history(period='2y')
            spy_returns = spy_data['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(spy_returns.index)
            portfolio_aligned = portfolio_returns.loc[common_dates]
            spy_aligned = spy_returns.loc[common_dates]
            
            covariance = np.cov(portfolio_aligned, spy_aligned)[0][1]
            market_variance = spy_aligned.var()
            beta = covariance / market_variance if market_variance != 0 else 1
        except:
            beta = 1.0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'beta': beta,
            'portfolio_returns': portfolio_returns
        }
    
    def train_risk_model(self, returns_data):
        """Train ML model for risk prediction"""
        try:
            # Prepare features
            features = []
            targets = []
            
            for symbol in returns_data.columns:
                symbol_returns = returns_data[symbol].dropna()
                if len(symbol_returns) < 30:
                    continue
                    
                for i in range(20, len(symbol_returns)):
                    # Features: rolling statistics
                    window_data = symbol_returns[i-20:i]
                    feature_row = [
                        window_data.mean(),
                        window_data.std(),
                        window_data.skew(),
                        window_data.kurtosis(),
                        window_data.min(),
                        window_data.max(),
                        np.percentile(window_data, 25),
                        np.percentile(window_data, 75)
                    ]
                    features.append(feature_row)
                    # Target: future volatility
                    future_window = symbol_returns[i:i+5]
                    if len(future_window) == 5:
                        targets.append(future_window.std())
            
            if len(features) > 100:
                X = np.array(features)
                y = np.array(targets)
                
                # Handle NaN values
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[mask]
                y = y[mask]
                
                if len(X) > 50:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Scale features
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    
                    # Train model
                    self.risk_model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = self.risk_model.predict(X_test_scaled)
                    r2 = r2_score(y_test, y_pred)
                    
                    return True, r2
            
            return False, 0
        except Exception as e:
            return False, 0
    
    def predict_risk_score(self, returns_data):
        """Predict risk score using ML model"""
        try:
            risk_scores = {}
            for symbol in returns_data.columns:
                symbol_returns = returns_data[symbol].dropna()
                if len(symbol_returns) >= 20:
                    recent_data = symbol_returns[-20:]
                    features = [[
                        recent_data.mean(),
                        recent_data.std(),
                        recent_data.skew(),
                        recent_data.kurtosis(),
                        recent_data.min(),
                        recent_data.max(),
                        np.percentile(recent_data, 25),
                        np.percentile(recent_data, 75)
                    ]]
                    
                    # Handle NaN values
                    features_array = np.array(features)
                    if not np.isnan(features_array).any():
                        features_scaled = self.scaler.transform(features_array)
                        risk_score = self.risk_model.predict(features_scaled)[0]
                        risk_scores[symbol] = max(0, min(100, risk_score * 1000))  # Scale to 0-100
                    else:
                        risk_scores[symbol] = 50  # Default medium risk
                else:
                    risk_scores[symbol] = 50
            
            return risk_scores
        except:
            return {symbol: 50 for symbol in returns_data.columns}

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 SmartPortfolio Risk Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Investment Portfolio Risk Assessment Dashboard</p>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = PortfolioRiskAnalyzer()
    
    # Sidebar
    st.sidebar.title("🎯 Portfolio Configuration")
    st.sidebar.markdown("---")
    
    # Sample portfolios for easy testing
    sample_portfolios = {
        "Tech Growth Portfolio": {"AAPL": 30, "MSFT": 25, "GOOGL": 20, "NVDA": 15, "TSLA": 10},
        "Balanced Portfolio": {"SPY": 40, "BND": 30, "VTI": 20, "GLD": 10},
        "Dividend Portfolio": {"JNJ": 25, "PG": 20, "KO": 20, "PFE": 20, "T": 15},
        "Custom Portfolio": {}
    }
    
    selected_portfolio = st.sidebar.selectbox("Choose a sample portfolio or create custom:", list(sample_portfolios.keys()))
    
    if selected_portfolio == "Custom Portfolio":
        st.sidebar.subheader("Build Your Portfolio")
        num_stocks = st.sidebar.number_input("Number of stocks:", min_value=2, max_value=10, value=4)
        
        portfolio = {}
        total_weight = 0
        
        for i in range(num_stocks):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                symbol = st.text_input(f"Stock {i+1}:", key=f"stock_{i}", placeholder="AAPL").upper()
            with col2:
                weight = st.number_input(f"Weight %:", min_value=0.0, max_value=100.0, value=25.0, key=f"weight_{i}")
            
            if symbol:
                portfolio[symbol] = weight
                total_weight += weight
        
        if total_weight != 100:
            st.sidebar.warning(f"Weights sum to {total_weight:.1f}%. Adjust to 100%.")
    else:
        portfolio = sample_portfolios[selected_portfolio]
        total_weight = sum(portfolio.values())
    
    # Analysis button
    analyze_button = st.sidebar.button("🔍 Analyze Portfolio", type="primary")
    
    if analyze_button and portfolio and abs(total_weight - 100) < 0.1:
        # Convert weights to proportions
        weights = np.array(list(portfolio.values())) / 100
        symbols = list(portfolio.keys())
        
        st.success(f"Analyzing portfolio: {', '.join(symbols)}")
        
        # Fetch data
        with st.spinner("Fetching market data..."):
            price_data = analyzer.fetch_stock_data(symbols)
        
        if price_data:
            # Calculate returns
            returns_data = analyzer.calculate_returns(price_data)
            
            # Train ML model
            with st.spinner("Training AI risk model..."):
                model_trained, model_score = analyzer.train_risk_model(returns_data)
            
            # Calculate risk metrics
            risk_metrics = analyzer.calculate_risk_metrics(returns_data, weights)
            
            # Create main dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                annual_return_pct = risk_metrics['annual_return'] * 100
                st.metric("Annual Return", f"{annual_return_pct:.2f}%", delta=f"{annual_return_pct-8:.1f}% vs Market")
            
            with col2:
                volatility_pct = risk_metrics['annual_volatility'] * 100
                st.metric("Volatility", f"{volatility_pct:.2f}%", delta=f"{15-volatility_pct:.1f}% vs Benchmark")
            
            with col3:
                st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}", delta=f"{risk_metrics['sharpe_ratio']-1.2:.2f} vs Target")
            
            with col4:
                st.metric("Beta", f"{risk_metrics['beta']:.2f}", delta="Market Sensitivity")
            
            # Risk Score
            st.markdown("---")
            
            if model_trained:
                risk_scores = analyzer.predict_risk_score(returns_data)
                portfolio_risk_score = np.average(list(risk_scores.values()), weights=weights)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("🤖 AI Risk Assessment")
                    if portfolio_risk_score < 30:
                        st.markdown(f'<div class="risk-low"><h3>LOW RISK: {portfolio_risk_score:.1f}/100</h3><p>Conservative portfolio with stable returns</p></div>', unsafe_allow_html=True)
                    elif portfolio_risk_score < 60:
                        st.markdown(f'<div class="risk-medium"><h3>MEDIUM RISK: {portfolio_risk_score:.1f}/100</h3><p>Balanced risk-reward profile</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-high"><h3>HIGH RISK: {portfolio_risk_score:.1f}/100</h3><p>Aggressive portfolio with high volatility</p></div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Model Accuracy", f"{model_score:.1%}" if model_score > 0 else "Training...", delta="R² Score")
            
            # Visualizations
            st.markdown("---")
            
            # Portfolio composition
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Portfolio Composition")
                fig_pie = px.pie(values=list(portfolio.values()), names=list(portfolio.keys()), 
                               color_discrete_sequence=px.colors.qualitative.Set3)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("📈 Cumulative Returns")
                portfolio_cumulative = (1 + risk_metrics['portfolio_returns']).cumprod()
                fig_returns = px.line(x=portfolio_cumulative.index, y=portfolio_cumulative.values,
                                    title="Portfolio Performance Over Time")
                fig_returns.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return")
                st.plotly_chart(fig_returns, use_container_width=True)
            
            # Correlation matrix
            st.subheader("🔗 Asset Correlation Matrix")
            corr_matrix = returns_data.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                               color_continuous_scale='RdYlBu_r', title="Correlation Between Assets")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Individual stock risk scores
            if model_trained:
                st.subheader("🎯 Individual Stock Risk Scores")
                risk_df = pd.DataFrame(list(risk_scores.items()), columns=['Stock', 'Risk Score'])
                fig_risk = px.bar(risk_df, x='Stock', y='Risk Score', 
                                color='Risk Score', color_continuous_scale='RdYlGn_r',
                                title="AI-Predicted Risk Scores by Stock")
                st.plotly_chart(fig_risk, use_container_width=True)
            
            # Risk metrics table
            st.subheader("📋 Detailed Risk Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Value at Risk (95%)', 'Maximum Drawdown', 'Beta', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio'],
                'Value': [f"{risk_metrics['var_95']*100:.2f}%", f"{risk_metrics['max_drawdown']*100:.2f}%", 
                         f"{risk_metrics['beta']:.2f}", f"{risk_metrics['annual_return']*100:.2f}%",
                         f"{risk_metrics['annual_volatility']*100:.2f}%", f"{risk_metrics['sharpe_ratio']:.2f}"],
                'Interpretation': ['Daily loss not exceeded 95% of time', 'Worst peak-to-trough decline',
                                 'Sensitivity to market movements', 'Expected annual return',
                                 'Annual price volatility', 'Risk-adjusted return measure']
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 AI Recommendations")
            
            recommendations = []
            if risk_metrics['sharpe_ratio'] < 0.5:
                recommendations.append("⚠️ Consider reducing high-risk assets to improve risk-adjusted returns")
            if risk_metrics['annual_volatility'] > 0.25:
                recommendations.append("📉 Portfolio volatility is high - consider diversification")
            if max(weights) > 0.4:
                recommendations.append("⚖️ Consider reducing concentration in single positions")
            if risk_metrics['beta'] > 1.5:
                recommendations.append("📊 Portfolio is highly sensitive to market movements")
            
            if not recommendations:
                recommendations.append("✅ Portfolio shows balanced risk characteristics")
            
            for rec in recommendations:
                st.info(rec)
        
        else:
            st.error("Failed to fetch market data. Please check stock symbols.")
    
    elif analyze_button:
        st.error("Please ensure portfolio weights sum to 100%")
    
    # Information section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        ### SmartPortfolio Risk Analyzer
        
        This application demonstrates advanced financial analytics and machine learning capabilities:
        
        **🔧 Technical Features:**
        - Real-time market data integration via Yahoo Finance API
        - Machine Learning risk prediction using Random Forest
        - Advanced financial metrics (VaR, Sharpe Ratio, Beta, Maximum Drawdown)
        - Interactive visualizations with Plotly
        - Professional UI/UX design
        
        **📊 Risk Metrics Explained:**
        - **Value at Risk (VaR)**: Maximum expected loss over a specific time period
        - **Sharpe Ratio**: Risk-adjusted return measure
        - **Beta**: Sensitivity to market movements
        - **Maximum Drawdown**: Largest peak-to-trough decline
        
        **🤖 AI Model:**
        - Trained on historical volatility patterns
        - Uses rolling statistical features
        - Provides predictive risk scoring
        
        **Built with:** Python, Streamlit, scikit-learn, yfinance, Plotly
        """)

if __name__ == "__main__":
    main()
