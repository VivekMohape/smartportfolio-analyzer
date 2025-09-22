# 📈 SmartPortfolio Risk Analyzer

AI-Powered Investment Portfolio Risk Assessment Dashboard built with Streamlit, Machine Learning, and Real-time Market Data.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-app-url-here)

## ✨ Features

- **🤖 AI Risk Scoring**: Machine Learning models predict portfolio risk using Random Forest
- **📊 Real-time Data**: Live market data integration via Yahoo Finance API
- **📈 Advanced Analytics**: VaR, Sharpe Ratio, Beta, Maximum Drawdown calculations
- **🎨 Professional UI**: Custom-styled responsive dashboard with interactive charts
- **⚡ One-Click Analysis**: Pre-built sample portfolios for instant testing
- **📱 Mobile Responsive**: Works seamlessly on all device sizes

## 🎯 Quick Start

### Option 1: Streamlit Cloud (Recommended)
1. Click the demo link above
2. Select a sample portfolio (Tech Growth, Balanced, or Dividend)
3. Click "🔍 Analyze Portfolio"
4. Explore real-time risk analytics!

### Option 2: Local Development
```bash
# Clone the repository
git clone <your-repo-url>
cd smartportfolio-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## 📊 Sample Portfolios

### Tech Growth Portfolio 🚀
- Apple (AAPL) - 30%
- Microsoft (MSFT) - 25%
- Google (GOOGL) - 20%
- NVIDIA (NVDA) - 15%
- Tesla (TSLA) - 10%

### Balanced Portfolio ⚖️
- S&P 500 ETF (SPY) - 40%
- Bond ETF (BND) - 30%
- Total Stock Market (VTI) - 20%
- Gold ETF (GLD) - 10%

### Dividend Portfolio 💰
- Johnson & Johnson (JNJ) - 25%
- Procter & Gamble (PG) - 20%
- Coca-Cola (KO) - 20%
- Pfizer (PFE) - 20%
- AT&T (T) - 15%

## 🔧 Technical Stack

- **Frontend**: Streamlit with custom CSS styling
- **Data**: Yahoo Finance API (yfinance)
- **ML**: scikit-learn Random Forest for risk prediction
- **Visualization**: Plotly for interactive charts
- **Analytics**: Pandas, NumPy for financial calculations

## 📈 Risk Metrics Explained

| Metric | Description |
|--------|-------------|
| **Value at Risk (VaR)** | Maximum expected loss over specific time period (95% confidence) |
| **Sharpe Ratio** | Risk-adjusted return measure (higher = better) |
| **Beta** | Sensitivity to market movements (1.0 = market average) |
| **Maximum Drawdown** | Largest peak-to-trough decline in portfolio value |
| **Annual Volatility** | Standard deviation of returns (lower = less risky) |

## 🔒 Streamlit Secrets

**Currently: NO SECRETS REQUIRED** ✅

This application uses free public APIs and doesn't require any API keys or sensitive configuration.

For future enhancements requiring API keys:
```toml
# .streamlit/secrets.toml (not needed currently)
[api_keys]
# alpha_vantage = "your_key_here"
# quandl = "your_key_here"
```

## 🛠️ Development

### Project Structure
```
smartportfolio-analyzer/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── Instructions.md         # Deployment guide
├── README.md              # This file
├── .gitignore            # Git ignore rules
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

### Adding New Features
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test locally
4. Submit pull request

## 📱 Screenshots

*Add screenshots of your dashboard here when deploying*

## ⚠️ Limitations

- **Market Hours**: Real-time data availability depends on market hours
- **API Rate Limits**: Yahoo Finance has usage limits for free tier
- **Internet Required**: Needs connection for live data (no offline mode)

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Yahoo Finance for market data
- Streamlit team for the amazing framework
- scikit-learn for ML capabilities

---

**Built for portfolio optimization and financial risk assessment demonstrations.**