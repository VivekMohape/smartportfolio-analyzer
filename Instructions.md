# SmartPortfolio Risk Analyzer - Deployment Guide

## 🚀 Quick Start

### Method 1: Local Development
```bash
# Clone or create project directory
mkdir smartportfolio-analyzer
cd smartportfolio-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Method 2: Streamlit Cloud (Recommended for Demo)
1. Create a GitHub repository with your code
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy directly from repository
5. Share the public URL with interviewers

## 📁 Project Structure
```
smartportfolio-analyzer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── assets/
    ├── demo_screenshots/ # Screenshots for README
    └── presentation/     # Interview materials
```

## ⚙️ Configuration Files

### .streamlit/config.toml
```toml
[theme]
primaryColor = "#1f4e79"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 200
enableCORS = false
```

## 🌐 Environment Variables (Optional)
```bash
# For production deployment
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🐳 Docker Deployment (Advanced)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔍 Testing Instructions
1. **Sample Portfolios**: Use pre-built portfolios for instant testing
2. **Custom Portfolio**: Create your own with popular stocks (AAPL, MSFT, GOOGL)
3. **Live Demo**: Ensure internet connection for real-time data

## 📊 Performance Optimization
- Data caching implemented for faster subsequent runs
- Progressive loading with status indicators
- Efficient DataFrame operations
- Minimal API calls to Yahoo Finance

## 🛠️ Troubleshooting
- **Data fetch errors**: Check internet connection and stock symbols
- **Module errors**: Ensure all requirements are installed
- **Streamlit issues**: Update to latest version: `pip install --upgrade streamlit`

## 📱 Mobile Responsiveness
The application is optimized for both desktop and mobile viewing, making it perfect for live demos.

## 🔐 Security Notes
- No sensitive data storage required
- Uses public market data only
- No authentication needed for demo purposes
- Safe for public deployment