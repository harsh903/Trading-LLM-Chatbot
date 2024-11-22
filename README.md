# Advanced Trading Analysis Dashboard

## Project Description
The Advanced Trading Analysis Dashboard is a sophisticated financial analysis tool that combines real-time market data with AI-powered insights. Built using Python and Streamlit, this platform provides traders and investors with comprehensive market analysis, including technical indicators, sentiment analysis, and risk assessment.

The dashboard integrates multiple data sources and advanced analytics to offer actionable trading insights. It features real-time stock tracking, technical analysis, news sentiment evaluation, and risk management tools, all presented through an intuitive user interface.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Keys](#api-keys)
- [Dependencies](#dependencies)

## Features

### Real-Time Market Analysis
- Live stock price tracking
- Technical indicators (RSI, Moving Averages, MACD)
- Volume analysis
- Support/Resistance levels

### AI-Powered Insights
- Market sentiment analysis
- News impact assessment
- Trading recommendations
- Risk evaluation

### Technical Analysis
- Interactive price charts
- Multiple timeframe analysis
- Trend identification
- Pattern recognition

### Risk Management
- Risk scoring system
- Geopolitical risk analysis
- Market sentiment evaluation
- Position sizing recommendations

### News Integration
- Real-time news updates
- Sentiment analysis
- Impact assessment
- Trend correlation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-analysis-dashboard.git
cd trading-analysis-dashboard
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- macOS/Linux:
```bash
source venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Set up your API keys in a `.env` file:
```env
SAMBANOVA_API_KEY=your_api_key_here
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the dashboard at `http://localhost:8501`

## Project Structure

```
trading-analysis-dashboard/
├── app.py                  # Main Streamlit application
├── trading_bot.py          # Core trading analysis functionality
├── requirements.txt        # Project dependencies
├── .env                   # Environment variables (create this)
└── README.md              # Project documentation
```

## API Keys

The application requires the following API key:
- **SambaNova API Key** - For AI analysis capabilities
  - Sign up at [SambaNova](https://sambanova.ai/)
  - Create an API key
  - Add to your `.env` file

## Dependencies

Main dependencies include:
```
streamlit==1.31.0
pandas==2.1.4
numpy==1.24.3
yfinance==0.2.36
plotly==5.18.0
ta==0.10.2
langchain==0.1.4
langchain-openai==0.0.5
python-dotenv==1.0.1
```

For a complete list, see `requirements.txt`

## System Requirements

- Python 3.10+
- 4GB RAM minimum
- Internet connection for real-time data
- Modern web browser

## Getting Started

1. **Basic Usage**:
```python
# Run the application
streamlit run app.py

# Enter your API key in the sidebar
# Enter a stock ticker to analyze
# View analysis results
```

2. **Example Analysis**:
```python
# Enter a stock ticker (e.g., AAPL)
# Click "Analyze" button
# View technical analysis, news, and AI insights
```

## Configuration

Key configuration options in `app.py`:
```python
# Time period for analysis
analysis_period = ["1mo", "3mo", "6mo", "1y", "2y"]

# Analysis depth options
analysis_depth = ["Basic", "Standard", "Advanced", "Expert"]
```
