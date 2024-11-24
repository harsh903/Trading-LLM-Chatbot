import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
from trading_bot import (
    FinanceDataManager,
    RiskDetectionManager,
    ContrarianAnalysis,
    MarketAnalysis,
    TradingBias
)
import json
from langchain_openai import ChatOpenAI

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="🚀 Advanced Trading Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

def create_candlestick_chart(ticker: str, period: str = "6mo") -> go.Figure:
    """Create an advanced candlestick chart with technical indicators"""
    stock = yf.Ticker(ticker)
    print(stock)
    hist = stock.history(period=period)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='Price'
    ), row=1, col=1)

    # Add Moving Averages
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['MA20'],
        line=dict(color='orange', width=1),
        name='20-day MA'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['MA50'],
        line=dict(color='blue', width=1),
        name='50-day MA'
    ), row=1, col=1)

    # Volume bars
    colors = ['red' if close < open else 'green' for close, open in zip(hist['Close'], hist['Open'])]
    
    fig.add_trace(go.Bar(
        x=hist.index,
        y=hist['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.5
    ), row=2, col=1)

    fig.update_layout(
        height=800,
        title=f'{ticker} Price Chart',
        yaxis_title='Price',
        yaxis2_title='Volume',
        template='plotly_white'
    )

    return fig

def display_technical_metrics(stock_data: dict):
    """Display technical analysis metrics in an organized layout"""
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.subheader("📊 Technical Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rsi = stock_data['technical_indicators']['rsi']
        rsi_color = 'red' if rsi > 70 else 'green' if rsi < 30 else 'orange'
        st.metric(
            "RSI (14)",
            value=f"{rsi:.2f}",
            delta=f"{'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}"
        )
        st.write(f"Status: :{'red' if rsi > 70 else 'green' if rsi < 30 else 'blue'}[{stock_data['technical_indicators']['trend'].upper()}]")
    
    with col2:
        st.metric(
            "Moving Averages",
            value=f"MA20: ${stock_data['technical_indicators']['sma_20']:.2f}",
            delta=f"MA50: ${stock_data['technical_indicators']['sma_50']:.2f}"
        )
        st.write(f"Trend: {stock_data['technical_indicators']['trend'].upper()}")
    
    with col3:
        st.metric(
            "Volatility",
            value=f"{stock_data['technical_indicators']['volatility']}%",
            delta="High" if stock_data['technical_indicators']['volatility'] > 30 else "Low"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_fundamental_metrics(stock_data: dict):
    """Display fundamental analysis metrics"""
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.subheader("📈 Fundamental Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Market Cap",
            value=f"${stock_data['fundamental_data']['market_cap']:,.0f}"
        )
        st.metric(
            "P/E Ratio",
            value=f"{stock_data['fundamental_data']['pe_ratio']:.2f}"
        )
    
    with col2:
        st.metric(
            "EPS",
            value=f"${stock_data['fundamental_data']['eps']:.2f}"
        )
        st.metric(
            "Profit Margin",
            value=f"{stock_data['fundamental_data']['profit_margins']*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Debt/Equity",
            value=f"{stock_data['fundamental_data']['debt_to_equity']:.2f}"
        )
        st.metric(
            "Beta",
            value=f"{stock_data['fundamental_data']['beta']:.2f}"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_market_sentiment(stock_data: dict, news_data: list):
    """Display market sentiment analysis"""
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.subheader("📰 Market Sentiment")
    
    # News sentiment analysis
    positive_news = sum(1 for news in news_data if news.get('type') == 'positive')
    negative_news = sum(1 for news in news_data if news.get('type') == 'negative')
    neutral_news = len(news_data) - positive_news - negative_news
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=['Positive', 'Negative', 'Neutral'],
            values=[positive_news, negative_news, neutral_news],
            hole=.3
        )])
        fig.update_layout(title='News Sentiment Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("Recent Headlines:")
        for news in news_data[:5]:
            st.markdown(f"• **{news['date']}**: {news['title']}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_trade_signals(stock_data: dict, news_data: list):
    """Display trading signals and recommendations"""
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.subheader("🎯 Trading Signals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bullish_signals = ContrarianAnalysis.find_bullish_signals(stock_data, news_data)
        st.write("🟢 Bullish Signals:")
        for category, signals in bullish_signals.items():
            if signals:
                st.write(f"**{category.title()}:**")
                for signal in signals:
                    st.write(f"• {signal['signal']} ({signal['strength']})")
    
    with col2:
        bearish_signals = ContrarianAnalysis.find_bearish_signals(stock_data, news_data)
        st.write("🔴 Bearish Signals:")
        for category, signals in bearish_signals.items():
            if signals:
                st.write(f"**{category.title()}:**")
                for signal in signals:
                    st.write(f"• {signal['signal']} ({signal['strength']})")
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_risk_analysis(risk_data: dict):
    """Display enhanced risk analysis metrics"""
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.subheader("⚠️ Risk Analysis")
    
    # Risk score gauge with trend indicator
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_data['risk_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Risk Score ({risk_data['risk_trend'].title()})"},
            gauge={
                'axis': {'range': [None, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_data['risk_score']
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        trend_color = {
            "increasing": "red",
            "stable": "orange",
            "decreasing": "green"
        }.get(risk_data['risk_trend'], "gray")
        
        st.markdown(f"""
            ### Risk Trend
            <p style='color: {trend_color}; font-size: 20px;'>
            {risk_data['risk_trend'].upper()} 
            {'⬆️' if risk_data['risk_trend'] == 'increasing' else '➡️' if risk_data['risk_trend'] == 'stable' else '⬇️'}
            </p>
        """, unsafe_allow_html=True)
    
    # Risk breakdown
    st.markdown("### Risk Category Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        for category, score in risk_data['risk_breakdown'].items():
            if score > 0:  # Only show categories with non-zero scores
                progress_color = (
                    "red" if score > 70 
                    else "orange" if score > 30 
                    else "green"
                )
                st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span>{category.title()}</span>
                        <span style='color: {progress_color}'>{score:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(score/100)
    
    with col2:
        st.markdown("### High Priority Risks")
        if risk_data['high_priority_risks']:
            for risk in risk_data['high_priority_risks']:
                with st.expander(f"🚨 {risk['title'][:50]}..."):
                    st.write(f"**Category:** {risk['category'].title()}")
                    st.write(f"**Date:** {risk['date']}")
                    st.write(f"**Severity:** {risk['severity'].upper()}")
                    st.write(risk['summary'])
        else:
            st.info("No high priority risks detected")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional risk insights
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.subheader("🔍 Risk Insights")
    
    # Calculate risk statistics
    high_risk_categories = [k for k, v in risk_data['risk_breakdown'].items() if v > 70]
    moderate_risk_categories = [k for k, v in risk_data['risk_breakdown'].items() if 30 < v <= 70]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Risk Areas")
        if high_risk_categories:
            st.error("High Risk Categories:")
            for category in high_risk_categories:
                st.markdown(f"• {category.title()}")
        
        if moderate_risk_categories:
            st.warning("Moderate Risk Categories:")
            for category in moderate_risk_categories:
                st.markdown(f"• {category.title()}")
                
        if not high_risk_categories and not moderate_risk_categories:
            st.success("No significant risk areas detected")
    
    with col2:
        st.markdown("### Risk Management Suggestions")
        if risk_data['risk_score'] > 70:
            st.error("⚠️ Immediate action recommended:")
            st.markdown("• Consider reducing exposure")
            st.markdown("• Implement hedging strategies")
            st.markdown("• Monitor high-risk categories closely")
        elif risk_data['risk_score'] > 30:
            st.warning("⚠️ Enhanced monitoring recommended:")
            st.markdown("• Review risk mitigation strategies")
            st.markdown("• Monitor risk trends")
            st.markdown("• Prepare contingency plans")
        else:
            st.success("✅ Standard monitoring sufficient:")
            st.markdown("• Maintain current risk controls")
            st.markdown("• Regular risk assessment")
            st.markdown("• Monitor for changes")
    
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.title("🚀 Advanced Trading Analysis Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("SambaNova API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API Key configured successfully!")
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Settings")
        analysis_period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
        
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Basic", "Standard", "Advanced", "Expert"],
            value="Advanced"
        )
    
    # Main content
    if st.session_state.api_key is None:
        st.warning("⚠️ Please enter your SambaNova API Key in the sidebar to continue.")
        return
    
    # Stock ticker input with autocomplete
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("🎯 Enter Stock Ticker:", value="AAPL").upper()
    with col2:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_button:
        try:
            with st.spinner("📊 Fetching market data and generating analysis..."):
                # Get stock data
                stock_data = FinanceDataManager.get_stock_data(ticker)
                news_data = FinanceDataManager.get_stock_news(ticker)
                risk_data = RiskDetectionManager.analyze_geopolitical_risks(news_data, stock_data)
                
                if stock_data is None:
                    st.error(f"❌ Could not fetch data for ticker {ticker}")
                    return
                
                # Main dashboard tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Overview",
                    "📈 Technical Analysis",
                    "📰 News & Sentiment",
                    "⚠️ Risk Analysis"
                ])
                
                with tab1:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Current Price",
                            f"${stock_data['price_data']['current_price']}",
                            f"{stock_data['price_data']['daily_change']}%"
                        )
                    with col2:
                        st.metric(
                            "Trading Volume",
                            f"{stock_data['price_data']['volume']:,}"
                        )
                    with col3:
                        st.metric(
                            "Market Cap",
                            f"${stock_data['fundamental_data']['market_cap']:,.0f}"
                        )
                    
                    # Charts
                    st.plotly_chart(
                        create_candlestick_chart(ticker, analysis_period),
                        use_container_width=True
                    )
                    
                    # Metrics
                    display_technical_metrics(stock_data)
                    display_fundamental_metrics(stock_data)
                
                with tab2:
                    # Technical analysis details
                    market_analysis = MarketAnalysis.analyze_price_action(stock_data)
                    volume_analysis = MarketAnalysis.analyze_volume_profile(stock_data)
                    trade_ideas = MarketAnalysis.generate_trade_ideas(stock_data, risk_data)
                    
                    # Display technical analysis components
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.subheader("🎯 Technical Analysis Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### Trend Analysis")
                        st.write(f"Primary Trend: **{market_analysis['trend']['primary'].upper()}**")
                        st.write(f"Trend Strength: **{market_analysis['trend']['strength'].upper()}**")
                        st.write(f"Momentum: **{market_analysis['trend']['momentum'].upper()}**")
                        
                        st.write("#### Support & Resistance")
                        st.write(f"Support Level: **${market_analysis['support_resistance']['support']:.2f}**")
                        st.write(f"Resistance Level: **${market_analysis['support_resistance']['resistance']:.2f}**")
                        st.write(f"Price Location: **{market_analysis['support_resistance']['price_location'].replace('_', ' ').title()}**")
                    
                    with col2:
                        st.write("#### Volume Analysis")
                        st.write(f"Current Volume: **{volume_analysis['current_volume']:,}**")
                        st.write(f"Average Volume: **{volume_analysis['average_volume']:,}**")
                        st.write(f"Volume Ratio: **{volume_analysis['volume_ratio']:.2f}**")
                        st.write(f"Trend Confirmation: **{volume_analysis['trend_confirmation'].upper()}**")
                        st.write(f"Volume Interpretation: **{volume_analysis['interpretation'].upper()}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display trade ideas
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.subheader("💡 Trade Ideas")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### Primary Strategy")
                        st.write(f"Direction: **{trade_ideas['primary_strategy']['direction'].upper()}**")
                        st.write("Entry Points:")
                        for i, point in enumerate(trade_ideas['primary_strategy']['entry_points'], 1):
                            st.write(f"  {i}. **${point:.2f}**")
                        st.write(f"Stop Loss: **${trade_ideas['primary_strategy']['stop_loss']:.2f}**")
                        st.write(f"Take Profit: **${trade_ideas['primary_strategy']['take_profit']:.2f}**")
                        st.write(f"Position Size: **{trade_ideas['primary_strategy']['position_size'].upper()}**")
                    
                    with col2:
                        st.write("#### Risk Management")
                        st.write(f"Max Position Size: **{trade_ideas['risk_management']['max_position_size']}**")
                        st.write(f"Suggested Leverage: **{trade_ideas['risk_management']['suggested_leverage']}**")
                        st.write(f"Hedging Required: **{'Yes' if trade_ideas['risk_management']['hedging_required'] else 'No'}**")
                        
                        st.write("#### Alternative Strategy")
                        st.write(f"Type: **{trade_ideas['alternative_strategy']['type'].replace('_', ' ').title()}**")
                        st.write(f"Entry Trigger: **${trade_ideas['alternative_strategy']['trigger_points']['entry']:.2f}**")
                        st.write(f"Exit Target: **${trade_ideas['alternative_strategy']['trigger_points']['exit']:.2f}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display signals
                    display_trade_signals(stock_data, news_data)
                
                with tab3:
                    # News and sentiment analysis
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.subheader("📰 Latest News")
                    
                    for news in news_data:
                        with st.expander(f"{news['date']} - {news['title']}"):
                            st.write(f"**Source:** {news['publisher']}")
                            st.write(news['summary'])
                            if news['url']:
                                st.markdown(f"[Read More]({news['url']})")
                            
                            # Add sentiment indicator
                            sentiment_color = {
                                'positive': 'green',
                                'negative': 'red',
                                'neutral': 'blue'
                            }.get(news['type'], 'gray')
                            st.markdown(f"**Sentiment:** :{sentiment_color}[{news['type'].upper()}]")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display market sentiment analysis
                    display_market_sentiment(stock_data, news_data)
                
                with tab4:
                    # Risk analysis
                    display_risk_analysis(risk_data)
                    
                    # Additional risk metrics
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    st.subheader("🔍 Detailed Risk Metrics")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### Technical Risk Factors")
                        st.write(f"Volatility: **{stock_data['technical_indicators']['volatility']}%**")
                        st.write(f"Trend Stability: **{market_analysis['trend']['strength']}**")
                        st.write(f"Volume Risk: **{volume_analysis['interpretation']}**")
                    
                    with col2:
                        st.write("#### Risk Trends")
                        st.write(f"Risk Trend: **{risk_data['risk_trend'].upper()}**")
                        st.write("High Priority Categories:")
                        high_risk_categories = [k for k, v in risk_data['risk_breakdown'].items() if v > 70]
                        for category in high_risk_categories:
                            st.write(f"• **{category.title()}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Generate AI Analysis
                if st.button("🤖 Generate AI Analysis"):
                    with st.spinner("Generating comprehensive AI analysis..."):
                        try:
                            analysis_agent = ChatOpenAI(
                                base_url="https://api.sambanova.ai/v1/",
                                api_key=st.session_state.api_key,
                                model="Meta-Llama-3.1-70B-Instruct",
                                temperature=0.7,
                                max_tokens=4000
                            )
                            
                            analysis_prompt = f"""Analyze {ticker} based on the following data:
                            
                            MARKET DATA:
                            {json.dumps(stock_data, indent=2)}
                            
                            RISK ANALYSIS:
                            {json.dumps(risk_data, indent=2)}
                            
                            NEWS DATA:
                            {json.dumps(news_data, indent=2)}
                            
                            MARKET ANALYSIS:
                            {json.dumps(market_analysis, indent=2)}
                            
                            Provide a comprehensive analysis including:
                            1. Market Position Assessment
                            2. Technical Analysis Interpretation
                            3. Risk-Adjusted Outlook
                            4. Investment Recommendation
                            5. Key Monitoring Points
                            
                            Format your response in a clear, structured manner with actionable insights."""
                            
                            analysis = analysis_agent.invoke(analysis_prompt)
                            
                            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                            st.markdown("### 🤖 AI Analysis Report")
                            st.write(analysis.content)
                            
                            # Add download button for analysis
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{ticker}_analysis_{timestamp}.txt"
                            st.download_button(
                                label="📥 Download Analysis Report",
                                data=analysis.content,
                                file_name=filename,
                                mime="text/plain"
                            )
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error generating AI analysis: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()