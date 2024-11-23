import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from langchain_openai import ChatOpenAI
import numpy as np
import json
import ta

class TradingBias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle_band = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)
        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_support_resistance(hist: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        rolling_min = hist['Low'].rolling(window=window, center=True).min()
        rolling_max = hist['High'].rolling(window=window, center=True).max()
        
        support = rolling_min.iloc[-1]
        resistance = rolling_max.iloc[-1]
        
        return support, resistance

class FinanceDataManager:
    @staticmethod
    def get_stock_info(ticker):
        """
        This Function is created by soumanjyotiofficial
        Purpose of this function is to extract About company info
        and send to agent who will summerise the info of the company
        and Show out put in the dashboard.
        """
        return yf.Ticker("TCS.NS").info['longBusinessSummary']
    @staticmethod
    def get_stock_data(ticker: str) -> dict:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")

            # Basic Technical Indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = TechnicalIndicators.calculate_rsi(hist['Close'])
            macd, signal, histogram = TechnicalIndicators.calculate_macd(hist['Close'])
            upper_bb, middle_bb, lower_bb = TechnicalIndicators.calculate_bollinger_bands(hist['Close'])
            support, resistance = TechnicalIndicators.calculate_support_resistance(hist)

            # Advanced Technical Indicators
            hist['ADX'] = ta.trend.adx(hist['High'], hist['Low'], hist['Close'])
            hist['OBV'] = ta.volume.on_balance_volume(hist['Close'], hist['Volume'])
            
            last_idx = hist.index[-1]
            second_last_idx = hist.index[-2]

            # Calculate volatility
            returns = hist['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility

            # Calculate average volume
            avg_volume = hist['Volume'].mean()
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume

            return {
                "price_data": {
                    "current_price": round(hist.loc[last_idx, 'Close'], 2),
                    "daily_change": round(((hist.loc[last_idx, 'Close'] - hist.loc[second_last_idx, 'Close']) / 
                                        hist.loc[second_last_idx, 'Close']) * 100, 2),
                    "high_52w": round(hist['High'].max(), 2),
                    "low_52w": round(hist['Low'].min(), 2),
                    "volume": int(hist.loc[last_idx, 'Volume']),
                    "avg_volume": int(avg_volume),
                    "volume_ratio": round(volume_ratio, 2),
                },
                "technical_indicators": {
                    "sma_20": round(hist.loc[last_idx, 'SMA_20'], 2),
                    "sma_50": round(hist.loc[last_idx, 'SMA_50'], 2),
                    "rsi": round(hist.loc[last_idx, 'RSI'], 2),
                    "macd": round(macd.iloc[-1], 2),
                    "macd_signal": round(signal.iloc[-1], 2),
                    "macd_hist": round(histogram.iloc[-1], 2),
                    "bb_upper": round(upper_bb.iloc[-1], 2),
                    "bb_middle": round(middle_bb.iloc[-1], 2),
                    "bb_lower": round(lower_bb.iloc[-1], 2),
                    "adx": round(hist['ADX'].iloc[-1], 2),
                    "obv": int(hist['OBV'].iloc[-1]),
                    "support": round(support, 2),
                    "resistance": round(resistance, 2),
                    "volatility": round(volatility * 100, 2),  # In percentage
                    "trend": "bullish" if hist.loc[last_idx, 'SMA_20'] > hist.loc[last_idx, 'SMA_50'] else "bearish",
                },
                "fundamental_data": {
                    "market_cap": stock.info.get('marketCap'),
                    "pe_ratio": stock.info.get('forwardPE'),
                    "eps": stock.info.get('trailingEps'),
                    "revenue_growth": stock.info.get('revenueGrowth'),
                    "profit_margins": stock.info.get('profitMargins'),
                    "debt_to_equity": stock.info.get('debtToEquity'),
                    "dividend_yield": stock.info.get('dividendYield'),
                    "beta": stock.info.get('beta'),
                    "enterprise_value": stock.info.get('enterpriseValue'),
                    "price_to_book": stock.info.get('priceToBook'),
                },
                "analyst_data": {
                    "target_high": stock.info.get('targetHighPrice'),
                    "target_low": stock.info.get('targetLowPrice'),
                    "target_mean": stock.info.get('targetMeanPrice'),
                    "recommendation": stock.info.get('recommendationKey'),
                    "num_analysts": stock.info.get('numberOfAnalystOpinions'),
                }
            }
        except Exception as e:
            print(f"Error in get_stock_data: {str(e)}")
            raise

    @staticmethod
    def get_stock_news(ticker: str, days: int = 30) -> list:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            # Filter and process news
            processed_news = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for item in news:
                pub_date = datetime.fromtimestamp(item['providerPublishTime'])
                if pub_date >= cutoff_date:
                    processed_news.append({
                        "title": item.get('title'),
                        "publisher": item.get('publisher'),
                        "summary": item.get('summary'),
                        "date": pub_date.strftime('%Y-%m-%d'),
                        "url": item.get('link'),
                        "type": "positive" if any(word in item.get('title', '').lower() 
                                               for word in ['rise', 'gain', 'up', 'surge', 'jump'])
                               else "negative" if any(word in item.get('title', '').lower() 
                                                   for word in ['fall', 'drop', 'down', 'plunge', 'decline'])
                               else "neutral"
                    })
            
            return sorted(processed_news, key=lambda x: x['date'], reverse=True)
        except Exception as e:
            print(f"Error in get_stock_news: {str(e)}")
            return []

class RiskDetectionManager:
    @staticmethod
    def analyze_geopolitical_risks(news_data: List[Dict[str, Any]], company_data: Dict[str, Any]) -> Dict[str, Any]:
        if not news_data:
            return {
                "risk_analysis": {category: [] for category in [
                    "geopolitical", "regulatory", "reputational", 
                    "financial", "operational", "market"
                ]},
                "risk_score": 0,
                "risk_trend": "stable",
                "risk_breakdown": {
                    "geopolitical": 0,
                    "regulatory": 0,
                    "reputational": 0,
                    "financial": 0,
                    "operational": 0,
                    "market": 0
                },
                "high_priority_risks": [],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        risk_categories = {
            "geopolitical": [],
            "regulatory": [],
            "reputational": [],
            "financial": [],
            "operational": [],
            "market": []
        }

        risk_keywords = {
            "geopolitical": ["sanctions", "trade war", "conflict", "international", "embargo", "tariff", "political"],
            "regulatory": ["sec", "investigation", "regulatory", "compliance", "lawsuit", "legal", "fine"],
            "reputational": ["scandal", "controversy", "allegation", "misconduct", "criticism"],
            "financial": ["debt", "loss", "bankruptcy", "default", "downgrade", "liquidity"],
            "operational": ["supply chain", "disruption", "shortage", "recall", "failure", "delay"],
            "market": ["competition", "market share", "substitute", "trend", "demand", "pricing"]
        }

        # Process news with sentiment and severity
        for news in news_data:
            title = (news.get('title') or '').lower()
            summary = (news.get('summary') or '').lower()
            
            if not title and not summary:
                continue

            # Determine severity based on news type and content
            severity = "high" if news.get('type') == 'negative' else "medium"
            if any(critical_word in title.lower() for critical_word in 
                  ["urgent", "critical", "emergency", "crisis", "disaster"]):
                severity = "high"

            for category, keywords in risk_keywords.items():
                if any(keyword in title or keyword in summary for keyword in keywords):
                    risk_entry = {
                        "title": news['title'],
                        "date": news['date'],
                        "summary": news['summary'],
                        "severity": severity,
                        "category": category
                    }
                    risk_categories[category].append(risk_entry)

        # Calculate comprehensive risk metrics
        risk_metrics = RiskDetectionManager._calculate_risk_metrics(risk_categories, company_data)
        
        # Identify high priority risks (severe risks from any category)
        high_priority_risks = [
            risk for category_risks in risk_categories.values()
            for risk in category_risks
            if risk['severity'] == 'high'
        ]

        # Sort high priority risks by date
        high_priority_risks.sort(key=lambda x: x['date'], reverse=True)
        
        return {
            "risk_analysis": risk_categories,
            "risk_score": risk_metrics['total_score'],
            "risk_breakdown": risk_metrics['category_scores'],
            "risk_trend": risk_metrics['trend'],
            "high_priority_risks": high_priority_risks[:5],  # Top 5 high priority risks
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    @staticmethod
    def _calculate_risk_metrics(risk_categories: Dict[str, List[Dict[str, Any]]], 
                              company_data: Dict[str, Any]) -> Dict[str, Any]:
        category_weights = {
            "geopolitical": 0.2,
            "regulatory": 0.2,
            "reputational": 0.15,
            "financial": 0.2,
            "operational": 0.15,
            "market": 0.1
        }

        category_scores = {}
        total_weighted_score = 0
        
        for category, risks in risk_categories.items():
            # Base score calculation
            if risks:
                severity_multiplier = sum(1.5 if risk['severity'] == 'high' else 1.0 
                                       for risk in risks) / len(risks)
                base_score = min(100, len(risks) * 20)  # Cap at 100
                category_score = base_score * severity_multiplier
            else:
                category_score = 0

            # Apply category weight
            weighted_score = category_score * category_weights[category]
            category_scores[category] = round(category_score, 2)
            total_weighted_score += weighted_score

        # Normalize total score to 0-100 range
        total_score = min(100, total_weighted_score)

        # Determine risk trend
        if total_score > 70:
            trend = "increasing"
        elif total_score < 30:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            'total_score': round(total_score, 2),
            'category_scores': category_scores,
            'trend': trend
        }

class ContrarianAnalysis:
    @staticmethod
    def find_bearish_signals(stock_data: Dict[str, Any], news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        bearish_signals = {
            "technical": [],
            "fundamental": [],
            "sentiment": [],
            "risk": []
        }

        # Technical Analysis
        technical = stock_data['technical_indicators']
        if technical['rsi'] > 70:
            bearish_signals['technical'].append({
                "signal": "RSI indicates overbought conditions",
                "value": technical['rsi'],
                "strength": "strong" if technical['rsi'] > 80 else "moderate"
            })
        
        if technical['sma_20'] < technical['sma_50']:
            bearish_signals['technical'].append({
                "signal": "Death cross pattern in moving averages",
                "value": f"SMA20: {technical['sma_20']}, SMA50: {technical['sma_50']}",
                "strength": "strong"
            })

        price = stock_data['price_data']['current_price']
        if price > technical['bb_upper']:
            bearish_signals['technical'].append({
                "signal": "Price above upper Bollinger Band",
                "value": f"Price: {price}, BB Upper: {technical['bb_upper']}",
                "strength": "moderate"
            })

        # Fundamental Analysis
        fundamentals = stock_data['fundamental_data']
        if fundamentals['pe_ratio'] and fundamentals['pe_ratio'] > 30:
            bearish_signals['fundamental'].append({
                "signal": "High P/E ratio suggesting overvaluation",
                "value": fundamentals['pe_ratio'],
                "strength": "strong" if fundamentals['pe_ratio'] > 50 else "moderate"
            })

        if fundamentals['debt_to_equity'] and fundamentals['debt_to_equity'] > 2:
            bearish_signals['fundamental'].append({
                "signal": "High debt-to-equity ratio",
                "value": fundamentals['debt_to_equity'],
                "strength": "strong" if fundamentals['debt_to_equity'] > 3 else "moderate"
            })

        # Volume Analysis
        if stock_data['price_data']['volume_ratio'] < 0.7:
            bearish_signals['technical'].append({
                "signal": "Below average volume indicating weak momentum",
                "value": f"Volume ratio: {stock_data['price_data']['volume_ratio']}",
                "strength": "moderate"
            })

        # Sentiment Analysis from News
        negative_news_count = sum(1 for news in news_data if news.get('type') == 'negative')
        if negative_news_count > len(news_data) * 0.5:
            bearish_signals['sentiment'].append({
                "signal": "Negative news sentiment dominates",
                "value": f"{negative_news_count}/{len(news_data)} negative news",
                "strength": "strong" if negative_news_count > len(news_data) * 0.7 else "moderate"
            })

        # Market Analysis
        if technical['volatility'] > 30:
            bearish_signals['risk'].append({
                "signal": "High market volatility",
                "value": f"{technical['volatility']}% volatility",
                "strength": "strong" if technical['volatility'] > 50 else "moderate"
            })

        return bearish_signals

    @staticmethod
    def find_bullish_signals(stock_data: Dict[str, Any], news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        bullish_signals = {
            "technical": [],
            "fundamental": [],
            "sentiment": [],
            "opportunity": []
        }

        # Technical Analysis
        technical = stock_data['technical_indicators']
        if technical['rsi'] < 30:
            bullish_signals['technical'].append({
                "signal": "RSI indicates oversold conditions",
                "value": technical['rsi'],
                "strength": "strong" if technical['rsi'] < 20 else "moderate"
            })
        
        if technical['sma_20'] > technical['sma_50']:
            bullish_signals['technical'].append({
                "signal": "Golden cross pattern in moving averages",
                "value": f"SMA20: {technical['sma_20']}, SMA50: {technical['sma_50']}",
                "strength": "strong"
            })

        price = stock_data['price_data']['current_price']
        if price < technical['bb_lower']:
            bullish_signals['technical'].append({
                "signal": "Price below lower Bollinger Band",
                "value": f"Price: {price}, BB Lower: {technical['bb_lower']}",
                "strength": "moderate"
            })

        # MACD Analysis
        if technical['macd_hist'] > 0 and technical['macd'] > technical['macd_signal']:
            bullish_signals['technical'].append({
                "signal": "MACD bullish crossover",
                "value": f"MACD: {technical['macd']}, Signal: {technical['macd_signal']}",
                "strength": "strong"
            })

        # Fundamental Analysis
        fundamentals = stock_data['fundamental_data']
        if fundamentals['pe_ratio'] and fundamentals['pe_ratio'] < 15:
            bullish_signals['fundamental'].append({
                "signal": "Attractive P/E ratio suggesting undervaluation",
                "value": fundamentals['pe_ratio'],
                "strength": "strong" if fundamentals['pe_ratio'] < 10 else "moderate"
            })

        if fundamentals['profit_margins'] and fundamentals['profit_margins'] > 0.2:
            bullish_signals['fundamental'].append({
                "signal": "Strong profit margins",
                "value": f"{fundamentals['profit_margins'] * 100:.1f}%",
                "strength": "strong" if fundamentals['profit_margins'] > 0.3 else "moderate"
            })

        # Volume Analysis
        if stock_data['price_data']['volume_ratio'] > 1.5:
            bullish_signals['technical'].append({
                "signal": "Above average volume indicating strong momentum",
                "value": f"Volume ratio: {stock_data['price_data']['volume_ratio']}",
                "strength": "strong" if stock_data['price_data']['volume_ratio'] > 2 else "moderate"
            })

        # Sentiment Analysis from News
        positive_news_count = sum(1 for news in news_data if news.get('type') == 'positive')
        if positive_news_count > len(news_data) * 0.5:
            bullish_signals['sentiment'].append({
                "signal": "Positive news sentiment dominates",
                "value": f"{positive_news_count}/{len(news_data)} positive news",
                "strength": "strong" if positive_news_count > len(news_data) * 0.7 else "moderate"
            })

        # Support/Resistance Analysis
        if price < technical['support'] * 1.02:  # Within 2% of support
            bullish_signals['opportunity'].append({
                "signal": "Price near support level",
                "value": f"Support: ${technical['support']:.2f}",
                "strength": "strong"
            })

        return bullish_signals

class MarketAnalysis:
    @staticmethod
    def analyze_price_action(stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price action patterns and trends"""
        technical = stock_data['technical_indicators']
        price_data = stock_data['price_data']
        
        analysis = {
            "trend": {
                "primary": technical['trend'],
                "strength": "strong" if abs(technical['sma_20'] - technical['sma_50']) / technical['sma_50'] > 0.05 else "moderate",
                "momentum": "positive" if technical['macd'] > technical['macd_signal'] else "negative"
            },
            "support_resistance": {
                "support": technical['support'],
                "resistance": technical['resistance'],
                "price_location": "near_support" if price_data['current_price'] < technical['support'] * 1.02 
                                 else "near_resistance" if price_data['current_price'] > technical['resistance'] * 0.98
                                 else "mid_range"
            },
            "volatility": {
                "value": technical['volatility'],
                "interpretation": "high" if technical['volatility'] > 30 
                                else "low" if technical['volatility'] < 15 
                                else "moderate"
            }
        }
        return analysis

    @staticmethod
    def analyze_volume_profile(stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume patterns and their implications"""
        volume_data = {
            "current_volume": stock_data['price_data']['volume'],
            "average_volume": stock_data['price_data']['avg_volume'],
            "volume_ratio": stock_data['price_data']['volume_ratio'],
            "trend_confirmation": "strong" if stock_data['price_data']['volume_ratio'] > 1.5 else "weak",
            "interpretation": "bullish" if stock_data['price_data']['volume_ratio'] > 1.5 and 
                                        stock_data['technical_indicators']['trend'] == "bullish"
                            else "bearish" if stock_data['price_data']['volume_ratio'] > 1.5 and 
                                            stock_data['technical_indicators']['trend'] == "bearish"
                            else "neutral"
        }
        return volume_data

    @staticmethod
    def generate_trade_ideas(stock_data: Dict[str, Any], risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable trade ideas based on analysis"""
        technical = stock_data['technical_indicators']
        price = stock_data['price_data']['current_price']
        
        # Calculate risk-reward levels
        stop_loss = price * 0.95  # 5% stop loss
        take_profit = price * (1 + (1 - risk_data['risk_score']/100) * 0.2)  # Variable take profit based on risk
        
        trade_ideas = {
            "primary_strategy": {
                "direction": "long" if technical['trend'] == "bullish" and technical['rsi'] < 60
                            else "short" if technical['trend'] == "bearish" and technical['rsi'] > 40
                            else "neutral",
                "entry_points": [price * 0.99, price * 0.98, price * 0.97],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": "moderate" if risk_data['risk_score'] < 50 else "small"
            },
            "alternative_strategy": {
                "type": "mean_reversion" if technical['rsi'] > 70 or technical['rsi'] < 30 else "trend_following",
                "trigger_points": {
                    "entry": technical['support'] if technical['trend'] == "bullish" else technical['resistance'],
                    "exit": technical['resistance'] if technical['trend'] == "bullish" else technical['support']
                }
            },
            "risk_management": {
                "max_position_size": f"{100 - risk_data['risk_score']}%",
                "suggested_leverage": "1x" if risk_data['risk_score'] > 70 else "2x",
                "hedging_required": risk_data['risk_score'] > 60
            }
        }
        return trade_ideas
