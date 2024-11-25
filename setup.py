from setuptools import setup, find_packages

setup(
    name="trading-llm-chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.31.0",
        "pandas>=2.1.4",
        "numpy>=1.26.0,<2.0.0",
        "yfinance>=0.2.36",
        "python-dotenv>=1.0.0",
        "plotly>=5.18.0",
        "ta>=0.10.2",
        "langgraph>=0.0.3",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "requests>=2.31.0",
        "python-dateutil>=2.8.2",
        "pytz>=2024.1",
    ],
    python_requires=">=3.12",
)
