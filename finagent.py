# Run `pip install requests textblob` to install dependencies.
# headers
import request
from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv()


def get_company_symbol(company: str) -> str:
    """Use this function to get the symbol for an Indian company.

    Args:
        company (str): The name of the company.

    Returns:
        str: The symbol for the company on NSE/BSE.
    """
    symbols = {
        "Reliance Industries": "RELIANCE",
        "Tata Consultancy Services": "TCS",
        "Infosys": "INFY",
        "HDFC Bank": "HDFCBANK",
        "ICICI Bank": "ICICIBANK",
        "State Bank of India": "SBIN",
        "Bajaj Finance": "BAJFINANCE",
        "Wipro": "WIPRO",
    }
    return symbols.get(company, "Unknown")


def get_news_data(company: str) -> str:
    """Fetch latest news data from an Indian news source.

    Args:
        company (str): The name of the company.

    Returns:
        str: Latest news headlines related to the company.
    """
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&apiKey=YOUR_NEWS_API_KEY"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        news_headlines = [article["title"] for article in articles[:5]]  # Get top 5 headlines
        return " | ".join(news_headlines)
    return "No recent news available."


def analyze_sentiment(news: str) -> int:
    """Perform sentiment analysis on news headlines and return a score out of 100.

    Args:
        news (str): The news text to analyze.

    Returns:
        int: Sentiment score (0-100), where 0 is very negative, 50 is neutral, and 100 is very positive.
    """
    sentiment = TextBlob(news).sentiment.polarity
    score = int((sentiment + 1) * 50)  # Convert polarity (-1 to 1) to a scale of 0 to 100
    return score


agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[get_company_symbol, get_news_data, analyze_sentiment],
    instructions=[
        "Use tables to display data.",
        "If you need to find the symbol for an Indian company, use the get_company_symbol tool.",
        "Fetch latest news from Indian news sources for better insights.",
        "Analyze news sentiment and provide a score from 0 to 100 indicating negative, neutral, or positive sentiment.",
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

agent.print_response(
    "Summarize and compare analyst recommendations, fundamentals, latest news, and sentiment analysis for RELIANCE and TCS. Show in tables.",
    stream=True,
)
