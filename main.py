import os
import json
import logging
import asyncio
import random
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Stock Analysis Panel")

# CORS - 允许所有来源（Render 部署时无需担心跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase 初始化
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    logger.warning("Supabase credentials missing. Storage disabled.")
    supabase: Optional[Client] = None
else:
    supabase = create_client(supabase_url, supabase_key)

# DeepSeek 初始化
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    logger.warning("DeepSeek API key missing. LLM analysis disabled.")
    deepseek_client = None
else:
    deepseek_client = AsyncOpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1"
    )

class AnalyzeRequest(BaseModel):
    symbol: str

def make_json_serializable(obj: Any) -> Any:
    """递归将对象转换为 JSON 可序列化类型（处理 numpy 数值）"""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    # 处理 numpy 整数/浮点数
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)

async def fetch_stock_data(symbol: str, retries: int = 3, base_delay: float = 5.0) -> Optional[Dict[str, Any]]:
    """获取股票数据，带指数退避和随机抖动"""
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            if not info or len(info) == 0 or info.get('regularMarketPrice') is None:
                raise Exception("No market price data returned")
            
            hist = ticker.history(period="5d")
            recent_closes: List[float] = []
            if not hist.empty and 'Close' in hist:
                recent_closes = [round(x, 2) for x in hist['Close'].tolist()]
            
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            previous_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
            day_change = info.get('regularMarketChangePercent')
            volume = info.get('regularMarketVolume')
            market_cap = info.get('marketCap')
            pe_ratio = info.get('trailingPE')
            
            data = {
                "symbol": symbol.upper(),
                "current_price": current_price,
                "previous_close": previous_close,
                "day_change_percent": day_change,
                "volume": volume,
                "market_cap": market_cap,
                "pe_ratio": pe_ratio,
                "recent_5d_close": recent_closes
            }
            return make_json_serializable(data)
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = "Rate limited" in error_msg or "Too Many Requests" in error_msg
            if is_rate_limit:
                logger.warning(f"Rate limited for {symbol} (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    logger.info(f"Waiting {delay:.1f} seconds before retry...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retries failed due to rate limiting for {symbol}")
                    return None
            else:
                logger.warning(f"Attempt {attempt+1}/{retries} failed for {symbol}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(base_delay)
                else:
                    logger.error(f"All retries failed for {symbol}")
                    return None
    return None

async def analyze_with_llm(stock_data: Dict[str, Any]) -> Dict[str, str]:
    if not deepseek_client:
        raise HTTPException(status_code=503, detail="LLM service not configured")
    
    system_prompt = (
        "You are a strict JSON-only financial analyst. "
        "Never output any text, markdown, or explanation. "
        "Only output valid JSON objects."
    )
    
    user_prompt = f"""
Analyze the following stock data and return a JSON object with exactly three fields:
- "summary": a brief assessment (1-2 sentences)
- "sentiment": one of "Bullish", "Bearish", or "Neutral"
- "risk_level": one of "High", "Medium", "Low"

Stock Data:
- Symbol: {stock_data.get('symbol')}
- Current Price: {stock_data.get('current_price')}
- Previous Close: {stock_data.get('previous_close')}
- Day Change %: {stock_data.get('day_change_percent')}
- Volume: {stock_data.get('volume')}
- Market Cap: {stock_data.get('market_cap')}
- P/E Ratio: {stock_data.get('pe_ratio')}
- Recent 5 days closing prices: {stock_data.get('recent_5d_close')}

Output ONLY the JSON object. Example:
{{"summary": "Strong upward trend with high volume.", "sentiment": "Bullish", "risk_level": "Medium"}}
"""
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        required_fields = ["summary", "sentiment", "risk_level"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing field: {field}")
        
        valid_sentiments = ["Bullish", "Bearish", "Neutral"]
        valid_risks = ["High", "Medium", "Low"]
        if result["sentiment"] not in valid_sentiments:
            result["sentiment"] = "Neutral"
        if result["risk_level"] not in valid_risks:
            result["risk_level"] = "Medium"
            
        return result
    except Exception as e:
        logger.error(f"LLM analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {str(e)}")

@app.get("/api/stock/{symbol}")
async def get_stock(symbol: str):
    data = await fetch_stock_data(symbol)
    if not data:
        raise HTTPException(status_code=404, detail="Stock symbol not found or temporarily rate limited. Please try another symbol or wait a few minutes.")
    return data

@app.post("/api/analyze")
async def analyze_stock(req: AnalyzeRequest):
    symbol = req.symbol.upper()
    stock_data = await fetch_stock_data(symbol)
    if not stock_data:
        raise HTTPException(status_code=404, detail="Unable to fetch stock data. Possibly rate limited. Try again later.")
    analysis = await analyze_with_llm(stock_data)
    
    # 存储到 Supabase
    if supabase:
        try:
            # 确保数据可序列化（已处理）
            record = {
                "symbol": stock_data["symbol"],
                "stock_data": stock_data,
                "analysis_result": analysis,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            supabase.table("stock_analyses").insert(record).execute()
            logger.info(f"Stored analysis for {symbol} to Supabase")
        except Exception as e:
            logger.error(f"Supabase storage error: {e}")
    return analysis

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_CONTENT

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Stock Analysis Panel</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        h1 {
            color: #333;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            color: #666;
            margin-bottom: 2rem;
        }
        .input-group {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        input {
            flex: 1;
            padding: 12px 20px;
            font-size: 1rem;
            border: 2px solid #ddd;
            border-radius: 10px;
            transition: border-color 0.3s;
        }
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s, transform 0.1s;
        }
        button:hover {
            background: #5a67d8;
        }
        button:active {
            transform: scale(0.98);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .stock-info {
            background: #f7f9fc;
            border-radius: 15px;
            padding: 1.5rem;
            margin-top: 1rem;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .info-item {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .info-label {
            font-size: 0.85rem;
            color: #888;
            margin-bottom: 0.3rem;
        }
        .info-value {
            font-size: 1.25rem;
            font-weight: 600;
            color: #333;
        }
        .analysis-result {
            background: linear-gradient(135deg, #f5f7fa 0%, #eef2f7 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin-top: 1rem;
        }
        .sentiment {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .sentiment.Bullish { background: #48bb78; color: white; }
        .sentiment.Bearish { background: #f56565; color: white; }
        .sentiment.Neutral { background: #ed8936; color: white; }
        .risk {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .risk.High { background: #c53030; color: white; }
        .risk.Medium { background: #ecc94b; color: #333; }
        .risk.Low { background: #48bb78; color: white; }
        .loading {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
        .error {
            background: #fed7d7;
            color: #c53030;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
        }
        .flex-between {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        hr {
            margin: 1rem 0;
            border: none;
            border-top: 1px solid #e2e8f0;
        }
        @media (max-width: 768px) {
            body { padding: 1rem; }
            .card { padding: 1.5rem; }
            .input-group { flex-direction: column; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>📈 AI Stock Analysis Panel</h1>
        <div class="subtitle">Get real-time stock data and AI-powered insights</div>
        
        <div class="input-group">
            <input type="text" id="symbol" placeholder="Enter stock symbol (e.g., AAPL, MSFT, TSLA)" value="AAPL">
            <button id="fetchBtn" onclick="fetchStock()">Get Quote</button>
            <button id="analyzeBtn" onclick="analyzeStock()" style="background:#48bb78;">🤖 AI Analyze</button>
        </div>
        
        <div id="stockSection" style="display: none;">
            <h3>📊 Stock Information</h3>
            <div id="stockData" class="stock-info"></div>
        </div>
        
        <div id="analysisSection" style="display: none;">
            <h3>🧠 AI Analysis</h3>
            <div id="analysisResult" class="analysis-result"></div>
        </div>
        
        <div id="errorMsg" class="error" style="display: none;"></div>
    </div>
</div>

<script>
    let currentSymbol = '';
    
    async function fetchStock() {
        const symbolInput = document.getElementById('symbol');
        const symbol = symbolInput.value.trim().toUpperCase();
        if (!symbol) {
            showError('Please enter a stock symbol');
            return;
        }
        currentSymbol = symbol;
        
        const fetchBtn = document.getElementById('fetchBtn');
        fetchBtn.disabled = true;
        fetchBtn.textContent = 'Loading...';
        hideError();
        
        try {
            const response = await fetch(`/api/stock/${symbol}`);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Stock not found');
            }
            const data = await response.json();
            displayStockData(data);
            document.getElementById('stockSection').style.display = 'block';
        } catch (error) {
            showError(error.message);
            document.getElementById('stockSection').style.display = 'none';
        } finally {
            fetchBtn.disabled = false;
            fetchBtn.textContent = 'Get Quote';
        }
    }
    
    function displayStockData(data) {
        const formatNumber = (num) => {
            if (num === null || num === undefined) return 'N/A';
            if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
            if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
            return num.toLocaleString();
        };
        
        const changeClass = data.day_change_percent >= 0 ? 'color: #48bb78;' : 'color: #f56565;';
        const changeSymbol = data.day_change_percent >= 0 ? '+' : '';
        
        const html = `
            <div class="info-grid">
                <div class="info-item"><div class="info-label">Symbol</div><div class="info-value">${data.symbol}</div></div>
                <div class="info-item"><div class="info-label">Current Price</div><div class="info-value">$${data.current_price?.toFixed(2) || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">Day Change</div><div class="info-value" style="${changeClass}">${changeSymbol}${data.day_change_percent?.toFixed(2) || 'N/A'}%</div></div>
                <div class="info-item"><div class="info-label">Previous Close</div><div class="info-value">$${data.previous_close?.toFixed(2) || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">Volume</div><div class="info-value">${formatNumber(data.volume)}</div></div>
                <div class="info-item"><div class="info-label">Market Cap</div><div class="info-value">$${formatNumber(data.market_cap)}</div></div>
                <div class="info-item"><div class="info-label">P/E Ratio</div><div class="info-value">${data.pe_ratio?.toFixed(2) || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">5-Day Close</div><div class="info-value">${data.recent_5d_close?.join(' → ') || 'N/A'}</div></div>
            </div>
        `;
        document.getElementById('stockData').innerHTML = html;
    }
    
    async function analyzeStock() {
        const symbolInput = document.getElementById('symbol');
        let symbol = symbolInput.value.trim().toUpperCase();
        if (!symbol) {
            showError('Please enter a stock symbol');
            return;
        }
        currentSymbol = symbol;
        
        const analyzeBtn = document.getElementById('analyzeBtn');
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        hideError();
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: symbol })
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Analysis failed');
            }
            const analysis = await response.json();
            displayAnalysis(analysis);
            document.getElementById('analysisSection').style.display = 'block';
            await fetchStock();
        } catch (error) {
            showError(error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '🤖 AI Analyze';
        }
    }
    
    function displayAnalysis(analysis) {
        const sentimentClass = analysis.sentiment;
        const riskClass = analysis.risk_level;
        
        const html = `
            <div class="flex-between" style="margin-bottom: 1rem;">
                <div><strong>Sentiment:</strong> <span class="sentiment ${sentimentClass}">${analysis.sentiment}</span></div>
                <div><strong>Risk Level:</strong> <span class="risk ${riskClass}">${analysis.risk_level}</span></div>
            </div>
            <hr>
            <div><strong>Summary:</strong></div>
            <div style="margin-top: 0.5rem; line-height: 1.6;">${analysis.summary}</div>
        `;
        document.getElementById('analysisResult').innerHTML = html;
    }
    
    function showError(msg) {
        const errorDiv = document.getElementById('errorMsg');
        errorDiv.textContent = msg;
        errorDiv.style.display = 'block';
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
    
    function hideError() {
        document.getElementById('errorMsg').style.display = 'none';
    }
    
    window.addEventListener('load', () => {
        fetchStock();
    });
</script>
</body>
</html>
"""