# main.py
import os
import json
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import yfinance as yf
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)s

# 初始化 FastAPI
app = FastAPI(title="AI Stock Analysis Panel")

# 初始化 Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    logger.warning("Supabase credentials missing. Storage disabled.")
    supabase: Optional[Client] = None
else:
    supabase = create_client(supabase_url, supabase_key)

# 初始化 OpenAI (Async)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OpenAI API key missing. LLM analysis disabled.")
    openai_client = None
else:
    openai_client = AsyncOpenAI(api_key=openai_api_key)

# 请求模型
class AnalyzeRequest(BaseModel):
    symbol: str

# 辅助函数：获取股票数据（内部使用）
async def fetch_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """使用 yfinance 获取股票行情数据"""
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        
        # 验证符号是否存在
        if not info or len(info) == 0 or info.get('regularMarketPrice') is None:
            return None
        
        # 获取最近5天收盘价
        hist = ticker.history(period="5d")
        recent_closes: List[float] = []
        if not hist.empty and 'Close' in hist:
            recent_closes = [round(x, 2) for x in hist['Close'].tolist()]
        
        # 提取关键指标
        current_price = info.get('regularMarketPrice') or info.get('currentPrice')
        previous_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
        day_change = info.get('regularMarketChangePercent')
        volume = info.get('regularMarketVolume')
        market_cap = info.get('marketCap')
        pe_ratio = info.get('trailingPE')
        
        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "previous_close": previous_close,
            "day_change_percent": day_change,
            "volume": volume,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "recent_5d_close": recent_closes
        }
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        return None

# 辅助函数：调用 LLM 分析（强制 JSON）
async def analyze_with_llm(stock_data: Dict[str, Any]) -> Dict[str, str]:
    """调用 OpenAI API 并强制返回严格 JSON 格式"""
    if not openai_client:
        raise HTTPException(status_code=503, detail="LLM service not configured")
    
    # 构建 prompt - 重点：要求只输出 JSON，不添加任何额外文本
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
        # 使用 response_format 强制 JSON (OpenAI 新版本)
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}  # 关键：强制 JSON 输出
        )
        
        content = response.choices[0].message.content
        # 尝试解析 JSON
        result = json.loads(content)
        
        # 验证必需字段
        required_fields = ["summary", "sentiment", "risk_level"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing field: {field}")
        
        # 验证 sentiment 和 risk_level 的值
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

# API 端点：获取股票行情数据
@app.get("/api/stock/{symbol}")
async def get_stock(symbol: str):
    """获取实时股票数据"""
    data = await fetch_stock_data(symbol)
    if not data:
        raise HTTPException(status_code=404, detail="Stock symbol not found or invalid")
    return data

# API 端点：AI 分析并存储到 Supabase
@app.post("/api/analyze")
async def analyze_stock(req: AnalyzeRequest):
    """执行 AI 分析并将结果存入 Supabase"""
    symbol = req.symbol.upper()
    
    # 1. 获取最新股票数据
    stock_data = await fetch_stock_data(symbol)
    if not stock_data:
        raise HTTPException(status_code=404, detail="Stock symbol not found or invalid")
    
    # 2. 调用 LLM 分析
    analysis = await analyze_with_llm(stock_data)
    
    # 3. 存储到 Supabase (如果配置了)
    if supabase:
        try:
            record = {
                "symbol": stock_data["symbol"],
                "stock_data": stock_data,
                "analysis_result": analysis,
                "created_at": "now()"
            }
            # 注意：需要提前创建表 stock_analyses
            supabase.table("stock_analyses").insert(record).execute()
            logger.info(f"Stored analysis for {symbol} to Supabase")
        except Exception as e:
            logger.error(f"Supabase storage error: {e}")
            # 不中断请求，继续返回分析结果
    
    # 4. 返回分析结果
    return analysis

# 健康检查
@app.get("/health")
async def health():
    return {"status": "ok"}

# 根路径：返回前端 HTML 界面
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_CONTENT

# 前端 HTML 内容
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
        
        // UI loading state
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
            
            // Also fetch fresh stock data to keep in sync
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
    
    // Auto-fetch on page load with default symbol
    window.addEventListener('load', () => {
        fetchStock();
    });
</script>
</body>
</html>
"""