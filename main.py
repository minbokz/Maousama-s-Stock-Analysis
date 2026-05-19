import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Stock Analysis Panel")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Supabase 初始化 ----------
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None

if not supabase_url or not supabase_key:
    logger.warning("Supabase credentials missing. Storage disabled.")
else:
    try:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        supabase = None

# ---------- Groq 初始化 ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if not GROQ_API_KEY:
    logger.warning("Groq API key missing. LLM analysis disabled.")
else:
    try:
        groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        groq_client = None

# ---------- yfinance 符号标准化 ----------
def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if symbol.endswith(('.SS', '.SZ', '.HK')):
        return symbol
    if symbol.startswith('SH.'):
        return symbol[3:] + '.SS'
    if symbol.startswith('SZ.'):
        return symbol[3:] + '.SZ'
    if symbol.isdigit() and len(symbol) == 6:
        if symbol.startswith(('6', '9')):
            return f"{symbol}.SS"
        else:
            return f"{symbol}.SZ"
    if symbol.isdigit() and len(symbol) == 4:
        return f"{symbol}.HK"
    if symbol.isdigit() and len(symbol) < 4:
        return symbol.zfill(4) + ".HK"
    return symbol

def _fetch_stock_data_sync(symbol: str) -> Optional[Dict[str, Any]]:
    yf_symbol = normalize_symbol(symbol)
    logger.info(f"Fetching data for {yf_symbol} using yfinance")

    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None):
            logger.warning(f"No info data for {yf_symbol}, trying to use history only")
        
        hist = ticker.history(period="1mo")
        if hist.empty:
            logger.error(f"No historical data for {yf_symbol}")
            return None

        company_name = info.get('longName') or info.get('shortName') or ''
        
        current_price = info.get('regularMarketPrice') or info.get('currentPrice')
        if current_price is None and not hist.empty:
            current_price = hist['Close'].iloc[-1]
        
        previous_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
        if previous_close is None and len(hist) >= 2:
            previous_close = hist['Close'].iloc[-2]
        
        if current_price is None or previous_close is None:
            logger.error(f"Insufficient price data for {yf_symbol}")
            return None

        day_change_percent = ((current_price - previous_close) / previous_close) * 100
        
        volume = info.get('regularMarketVolume')
        if volume is None and not hist.empty:
            volume = int(hist['Volume'].iloc[-1])
        
        market_cap = info.get('marketCap')
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        
        hist_sorted = hist.sort_index()
        recent_hist = hist_sorted.tail(30)
        historical_prices = [
            {"date": date.strftime("%Y-%m-%d"), "close": round(row['Close'], 2)}
            for date, row in recent_hist.iterrows()
        ]
        
        last_5 = hist_sorted.tail(5)
        recent_5d_close = [round(val, 2) for val in last_5['Close'].tolist()]
        
        return {
            "symbol": symbol.upper(),
            "company_name": company_name,
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2),
            "day_change_percent": round(day_change_percent, 2),
            "volume": volume,
            "market_cap": market_cap,
            "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
            "recent_5d_close": recent_5d_close,
            "historical_prices": historical_prices
        }
    except Exception as e:
        logger.error(f"yfinance error for {yf_symbol}: {e}")
        return None

async def fetch_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    return await asyncio.to_thread(_fetch_stock_data_sync, symbol)

# ---------- 请求模型 ----------
class AnalyzeRequest(BaseModel):
    symbol: str
    lang: Optional[str] = "en"   # 'en' or 'zh'

class SaveStockRequest(BaseModel):
    symbol: str

# ---------- Groq LLM 分析（支持中英文）----------
async def analyze_with_llm(stock_data: Dict[str, Any], lang: str = "en") -> Dict[str, str]:
    if not groq_client:
        raise HTTPException(status_code=503, detail="LLM service not configured (Groq API key missing)")

    system_prompt = (
        "You are a strict JSON-only financial analyst. "
        "Never output any text, markdown, or explanation. "
        "Only output valid JSON objects."
    )

    if lang == "zh":
        user_prompt = f"""
分析以下股票数据，返回一个包含三个字段的JSON对象：
- "summary": 简短评估（1-2句话，使用中文）
- "sentiment": 从 "Bullish", "Bearish", "Neutral" 中选择一个（保持英文）
- "risk_level": 从 "High", "Medium", "Low" 中选择一个（保持英文）

股票数据：
- 代码: {stock_data.get('symbol')}
- 公司: {stock_data.get('company_name')}
- 现价: {stock_data.get('current_price')}
- 昨收: {stock_data.get('previous_close')}
- 日涨跌幅 %: {stock_data.get('day_change_percent')}
- 成交量: {stock_data.get('volume')}
- 市值: {stock_data.get('market_cap')}
- 市盈率: {stock_data.get('pe_ratio')}
- 最近5日收盘价: {stock_data.get('recent_5d_close')}

仅输出JSON对象，示例：
{{"summary": "强势上涨趋势，成交量放大。", "sentiment": "Bullish", "risk_level": "Medium"}}
"""
    else:
        user_prompt = f"""
Analyze the following stock data and return a JSON object with exactly three fields:
- "summary": a brief assessment (1-2 sentences, in English)
- "sentiment": one of "Bullish", "Bearish", or "Neutral"
- "risk_level": one of "High", "Medium", or "Low"

Stock Data:
- Symbol: {stock_data.get('symbol')}
- Company: {stock_data.get('company_name')}
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
        response = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
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

# ---------- API 端点 ----------
@app.get("/api/stock/{symbol}")
async def get_stock(symbol: str):
    data = await fetch_stock_data(symbol)
    if not data or data.get('current_price') is None:
        raise HTTPException(status_code=404, detail="Stock symbol not found. Please try another symbol.")
    return data

@app.post("/api/analyze")
async def analyze_stock(req: AnalyzeRequest):
    symbol = req.symbol.upper()
    lang = req.lang or "en"
    stock_data = await fetch_stock_data(symbol)
    if not stock_data or stock_data.get('current_price') is None:
        raise HTTPException(status_code=404, detail="Unable to fetch stock data.")
    analysis = await analyze_with_llm(stock_data, lang)

    if supabase is None:
        logger.error("Supabase client not available – storage skipped.")
        raise HTTPException(status_code=500, detail="Supabase is not configured. Cannot store analysis.")
    
    try:
        record = {
            "symbol": stock_data["symbol"],
            "stock_data": stock_data,
            "analysis_result": analysis,
            "created_at": datetime.now(timezone.utc).isoformat()
            # 移除了 "language" 字段，避免表结构不匹配
        }
        supabase.table("stock_analyses").insert(record).execute()
        logger.info(f"Stored analysis for {symbol} to Supabase")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Supabase storage error: {error_msg}")
        if "row-level security policy" in error_msg.lower():
            raise HTTPException(
                status_code=403,
                detail="Supabase Row Level Security (RLS) is enabled. Please disable RLS for 'stock_analyses' table in Supabase dashboard."
            )
        raise HTTPException(status_code=500, detail=f"Failed to store analysis: {error_msg}")
    
    return analysis

@app.post("/api/save_stock")
async def save_stock_data(req: SaveStockRequest):
    symbol = req.symbol.upper()
    stock_data = await fetch_stock_data(symbol)
    if not stock_data or stock_data.get('current_price') is None:
        raise HTTPException(status_code=404, detail="Unable to fetch stock data.")

    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase is not configured. Cannot save data.")
    
    try:
        record = {
            "symbol": stock_data["symbol"],
            "stock_data": stock_data,
            "saved_at": datetime.now(timezone.utc).isoformat()
        }
        supabase.table("stock_snapshots").insert(record).execute()
        logger.info(f"Manually saved stock snapshot for {symbol} to Supabase")
        return {"status": "success", "message": f"Stock data for {symbol} saved successfully."}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Supabase save error: {error_msg}")
        if "row-level security policy" in error_msg.lower():
            raise HTTPException(
                status_code=403,
                detail="Supabase Row Level Security (RLS) is enabled. Please disable RLS for 'stock_snapshots' table."
            )
        raise HTTPException(status_code=500, detail=f"Failed to save stock data: {error_msg}")

@app.get("/health")
async def health():
    yfinance_ok = False
    try:
        test = yf.Ticker("AAPL")
        _ = test.info
        yfinance_ok = True
    except Exception:
        yfinance_ok = False
    
    deps = {
        "supabase": supabase is not None,
        "groq": groq_client is not None,
        "yfinance": yfinance_ok
    }
    return {"status": "ok", "dependencies": deps}

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_CONTENT

# ---------- 前端 HTML（带中英文切换）----------
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Stock Analysis Panel</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.2);
        }
        .header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 1rem; }
        h1 { color: #333; margin-bottom: 0.5rem; }
        .lang-switch { display: flex; gap: 10px; }
        .lang-btn {
            background: #f0f0f0;
            border: none;
            padding: 6px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }
        .lang-btn.active {
            background: #667eea;
            color: white;
        }
        .subtitle { color: #666; margin-bottom: 2rem; }
        .input-group { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
        input {
            flex: 1;
            padding: 12px 20px;
            font-size: 1rem;
            border: 2px solid #ddd;
            border-radius: 10px;
            transition: border-color 0.3s;
        }
        input:focus { outline: none; border-color: #667eea; }
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
        button:hover { background: #5a67d8; }
        button:active { transform: scale(0.98); }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .stock-info, .analysis-result {
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
        .info-label { font-size: 0.85rem; color: #888; margin-bottom: 0.3rem; }
        .info-value { font-size: 1.25rem; font-weight: 600; color: #333; }
        .sentiment, .risk {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .sentiment.Bullish { background: #48bb78; color: white; }
        .sentiment.Bearish { background: #f56565; color: white; }
        .sentiment.Neutral { background: #ed8936; color: white; }
        .risk.High { background: #c53030; color: white; }
        .risk.Medium { background: #ecc94b; color: #333; }
        .risk.Low { background: #48bb78; color: white; }
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
            flex-wrap: wrap;
            gap: 1rem;
        }
        hr { margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0; }
        .chart-container {
            margin-top: 2rem;
            padding: 1rem;
            background: white;
            border-radius: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        #priceChart { width: 100%; height: 400px; }
        .save-btn { background: #48bb78; }
        .save-btn:hover { background: #38a169; }
        @media (max-width: 768px) {
            body { padding: 1rem; }
            .card { padding: 1.5rem; }
            .input-group { flex-direction: column; }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
</head>
<body>
<div class="container">
    <div class="card">
        <div class="header">
            <div>
                <h1 id="pageTitle">📈 AI Stock Analysis Panel</h1>
                <div id="pageSubtitle" class="subtitle">Get real-time stock data and AI-powered insights</div>
            </div>
            <div class="lang-switch">
                <button id="langZhBtn" class="lang-btn">中文</button>
                <button id="langEnBtn" class="lang-btn active">English</button>
            </div>
        </div>
        
        <div class="input-group">
            <input type="text" id="symbol" placeholder='Enter stock code (e.g., 600000, 000001, sh.600000)'>
            <button id="fetchBtn">Get Quote</button>
            <button id="analyzeBtn" style="background:#48bb78;">🤖 AI Analyze</button>
            <button id="saveBtn" class="save-btn">💾 Save to Supabase</button>
        </div>
        
        <div id="stockSection" style="display: none;">
            <h3 id="stockInfoTitle">📊 Stock Information</h3>
            <div id="stockData" class="stock-info"></div>
        </div>

        <div id="chartSection" style="display: none;">
            <div class="chart-container">
                <h3 id="chartTitle">📉 30-Day Price Trend</h3>
                <div id="priceChart"></div>
            </div>
        </div>
        
        <div id="analysisSection" style="display: none;">
            <h3 id="analysisTitle">🧠 AI Analysis</h3>
            <div id="analysisResult" class="analysis-result"></div>
        </div>
        
        <div id="errorMsg" class="error" style="display: none;"></div>
    </div>
</div>

<script>
    // ---------- 多语言资源 ----------
    const i18n = {
        en: {
            pageTitle: "📈 AI Stock Analysis Panel",
            pageSubtitle: "Get real-time stock data and AI-powered insights",
            stockInfoTitle: "📊 Stock Information",
            chartTitle: "📉 30-Day Price Trend",
            analysisTitle: "🧠 AI Analysis",
            fetchBtn: "Get Quote",
            analyzeBtn: "🤖 AI Analyze",
            saveBtn: "💾 Save to Supabase",
            symbolPlaceholder: "Enter stock code (e.g., 600000, 000001, sh.600000)",
            company: "Company",
            symbol: "Symbol",
            currentPrice: "Current Price",
            dayChange: "Day Change",
            prevClose: "Previous Close",
            volume: "Volume",
            marketCap: "Market Cap",
            peRatio: "P/E Ratio",
            recent5d: "5-Day Close",
            sentiment_bullish: "Bullish",
            sentiment_bearish: "Bearish",
            sentiment_neutral: "Neutral",
            risk_high: "High",
            risk_medium: "Medium",
            risk_low: "Low",
            noData: "No historical price data available.",
            errorDefault: "An error occurred. Please try again.",
            saveSuccess: "Stock data saved successfully!",
            loading: "Loading...",
            analyzing: "Analyzing...",
            saving: "Saving..."
        },
        zh: {
            pageTitle: "📈 AI 股票分析面板",
            pageSubtitle: "获取实时股票数据和 AI 智能分析",
            stockInfoTitle: "📊 股票信息",
            chartTitle: "📉 近30日价格走势",
            analysisTitle: "🧠 AI 分析",
            fetchBtn: "获取行情",
            analyzeBtn: "🤖 AI 分析",
            saveBtn: "💾 保存到 Supabase",
            symbolPlaceholder: "请输入股票代码 (例: 600000, 000001, sh.600000)",
            company: "公司名称",
            symbol: "代码",
            currentPrice: "当前价格",
            dayChange: "日涨跌幅",
            prevClose: "昨日收盘",
            volume: "成交量",
            marketCap: "市值",
            peRatio: "市盈率",
            recent5d: "近5日收盘价",
            sentiment_bullish: "看涨",
            sentiment_bearish: "看跌",
            sentiment_neutral: "中性",
            risk_high: "高",
            risk_medium: "中",
            risk_low: "低",
            noData: "暂无历史价格数据",
            errorDefault: "发生错误，请重试",
            saveSuccess: "股票数据保存成功！",
            loading: "加载中...",
            analyzing: "分析中...",
            saving: "保存中..."
        }
    };

    let currentLocale = 'en';
    let currentSymbol = '';
    let currentStockData = null;
    let chartInstance = null;

    const langZhBtn = document.getElementById('langZhBtn');
    const langEnBtn = document.getElementById('langEnBtn');
    const fetchBtn = document.getElementById('fetchBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const saveBtn = document.getElementById('saveBtn');
    const symbolInput = document.getElementById('symbol');
    const pageTitle = document.getElementById('pageTitle');
    const pageSubtitle = document.getElementById('pageSubtitle');
    const stockInfoTitle = document.getElementById('stockInfoTitle');
    const chartTitle = document.getElementById('chartTitle');
    const analysisTitle = document.getElementById('analysisTitle');
    const errorDiv = document.getElementById('errorMsg');
    const stockSection = document.getElementById('stockSection');
    const chartSection = document.getElementById('chartSection');
    const analysisSection = document.getElementById('analysisSection');

    function updateUILabels() {
        const t = i18n[currentLocale];
        pageTitle.innerText = t.pageTitle;
        pageSubtitle.innerText = t.pageSubtitle;
        stockInfoTitle.innerText = t.stockInfoTitle;
        chartTitle.innerText = t.chartTitle;
        analysisTitle.innerText = t.analysisTitle;
        fetchBtn.innerText = t.fetchBtn;
        analyzeBtn.innerText = t.analyzeBtn;
        saveBtn.innerText = t.saveBtn;
        symbolInput.placeholder = t.symbolPlaceholder;
    }

    function displayStockData(data) {
        const t = i18n[currentLocale];
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
                <div class="info-item"><div class="info-label">${t.company}</div><div class="info-value">${data.company_name || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">${t.symbol}</div><div class="info-value">${data.symbol}</div></div>
                <div class="info-item"><div class="info-label">${t.currentPrice}</div><div class="info-value">¥${data.current_price?.toFixed(2) || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">${t.dayChange}</div><div class="info-value" style="${changeClass}">${changeSymbol}${data.day_change_percent?.toFixed(2) || 'N/A'}%</div></div>
                <div class="info-item"><div class="info-label">${t.prevClose}</div><div class="info-value">¥${data.previous_close?.toFixed(2) || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">${t.volume}</div><div class="info-value">${formatNumber(data.volume)}</div></div>
                <div class="info-item"><div class="info-label">${t.marketCap}</div><div class="info-value">¥${formatNumber(data.market_cap)}</div></div>
                <div class="info-item"><div class="info-label">${t.peRatio}</div><div class="info-value">${data.pe_ratio?.toFixed(2) || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">${t.recent5d}</div><div class="info-value">${data.recent_5d_close?.join(' → ') || 'N/A'}</div></div>
            </div>
        `;
        document.getElementById('stockData').innerHTML = html;
        
        if (chartInstance) {
            const option = chartInstance.getOption();
            option.title[0].text = t.chartTitle.replace('📉 ', '');
            chartInstance.setOption(option);
        }
    }

    function displayAnalysis(analysis) {
        const t = i18n[currentLocale];
        let sentimentText = analysis.sentiment;
        if (currentLocale === 'zh') {
            if (analysis.sentiment === 'Bullish') sentimentText = t.sentiment_bullish;
            else if (analysis.sentiment === 'Bearish') sentimentText = t.sentiment_bearish;
            else sentimentText = t.sentiment_neutral;
        }
        let riskText = analysis.risk_level;
        if (currentLocale === 'zh') {
            if (analysis.risk_level === 'High') riskText = t.risk_high;
            else if (analysis.risk_level === 'Medium') riskText = t.risk_medium;
            else riskText = t.risk_low;
        }
        const sentimentClass = analysis.sentiment;
        const riskClass = analysis.risk_level;
        
        const html = `
            <div class="flex-between" style="margin-bottom: 1rem;">
                <div><strong>Sentiment:</strong> <span class="sentiment ${sentimentClass}">${sentimentText}</span></div>
                <div><strong>Risk Level:</strong> <span class="risk ${riskClass}">${riskText}</span></div>
            </div>
            <hr>
            <div><strong>Summary:</strong></div>
            <div style="margin-top: 0.5rem; line-height: 1.6;">${analysis.summary}</div>
        `;
        document.getElementById('analysisResult').innerHTML = html;
    }

    function renderPriceChart(historicalPrices) {
        if (!historicalPrices || historicalPrices.length === 0) {
            const chartDom = document.getElementById('priceChart');
            if (chartDom) chartDom.innerHTML = `<div style="text-align:center;padding:2rem;">${i18n[currentLocale].noData}</div>`;
            return;
        }
        const chartDom = document.getElementById('priceChart');
        if (!chartDom) return;
        if (chartInstance) chartInstance.dispose();
        chartInstance = echarts.init(chartDom);
        const dates = historicalPrices.map(item => item.date);
        const closes = historicalPrices.map(item => item.close);
        const option = {
            title: { text: i18n[currentLocale].chartTitle.replace('📉 ', ''), left: 'center' },
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            xAxis: { type: 'category', data: dates, axisLabel: { rotate: 45, interval: 'auto' } },
            yAxis: { type: 'value', name: currentLocale === 'zh' ? '价格 (¥)' : 'Price (¥)', axisLabel: { formatter: '¥{value}' } },
            series: [{
                data: closes, type: 'line', smooth: false,
                lineStyle: { color: '#667eea', width: 3 },
                areaStyle: { opacity: 0.1, color: '#667eea' },
                symbol: 'circle', symbolSize: 6, itemStyle: { color: '#764ba2' }
            }],
            grid: { containLabel: true, left: '10%', right: '5%', top: '15%', bottom: '10%' }
        };
        chartInstance.setOption(option);
        window.addEventListener('resize', () => { if (chartInstance) chartInstance.resize(); });
    }

    async function fetchStock() {
        const symbol = symbolInput.value.trim().toUpperCase();
        if (!symbol) {
            showError(i18n[currentLocale].errorDefault);
            return;
        }
        currentSymbol = symbol;
        fetchBtn.disabled = true;
        fetchBtn.textContent = i18n[currentLocale].loading;
        hideError();
        
        try {
            const response = await fetch(`/api/stock/${symbol}`);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Stock not found');
            }
            const data = await response.json();
            currentStockData = data;
            displayStockData(data);
            stockSection.style.display = 'block';
            chartSection.style.display = 'block';
            renderPriceChart(data.historical_prices || []);
        } catch (error) {
            showError(error.message);
            stockSection.style.display = 'none';
            chartSection.style.display = 'none';
            analysisSection.style.display = 'none';
        } finally {
            fetchBtn.disabled = false;
            fetchBtn.textContent = i18n[currentLocale].fetchBtn;
        }
    }

    async function analyzeStock() {
        let symbol = symbolInput.value.trim().toUpperCase();
        if (!symbol) {
            showError(i18n[currentLocale].errorDefault);
            return;
        }
        currentSymbol = symbol;
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = i18n[currentLocale].analyzing;
        hideError();
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: symbol, lang: currentLocale })
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Analysis failed');
            }
            const analysis = await response.json();
            displayAnalysis(analysis);
            analysisSection.style.display = 'block';
            await fetchStock();
        } catch (error) {
            showError(error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = i18n[currentLocale].analyzeBtn;
        }
    }

    async function saveStockData() {
        let symbol = symbolInput.value.trim().toUpperCase();
        if (!symbol) {
            showError(i18n[currentLocale].errorDefault);
            return;
        }
        saveBtn.disabled = true;
        saveBtn.textContent = i18n[currentLocale].saving;
        hideError();

        try {
            const response = await fetch('/api/save_stock', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: symbol })
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Save failed');
            }
            alert(i18n[currentLocale].saveSuccess);
        } catch (error) {
            showError(error.message);
        } finally {
            saveBtn.disabled = false;
            saveBtn.textContent = i18n[currentLocale].saveBtn;
        }
    }

    async function setLanguage(lang) {
        if (lang === currentLocale) return;
        currentLocale = lang;
        if (lang === 'zh') {
            langZhBtn.classList.add('active');
            langEnBtn.classList.remove('active');
        } else {
            langEnBtn.classList.add('active');
            langZhBtn.classList.remove('active');
        }
        updateUILabels();
        
        if (currentStockData) {
            displayStockData(currentStockData);
            if (chartInstance) {
                const option = chartInstance.getOption();
                option.title[0].text = i18n[currentLocale].chartTitle.replace('📉 ', '');
                option.yAxis.name = currentLocale === 'zh' ? '价格 (¥)' : 'Price (¥)';
                chartInstance.setOption(option);
            }
        }
        if (analysisSection.style.display === 'block' && currentSymbol) {
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: currentSymbol, lang: currentLocale })
                });
                if (response.ok) {
                    const analysis = await response.json();
                    displayAnalysis(analysis);
                }
            } catch (err) {
                console.warn("Failed to re-analyze after language change", err);
            }
        }
    }

    function showError(msg) {
        errorDiv.textContent = msg;
        errorDiv.style.display = 'block';
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
    
    function hideError() {
        errorDiv.style.display = 'none';
    }

    fetchBtn.onclick = fetchStock;
    analyzeBtn.onclick = analyzeStock;
    saveBtn.onclick = saveStockData;
    langZhBtn.onclick = () => setLanguage('zh');
    langEnBtn.onclick = () => setLanguage('en');
    
    window.addEventListener('load', () => {
        updateUILabels();
        console.log('Ready. Language switching enabled.');
    });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)