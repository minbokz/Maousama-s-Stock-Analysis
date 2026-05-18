import os
import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
import baostock as bs
from openai import AsyncOpenAI

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

class SaveStockRequest(BaseModel):
    symbol: str

def format_stock_code(symbol: str) -> tuple[Optional[str], Optional[str]]:
    """将用户输入的股票代码转换为 BaoStock 格式，同时返回市场标识"""
    symbol = symbol.upper().strip()
    if symbol.startswith(('SH.', 'SH', 'SZ.', 'SZ')):
        if '.' in symbol:
            parts = symbol.split('.')
            if len(parts) == 2:
                market = parts[0].lower()
                code = parts[1]
                return f"{market}.{code}", market
        if symbol.startswith('SH'):
            code = symbol[2:]
            return f"sh.{code}", "sh"
        elif symbol.startswith('SZ'):
            code = symbol[2:]
            return f"sz.{code}", "sz"
    if symbol.isdigit() and len(symbol) == 6:
        if symbol.startswith(('6', '9')):
            return f"sh.{symbol}", "sh"
        else:
            return f"sz.{symbol}", "sz"
    return None, None

async def fetch_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """使用 BaoStock 获取股票数据，包含公司名称、市盈率、流通市值及30天历史数据"""
    bs_code, market = format_stock_code(symbol)
    if not bs_code:
        logger.warning(f"Invalid stock code format: {symbol}")
        return None

    def sync_fetch():
        lg = bs.login()
        if lg.error_code != '0':
            logger.error(f"BaoStock login failed: {lg.error_msg}")
            return None

        try:
            # ---------- 1. 获取公司名称 ----------
            company_name = None
            rs_basic = bs.query_stock_basic(code=bs_code)
            basic_df = rs_basic.get_data()
            if not basic_df.empty and 'code_name' in basic_df.columns:
                company_name = basic_df.iloc[0]['code_name']

            # ---------- 2. 获取最新交易日 ----------
            latest_trade_day = None
            for offset in range(10):
                check_date = (datetime.now() - timedelta(days=offset)).strftime('%Y-%m-%d')
                rs_trade = bs.query_trade_dates(start_date=check_date, end_date=check_date)
                trade_df = rs_trade.get_data()
                if not trade_df.empty and trade_df.iloc[0]['is_trading_day'] == '1':
                    latest_trade_day = check_date
                    break
            if not latest_trade_day:
                latest_trade_day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

            # ---------- 3. 获取最近30天K线数据（价格、成交量、市盈率等）----------
            rs_k = bs.query_history_k_data_plus(
                bs_code,
                "date,code,close,preclose,volume,pctChg,peTTM",
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=latest_trade_day,
                frequency="d",
                adjustflag="3"
            )
            k_df = rs_k.get_data()
            if k_df.empty:
                logger.error(f"No K-line data for {bs_code}")
                return None

            latest = k_df.iloc[-1]
            current_price = float(latest['close']) if latest['close'] else None
            previous_close = float(latest['preclose']) if latest['preclose'] else None
            day_change = float(latest['pctChg']) if latest['pctChg'] else None
            volume = int(latest['volume']) if latest['volume'] else None

            pe_ratio = None
            if latest['peTTM'] and latest['peTTM'] != '':
                try:
                    pe_ratio = float(latest['peTTM'])
                except (ValueError, TypeError):
                    pass

            # ---------- 4. 获取最新财报中的总股本和流通股本（单位：股） ----------
            total_shares = None      # 总股本（股）
            liqa_shares = None       # 流通股本（股）

            # 尝试获取当前年份4个季度的数据，优先使用有数据的最近季度
            now_year = datetime.now().year
            profit_data = None
            for year in [now_year, now_year - 1]:
                for quarter in [4, 3, 2, 1]:   # 按季度优先级：4→3→2→1
                    rs_profit = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                    df = rs_profit.get_data()
                    if not df.empty:
                        profit_data = df.iloc[0]
                        break
                if profit_data is not None:
                    break

            if profit_data is not None:
                # 注意：Baostock 返回的 totalShare / liqaShare 单位可能是“股”（通过实测确认）
                if 'totalShare' in profit_data and profit_data['totalShare'] not in ('', None, '--'):
                    try:
                        total_shares = float(profit_data['totalShare'])
                    except (ValueError, TypeError):
                        pass
                if 'liqaShare' in profit_data and profit_data['liqaShare'] not in ('', None, '--'):
                    try:
                        liqa_shares = float(profit_data['liqaShare'])
                    except (ValueError, TypeError):
                        pass

            # 如果流通股本获取失败，回退到总股本（并记录警告）
            effective_shares = liqa_shares if liqa_shares is not None else total_shares
            if effective_shares is None:
                logger.warning(f"Could not retrieve any share capital for {bs_code}, market cap will be None")
            else:
                if liqa_shares is None:
                    logger.warning(f"Using total shares instead of float shares for {bs_code} – market cap may be overestimated")

            # 计算流通市值（元）
            market_cap = None
            if current_price is not None and effective_shares is not None:
                market_cap = round(current_price * effective_shares, 0)

            # 最近5日收盘价
            recent_5d_close = []
            for i in range(min(5, len(k_df))):
                close_val = k_df.iloc[-1 - i]['close']
                if close_val:
                    recent_5d_close.append(round(float(close_val), 2))
            recent_5d_close.reverse()

            historical_prices = []
            for _, row in k_df.iterrows():
                date_str = row['date']
                close_val = row['close']
                if close_val:
                    historical_prices.append({
                        "date": date_str,
                        "close": round(float(close_val), 2)
                    })

            return {
                "symbol": symbol.upper(),
                "company_name": company_name,
                "current_price": current_price,
                "previous_close": previous_close,
                "day_change_percent": day_change,
                "volume": volume,
                "market_cap": market_cap,
                "pe_ratio": pe_ratio,
                "recent_5d_close": recent_5d_close,
                "historical_prices": historical_prices
            }
        except Exception as e:
            logger.error(f"BaoStock fetch error: {e}")
            return None
        finally:
            bs.logout()

    try:
        result = await asyncio.to_thread(sync_fetch)
        if result:
            for key, value in result.items():
                if value is None or (isinstance(value, float) and value != value):
                    result[key] = None
        return result
    except Exception as e:
        logger.error(f"Async fetch error: {e}")
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
    if not data or data.get('current_price') is None:
        raise HTTPException(status_code=404, detail="Stock symbol not found. Please try another symbol.")
    return data

@app.post("/api/analyze")
async def analyze_stock(req: AnalyzeRequest):
    symbol = req.symbol.upper()
    stock_data = await fetch_stock_data(symbol)
    if not stock_data or stock_data.get('current_price') is None:
        raise HTTPException(status_code=404, detail="Unable to fetch stock data.")
    analysis = await analyze_with_llm(stock_data)

    if supabase is None:
        logger.error("Supabase client not available – storage skipped.")
        raise HTTPException(status_code=500, detail="Supabase is not configured. Cannot store analysis.")
    
    try:
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
        raise HTTPException(status_code=500, detail=f"Failed to store analysis: {str(e)}")
    
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
        logger.error(f"Supabase save error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save stock data: {str(e)}")

@app.get("/health")
async def health():
    deps = {
        "supabase": supabase is not None,
        "deepseek": deepseek_client is not None,
        "baostock": True
    }
    return {"status": "ok", "dependencies": deps}

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_CONTENT

# ---------- 前端 HTML ----------
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
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
        h1 { color: #333; margin-bottom: 0.5rem; }
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
        <h1>📈 AI Stock Analysis Panel</h1>
        <div class="subtitle">Get real-time stock data and AI-powered insights (A股市场)</div>
        
        <div class="input-group">
            <input type="text" id="symbol" placeholder="Enter stock code (e.g., 600000, 000001, sh.600000)">
            <button id="fetchBtn" onclick="fetchStock()">Get Quote</button>
            <button id="analyzeBtn" onclick="analyzeStock()" style="background:#48bb78;">🤖 AI Analyze</button>
            <button id="saveBtn" onclick="saveStockData()" class="save-btn">💾 Save to Supabase</button>
        </div>
        
        <div id="stockSection" style="display: none;">
            <h3>📊 Stock Information</h3>
            <div id="stockData" class="stock-info"></div>
        </div>

        <div id="chartSection" style="display: none;">
            <div class="chart-container">
                <h3>📉 30-Day Price Trend</h3>
                <div id="priceChart"></div>
            </div>
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
    let currentStockData = null;
    let chartInstance = null;

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
            currentStockData = data;
            displayStockData(data);
            
            document.getElementById('stockSection').style.display = 'block';
            document.getElementById('chartSection').style.display = 'block';
            
            // 延迟渲染图表以确保容器宽度正确
            setTimeout(() => {
                renderPriceChart(data.historical_prices || []);
            }, 0);
        } catch (error) {
            showError(error.message);
            document.getElementById('stockSection').style.display = 'none';
            document.getElementById('chartSection').style.display = 'none';
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
                <div class="info-item"><div class="info-label">Company</div><div class="info-value">${data.company_name || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">Symbol</div><div class="info-value">${data.symbol}</div></div>
                <div class="info-item"><div class="info-label">Current Price</div><div class="info-value">¥${data.current_price?.toFixed(2) || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">Day Change</div><div class="info-value" style="${changeClass}">${changeSymbol}${data.day_change_percent?.toFixed(2) || 'N/A'}%</div></div>
                <div class="info-item"><div class="info-label">Previous Close</div><div class="info-value">¥${data.previous_close?.toFixed(2) || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">Volume</div><div class="info-value">${formatNumber(data.volume)}</div></div>
                <div class="info-item"><div class="info-label">Market Cap</div><div class="info-value">¥${formatNumber(data.market_cap)}</div></div>
                <div class="info-item"><div class="info-label">P/E Ratio</div><div class="info-value">${data.pe_ratio?.toFixed(2) || 'N/A'}</div></div>
                <div class="info-item"><div class="info-label">5-Day Close</div><div class="info-value">${data.recent_5d_close?.join(' → ') || 'N/A'}</div></div>
            </div>
        `;
        document.getElementById('stockData').innerHTML = html;
    }

    function renderPriceChart(historicalPrices) {
        if (!historicalPrices || historicalPrices.length === 0) {
            const chartDom = document.getElementById('priceChart');
            if (chartDom) chartDom.innerHTML = '<div style="text-align:center;padding:2rem;">No historical price data available.</div>';
            return;
        }
        const chartDom = document.getElementById('priceChart');
        if (!chartDom) return;
        if (chartInstance) chartInstance.dispose();
        chartInstance = echarts.init(chartDom);
        const dates = historicalPrices.map(item => item.date);
        const closes = historicalPrices.map(item => item.close);
        const option = {
            title: { text: 'Closing Price Trend (Last 30 Days)', left: 'center' },
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            xAxis: { type: 'category', data: dates, axisLabel: { rotate: 45, interval: 'auto' } },
            yAxis: { type: 'value', name: 'Price (¥)', axisLabel: { formatter: '¥{value}' } },
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
            await fetchStock();  // 刷新最新数据
        } catch (error) {
            showError(error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '🤖 AI Analyze';
        }
    }

    async function saveStockData() {
        const symbolInput = document.getElementById('symbol');
        let symbol = symbolInput.value.trim().toUpperCase();
        if (!symbol) {
            showError('Please enter a stock symbol to save');
            return;
        }
        const saveBtn = document.getElementById('saveBtn');
        saveBtn.disabled = true;
        saveBtn.textContent = 'Saving...';
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
            const result = await response.json();
            alert(result.message || 'Stock data saved successfully!');
        } catch (error) {
            showError(error.message);
        } finally {
            saveBtn.disabled = false;
            saveBtn.textContent = '💾 Save to Supabase';
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
        console.log('Ready. Please enter a stock symbol.');
    });
</script>
</body>
</html>
"""