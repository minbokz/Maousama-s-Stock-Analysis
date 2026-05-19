# Maousama-s-Stock-Analysis
# AI 股票分析面板

> 全栈股票分析应用：获取实时股票数据，通过 LLM 生成 JSON 格式的投资建议，并持久化到 Supabase。

## 🚀 在线访问

**[点击体验 →](https://your-app.onrender.com)**  
（请将 `your-app.onrender.com` 替换为你的 Render 实际域名）

## 📦 技术栈

- **后端**：FastAPI + Python 3.11
- **股票数据**：yfinance（支持 A 股、港股、美股）
- **LLM**：Groq (Llama 3.1 8B) – 免费且全球可用
- **数据库**：Supabase (PostgreSQL)
- **部署**：Render.com
- **前端**：原生 HTML + ECharts

## ✨ 核心功能

1. **股票数据查询**  
   输入代码（如 `600000`、`sh.600000`、`AAPL`），获取实时价格、涨跌幅、市盈率、30 日历史收盘价等。

2. **AI 分析**  
   调用 Groq LLM 分析股票数据，返回严格的 JSON 对象，包含：
   - `summary`：1-2 句投资建议
   - `sentiment`：`Bullish` / `Neutral` / `Bearish`
   - `risk_level`：`High` / `Medium` / `Low`

3. **数据持久化**  
   分析结果自动存入 Supabase `stock_analyses` 表，支持手动保存快照到 `stock_snapshots`。

## 📷 Prompt 设计（强制 JSON 输出）

为了确保 LLM **只输出纯 JSON，没有任何额外文本**，我们使用了系统提示 + `response_format` 参数：

```python
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

# 调用 Groq（支持 response_format）
response = await groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.0,
    response_format={"type": "json_object"}  # 强制 JSON
)
Prompt 截图
上述代码即为实际使用的 Prompt。该提示词明确禁止输出任何非 JSON 内容，并配合 response_format 参数，确保 LLM 绝不“乱说话”。

🐞 Debug 记录：Supabase RLS 写入被拒
问题现象
调用 /api/analyze 后返回 403 错误，日志显示：

text
Supabase storage error: {'message': 'new row violates row-level security policy for table "stock_analyses"', 'code': '42501'}
原因分析
Supabase 表默认开启了 行级安全策略（RLS），而客户端使用的是 anon 密钥，未授予写入权限。

解决过程（使用 AI 工具协助）

将错误信息直接粘贴给 Cursor，询问：“如何允许 FastAPI 写入 Supabase 表？”

AI 给出了两种方案：

方案 A（推荐开发阶段）：在 Supabase Dashboard 中为对应表禁用 RLS。

方案 B（生产环境）：使用 service_role 密钥绕过 RLS。

选择方案 A，在 Supabase → Authentication → Policies → 选择 stock_analyses 和 stock_snapshots → 点击 Disable RLS。

重新部署，问题解决。

经验总结

使用 Supabase 时，初期可关闭 RLS 快速验证业务逻辑。

上线前应重新启用 RLS，并配置合适的策略，或使用 service_role 仅在后端写入。

🛠️ 本地运行
bash
# 克隆仓库
git clone https://github.com/yourusername/ai-stock-panel.git
cd ai-stock-panel

# 安装依赖
pip install -r requirements.txt

# 配置环境变量（创建 .env 文件）
SUPABASE_URL=你的Supabase项目URL
SUPABASE_KEY=你的Supabase anon/public key
GROQ_API_KEY=你的Groq API密钥

# 启动服务
python main.py
访问 http://localhost:8000

☁️ 部署到 Render
将代码推送到 GitHub 仓库。

在 Render 创建新 Web Service，连接该仓库。

配置：

Environment：Python 3

Build Command：pip install -r requirements.txt

Start Command：uvicorn main:app --host 0.0.0.0 --port $PORT

在 Environment Variables 中添加 SUPABASE_URL、SUPABASE_KEY、GROQ_API_KEY。

点击 Deploy，等待部署完成。
