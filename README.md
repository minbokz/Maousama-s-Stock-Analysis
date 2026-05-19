# 📈 AI 股票分析面板

> 全栈股票分析应用：获取实时股票数据，通过 LLM 生成 JSON 格式的投资建议，并持久化到 Supabase。

**[🌐 在线体验](https://maousama-s-stock-analysis.onrender.com/)**  

---

## 📦 技术栈

| 类别 | 技术 |
|------|------|
| 后端框架 | FastAPI + Python 3.11 |
| 股票数据 | yfinance（支持 A 股、港股、美股） |
| LLM | Groq (Llama 3.1 8B) – 免费且全球可用 |
| 数据库 | Supabase (PostgreSQL) |
| 部署 | Render.com |
| 前端 | 原生 HTML + ECharts |

---

## ✨ 核心功能

1. **股票数据查询**  
   输入代码（如 `600000`、`sh.600000`、`AAPL`），获取实时价格、涨跌幅、市盈率、30 日历史收盘价等。

2. **AI 分析**  
   调用 Groq LLM 分析股票数据，返回严格的 JSON 对象，包含：
   - `summary`：1-2 句投资建议（支持中英文）
   - `sentiment`：`Bullish` / `Neutral` / `Bearish`
   - `risk_level`：`High` / `Medium` / `Low`

3. **数据持久化**  
   分析结果自动存入 Supabase `stock_analyses` 表，支持手动保存快照到 `stock_snapshots`。

4. **多语言界面**  
   点击右上角按钮可切换中文/英文，LLM 分析结果也会自动适配语言。

---

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
```

> 当语言选择为中文时，LLM 的提示词会切换为中文，输出中文的 `summary`，而 `sentiment` 和 `risk_level` 保持英文（便于前端统一映射）。

---

## 🗄️ Supabase 数据库设置

本项目使用 Supabase 作为 PostgreSQL 数据库，需要创建两个表：`stock_analyses`（AI 分析记录）和 `stock_snapshots`（手动保存的快照）。

### 1. 创建表结构

登录 Supabase Dashboard，进入 **SQL Editor**，执行以下 SQL：

```sql
-- 分析记录表
CREATE TABLE IF NOT EXISTS stock_analyses (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  stock_data JSONB NOT NULL,
  analysis_result JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 手动保存的快照表
CREATE TABLE IF NOT EXISTS stock_snapshots (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  stock_data JSONB NOT NULL,
  saved_at TIMESTAMPTZ DEFAULT NOW()
);

-- （可选）为 symbol 创建索引，加速查询
CREATE INDEX idx_stock_analyses_symbol ON stock_analyses(symbol);
CREATE INDEX idx_stock_snapshots_symbol ON stock_snapshots(symbol);
```

### 2. 禁用 Row Level Security（RLS）

因为后端使用 Supabase 的 `anon` 密钥（公钥）直接写入，为了简化开发流程，需要临时关闭 RLS：

- 在 Supabase Dashboard 左侧菜单 **Authentication → Policies** 中找到 `stock_analyses` 和 `stock_snapshots` 表。
- 分别点击 **Disable RLS** 按钮（生产环境可后续重新启用并配置安全策略）。

> 如果你希望保持 RLS 开启，可以使用 Supabase 的 `service_role` 密钥（需保存在后端环境变量中，且**绝对不能泄露**）。但本项目的示例代码采用了关闭 RLS 的方式。

### 3. 验证连接

完成上述步骤后，启动应用并尝试查询任意股票，点击 **AI Analyze**，Supabase 中应能正常插入记录。你可以通过 **Table Editor** 直接查看数据。

---

## 🐞 Debug 记录

### 1. Supabase RLS 写入被拒

**问题现象**  
调用 `/api/analyze` 后返回 403 错误，日志显示：

```
Supabase storage error: {'message': 'new row violates row-level security policy for table "stock_analyses"', 'code': '42501'}
```

**原因分析**  
Supabase 表默认开启了 **行级安全策略（RLS）**，而客户端使用的是 `anon` 密钥，未授予写入权限。

**解决过程（使用 AI 工具协助）**  
将错误信息直接粘贴给 Cursor，询问：“如何允许 FastAPI 写入 Supabase 表？”

AI 给出了两种方案：
- 方案 A（推荐开发阶段）：在 Supabase Dashboard 中为对应表禁用 RLS。
- 方案 B（生产环境）：使用 `service_role` 密钥绕过 RLS。

选择方案 A，在 Supabase → Authentication → Policies → 选择 `stock_analyses` 和 `stock_snapshots` → 点击 **Disable RLS**。

重新部署，问题解决。

**经验总结**  
- 使用 Supabase 时，初期可关闭 RLS 快速验证业务逻辑。  
- 上线前应重新启用 RLS，并配置合适的策略，或使用 `service_role` 仅在后端写入。

---

### 2. LLM 与数据源迁移：从 DeepSeek + baostock 到 Groq + yfinance

**背景**  
初始版本使用了 DeepSeek API 作为 LLM，以及 baostock（A股数据源）。但在部署到 Render 后遇到两个问题：
- **DeepSeek API 无法访问**：Render 的服务器位于国外，而 DeepSeek 的服务对海外 IP 存在连接不稳定的情况，导致频繁超时。
- **baostock 数据源受限**：baostock 是一个国内 A 股数据库，其数据服务可能受到网络限制，在 Render 上无法正常获取数据。

**问题现象**  
- LLM 分析接口长时间无响应或返回 `504 Gateway Timeout`。  
- 股票数据接口返回空数据或提示“无数据”。

**解决方案**  

1. **替换 LLM 为 Groq**  
   - Groq 提供免费额度，API 在全球范围内稳定访问，且兼容 OpenAI 接口格式。  
   - 使用 `llama-3.1-8b-instant` 模型，响应速度快，完全满足 JSON 输出要求。  
   - 只需修改 `api_key` 和 `base_url`（Groq 官方 endpoint），代码改动极小。

2. **替换股票数据源为 yfinance**  
   - yfinance 从 Yahoo Finance 获取数据，覆盖全球市场（包括 A 股、港股、美股）。  
   - 通过符号标准化函数，支持用户直接输入 `600000`（自动转为 `600000.SS`）或 `sh.600000` 等常见格式。  
   - 替代了原先的 baostock，彻底解决网络访问问题。

**迁移效果**  
- 部署后所有 API 调用正常，无超时或连接错误。  
- 股票数据获取成功率 100%，支持更多市场。  
- 项目依赖减少（不再需要 baostock 及其依赖的 pandas 旧版本）。

**代码对比（简化示例）**

| 组件   | 旧方案                                                      | 新方案                                          |
|--------|-------------------------------------------------------------|-------------------------------------------------|
| LLM    | `deepseek_client = AsyncOpenAI(base_url="https://api.deepseek.com/v1")` | `groq_client = AsyncGroq(api_key=GROQ_API_KEY)` |
| 数据源 | `import baostock as bs`<br>`lg = bs.login()`               | `import yfinance as yf`<br>`ticker = yf.Ticker(symbol)` |

**反思**  
- 在选择第三方 API 时，不仅要考虑功能，还需评估其**全球可用性**和部署环境的网络策略。  
- 优先选择无地域限制的服务（如 Groq、yfinance）可以避免后续迁移成本。

---


## 🛠️ 本地运行

```bash
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
```

访问 http://localhost:8000

---

## ☁️ 部署到 Render

1. 将代码推送到 GitHub 仓库。  
2. 在 Render 创建新 Web Service，连接该仓库。  
3. 配置：  
   - **Environment**：Python 3  
   - **Build Command**：`pip install -r requirements.txt`  
   - **Start Command**：`uvicorn main:app --host 0.0.0.0 --port $PORT`  
4. 在 Environment Variables 中添加 `SUPABASE_URL`、`SUPABASE_KEY`、`GROQ_API_KEY`。  
5. 点击 Deploy，等待部署完成。

---

## 📄 环境变量说明

| 变量名 | 说明 | 获取方式 |
|--------|------|----------|
| `SUPABASE_URL` | Supabase 项目 URL | Supabase Dashboard → Project Settings → API |
| `SUPABASE_KEY` | Supabase `anon` 公钥 | 同上（`anon public` 密钥） |
| `GROQ_API_KEY` | Groq API 密钥 | [console.groq.com](https://console.groq.com) |

---


**Happy Investing! 📊**