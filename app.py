import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv

# 引入自訂模組
from database import (
    init_db, get_db_stats, get_latest_month, get_latest_pe_date, get_latest_quarter,
    get_monthly_revenues_with_pe, get_quarterly_financials_list, get_refined_industries_map,
    get_gemini_report, save_gemini_report, get_connection, save_monthly_revenues,
    save_daily_pes, save_quarterly_financials, get_gemini_report_details
)
from crawler import (
    fetch_monthly_revenue, fetch_daily_pe, fetch_quarterly_financials
)
from gemini_service import (
    refine_stock_industries, analyze_industry_outliers,
    analyze_monthly_market_trends, analyze_quarterly_financial_trends, get_gemini_model,
    analyze_investor_conferences, analyze_turnaround_stocks
)

# 載入環境變數
load_dotenv()

# 設定 Streamlit 頁面參數
st.set_page_config(
    page_title="Antigravity 台股基本面 AI 分析儀",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化資料庫
init_db()

# 初始化對話框個股選取狀態
if 'active_dialog_stock' not in st.session_state:
    st.session_state.active_dialog_stock = None

# --- 個股深度透視與 K 線繪製函數 ---

def get_yfinance_data(stock_code):
    import yfinance as yf
    ticker = f"{stock_code}.TW"
    df = yf.download(ticker, period="6mo", progress=False, timeout=8)
    if df.empty or len(df) < 5:
        ticker = f"{stock_code}.TWO"
        df = yf.download(ticker, period="6mo", progress=False, timeout=8)
    if not df.empty:
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    return df, ticker

def draw_k_line_chart(stock_code):
    df_stock, ticker = get_yfinance_data(stock_code)
    if df_stock.empty or len(df_stock) < 5:
        st.warning(f"無法取得 {stock_code} 的 K 線數據。")
        return None
        
    # 計算均線
    df_stock['5MA'] = df_stock['Close'].rolling(window=5).mean()
    df_stock['10MA'] = df_stock['Close'].rolling(window=10).mean()
    df_stock['20MA'] = df_stock['Close'].rolling(window=20).mean()
    df_stock['60MA'] = df_stock['Close'].rolling(window=60).mean()

    fig = go.Figure()
    
    # 增加日 K 線
    fig.add_trace(go.Candlestick(
        x=df_stock.index,
        open=df_stock['Open'],
        high=df_stock['High'],
        low=df_stock['Low'],
        close=df_stock['Close'],
        name='日K線',
        increasing_line_color='#E53E3E', # 紅漲 (TW Stock UP)
        decreasing_line_color='#2F855A'  # 綠跌 (TW Stock DOWN)
    ))
    
    # 增加均線 Trace (使用與使用者上傳圖形高度相似的配色系統)
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['5MA'], mode='lines', name='5MA', line=dict(color='#FFA500', width=1.2)))      # 橘色
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['10MA'], mode='lines', name='10MA', line=dict(color='#00BFFF', width=1.2)))  # 淺藍
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['20MA'], mode='lines', name='20MA', line=dict(color='#FF1493', width=1.5)))  # 粉紅
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['60MA'], mode='lines', name='60MA', line=dict(color='#228B22', width=2)))    # 綠色
    
    fig.update_layout(
        title=f"{stock_code} 過去半年日 K 線與均線圖 ({ticker})",
        xaxis_title="日期",
        yaxis_title="股價 (TWD)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#1E293B',
        xaxis_rangeslider_visible=False,
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#F1F5F9')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9')
    return fig


def draw_k_line_chart_with_52w(stock_code):
    import yfinance as yf
    import pandas as pd
    import plotly.graph_objects as go
    
    ticker = f"{stock_code}.TW"
    df = yf.download(ticker, period="1y", progress=False, timeout=8)
    if df.empty or len(df) < 5:
        ticker = f"{stock_code}.TWO"
        df = yf.download(ticker, period="1y", progress=False, timeout=8)
    if df.empty:
        st.warning(f"無法取得 {stock_code} 的 K 線數據。")
        return None
        
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    
    # 計算均線
    df['5MA'] = df['Close'].rolling(window=5).mean()
    df['10MA'] = df['Close'].rolling(window=10).mean()
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['60MA'] = df['Close'].rolling(window=60).mean()
    
    # 52週高點 (以過去 250 天最大值計算)
    high_52w = float(df['High'].max())
    
    # 僅顯示過去半年 (約120天)
    df_plot = df.iloc[-120:]
    
    fig = go.Figure()
    
    # 增加日 K 線
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name='日K線',
        increasing_line_color='#E53E3E',
        decreasing_line_color='#2F855A'
    ))
    
    # 增加均線
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['5MA'], mode='lines', name='5MA', line=dict(color='#FFA500', width=1.2)))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['10MA'], mode='lines', name='10MA', line=dict(color='#00BFFF', width=1.2)))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['20MA'], mode='lines', name='20MA', line=dict(color='#FF1493', width=1.5)))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['60MA'], mode='lines', name='60MA', line=dict(color='#228B22', width=2)))
    
    # 增加 52W 高點水平線
    fig.add_hline(
        y=high_52w, 
        line_dash="dash", 
        line_color="#E53E3E", 
        line_width=1.5,
        annotation_text=f"52W 高點: {high_52w} 元", 
        annotation_position="top left",
        annotation_font_color="#E53E3E"
    )
    
    fig.update_layout(
        title=f"{stock_code} 實時均線與 52W 高點 K 線圖 ({ticker})",
        xaxis_title="日期",
        yaxis_title="股價 (TWD)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#1E293B',
        xaxis_rangeslider_visible=False,
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#F1F5F9')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9')
    return fig

@st.cache_data(ttl=1800)
def get_ma_convergence_table_data(report_text):
    from gemini_service import extract_valid_stock_codes, check_ma_convergence_batch
    valid_codes = extract_valid_stock_codes(report_text)
    if not valid_codes:
        return []
        
    batch_results = check_ma_convergence_batch(valid_codes)
    
    rows = []
    for code in valid_codes:
        # 獲取股票名稱
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT stock_name FROM monthly_revenue WHERE stock_code = ? LIMIT 1", (code,))
        r = cursor.fetchone()
        conn.close()
        name = r['stock_name'] if r else "未知"
        res_val = batch_results.get(code)
        if res_val:
            # 增加防禦性檢查，避免因為快取或版本不一致導致 4 元素與 5 元素元組解包失敗
            if len(res_val) == 4:
                success, spread, mas, price = res_val
                is_20ma_rising = False
            else:
                success, spread, mas, price, is_20ma_rising = res_val
            if success:
                # 決定燈號
                if spread < 3.0:
                    status = "🟢 強烈糾結 (<3%)"
                elif spread <= 5.0:
                    status = "🟡 輕微糾結 (3%-5%)"
                else:
                    status = "⚪ 未糾結 (>5%)"
                    
                trend_str = "📈 向上" if is_20ma_rising else "📉 向下或持平"
                
                rows.append({
                    "股票代號": code,
                    "股票名稱": name,
                    "即時股價": f"{price} 元",
                    "5MA": f"{mas.get('5MA')} 元",
                    "10MA": f"{mas.get('10MA')} 元",
                    "20MA": f"{mas.get('20MA')} 元",
                    "60MA": f"{mas.get('60MA')} 元",
                    "均線價差比 (Spread)": f"{spread}%",
                    "狀態": status,
                    "20MA 趨勢": trend_str
                })
    return rows

def render_verification_table(report_text, title="📊 均線糾結度量化驗證表 (實時數據)"):
    rows = get_ma_convergence_table_data(report_text)
    if rows:
        st.markdown(f"### {title}")
        st.caption("計算公式: Spread = (Max(5MA, 10MA, 20MA, 60MA) - Min(5MA, 10MA, 20MA, 60MA)) / 目前股價 * 100。數值越低代表均線越糾結整理。20MA 趨勢判定今日與5天前均線斜率。")
        df_verify = pd.DataFrame(rows)
        # 用來點擊可選取個股看 K 線
        st.dataframe(df_verify, use_container_width=True)
    else:
        st.info("報告中未提及任何已知的台股代碼，或暫時無法獲取均線數據。")

def clear_active_dialog_stock():
    st.session_state.active_dialog_stock = None
    if 'df_key_suffix' not in st.session_state:
        st.session_state.df_key_suffix = 0
    st.session_state.df_key_suffix += 1

def on_stock_picker_change():
    if 'detail_stock_select' in st.session_state:
        picker_val = st.session_state.detail_stock_select
        if picker_val != "請選擇個股...":
            parts = picker_val.split()
            if len(parts) >= 2:
                st.session_state.active_dialog_stock = (parts[0], parts[1])
            # 重設選取狀態
            st.session_state.detail_stock_select = "請選擇個股..."

def style_positive_red_negative_green(val):
    if pd.isna(val):
        return ''
    if isinstance(val, (int, float)):
        if val > 0:
            return 'color: #DC2626; font-weight: bold;'
        elif val < 0:
            return 'color: #16A34A; font-weight: bold;'
    return ''

@st.dialog("個股深度透視與基本面大解析", width="large", on_dismiss=clear_active_dialog_stock)
def show_stock_detail_dialog(stock_code, stock_name):
    st.markdown(f"## 🔍 {stock_code} {stock_name}")
    
    api_key = st.session_state.get('api_key_input')
    tab_ai, tab_tech, tab_fin = st.tabs(["🔮 Gemini AI 聯網深度解析", "📈 技術分析 (日 K 線)", "📊 歷年財務數據"])
    
    with tab_ai:
        # 先從資料庫獲取快取
        cached_report = get_gemini_report('stock_details', stock_code)
        
        report_placeholder = st.empty()
        
        if cached_report:
            report_placeholder.markdown(cached_report)
            st.write("---")
            if st.button("🔄 重新分析並更新 (即時更新)", key=f"re_analyze_{stock_code}"):
                if not api_key:
                    st.error("請先在側邊欄配置您的 Gemini API Key！")
                else:
                    with st.spinner("Gemini 正在搜尋最新資訊並編寫報告..."):
                        from gemini_service import get_stock_details_from_gemini
                        report = get_stock_details_from_gemini(api_key, stock_code, stock_name)
                        if "失敗" not in report and "未設定" not in report and "error" not in report.lower():
                            save_gemini_report('stock_details', stock_code, report)
                            report_placeholder.markdown(report)
                            st.success("✔ 報告已成功更新！")
                        else:
                            st.error(handle_gemini_error(report))
        else:
            if not api_key:
                st.warning("⚠️ 此個股目前無快取報告。請在側邊欄配置您的 Gemini API Key 以啟用聯網 AI 即時深度解析！")
            else:
                with st.spinner("Gemini 正在搜尋最新法說、小作文、題材、新聞並編寫報告，請稍候... (這大約需要 15-20 秒)"):
                    from gemini_service import get_stock_details_from_gemini
                    report = get_stock_details_from_gemini(api_key, stock_code, stock_name)
                    if "失敗" not in report and "未設定" not in report and "error" not in report.lower():
                        save_gemini_report('stock_details', stock_code, report)
                        report_placeholder.markdown(report)
                        st.success("✔ 報告分析完成！")
                    else:
                        st.error(handle_gemini_error(report))
            
    with tab_tech:
        with st.spinner("正在加載 K 線數據..."):
            fig = draw_k_line_chart(stock_code)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
    with tab_fin:
        # 檢查是否需要自動補全歷史季度數據 (少於 4 季則觸發 backfill)
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM quarterly_financials WHERE stock_code = ?', (stock_code,))
            existing_count = cursor.fetchone()['count']
        except Exception:
            existing_count = 0
        conn.close()
        
        if existing_count < 4:
            with st.spinner("正在從 Yahoo Finance 補全歷史季度財報數據 (2025Q4, 2025Q3 等)..."):
                from database import backfill_quarterly_financials_yfinance
                backfill_quarterly_financials_yfinance(stock_code)
                
        conn = get_connection()
        df_stock_q = pd.read_sql('''
            SELECT year, quarter, revenue, gross_profit, net_profit, eps, gross_margin, net_margin 
            FROM quarterly_financials 
            WHERE stock_code = ? 
            ORDER BY year DESC, quarter DESC 
            LIMIT 4
        ''', conn, params=(stock_code,))
        df_stock_m = pd.read_sql('''
            SELECT date_month, revenue, yoy, mom 
            FROM monthly_revenue 
            WHERE stock_code = ? 
            ORDER BY date_month DESC 
            LIMIT 6
        ''', conn, params=(stock_code,))
        conn.close()
        
        c_fin1, c_fin2 = st.columns(2)
        with c_fin1:
            st.markdown("#### 📅 最近四季季報")
            if df_stock_q.empty:
                st.info("資料庫中無此個股季報數據。")
            else:
                df_q_show = df_stock_q.copy()
                df_q_show['季度'] = df_q_show.apply(lambda r: f"{int(r['year'])} Q{int(r['quarter'])}", axis=1)
                df_q_show = df_q_show[['季度', 'eps', 'gross_margin', 'net_margin', 'revenue']]
                df_q_show.columns = ['季度', 'EPS (元)', '毛利率 (%)', '淨利率 (%)', '營收(千元)']
                
                styled_q = df_q_show.style.map(
                    style_positive_red_negative_green,
                    subset=['EPS (元)']
                ).format({
                    'EPS (元)': '{:+.2f}',
                    '毛利率 (%)': '{:.2f}%',
                    '淨利率 (%)': '{:.2f}%',
                    '營收(千元)': '{:,.0f}'
                }, na_rep='-')
                st.dataframe(styled_q, use_container_width=True, hide_index=True)
        with c_fin2:
            st.markdown("#### 📅 最近六個月營收")
            if df_stock_m.empty:
                st.info("資料庫中無此個股月營收數據。")
            else:
                df_m_show = df_stock_m.copy()
                df_m_show.columns = ['月份', '營收(千元)', '年增率 YoY (%)', '月增率 MoM (%)']
                
                styled_m = df_m_show.style.map(
                    style_positive_red_negative_green,
                    subset=['年增率 YoY (%)', '月增率 MoM (%)']
                ).format({
                    '營收(千元)': '{:,.0f}',
                    '年增率 YoY (%)': '{:+.1f}%',
                    '月增率 MoM (%)': '{:+.1f}%'
                }, na_rep='-')
                st.dataframe(styled_m, use_container_width=True, hide_index=True)

def handle_gemini_error(error_msg):
    if "429" in error_msg or "quota" in error_msg.lower() or "limit" in error_msg.lower():
        return f"⚠️ **Gemini API 額度或頻率已達上限 (429 Rate Limit/Quota Exceeded)**\n\n您的 API 金鑰已超出其所屬層級的頻率限制（免費版為每分鐘 15 次，付費版依帳戶層級而定）。請**稍等 1-2 分鐘**後再次嘗試。\n\n**原始錯誤訊息：**\n`{error_msg}`"
    return error_msg

# --- 自訂 CSS 樣式 (進階亮色玻璃擬態) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Noto+Sans+TC:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
    /* 全域字體與亮色背景設定 */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', 'Noto Sans TC', sans-serif;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    
    /* 玻璃擬態卡片 (亮色主題) */
    .stCard {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.04) !important;
    }
    
    /* 螢幕頂部裝飾條 */
    [data-testid="stHeader"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 6px;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        z-index: 999;
    }
    
    /* 標題漸層色 */
    .gradient-text {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .gradient-subtext {
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* 調整 streamlit 預設間距 */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* 側邊欄樣式 */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    /* 數值指標卡片樣式 */
    .metric-value {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
    }
    .metric-delta-pos {
        color: #DC2626 !important; /* 台灣股市紅漲 */
        font-weight: 600;
        font-size: 0.95rem;
    }
    .metric-delta-neg {
        color: #16A34A !important; /* 台灣股市綠跌 */
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* 標題欄設計 */
    h1, h2, h3, h4, h5, h6 {
        color: #1E293B !important;
        font-weight: 600;
    }
    
    p, span, label, div {
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- 側邊欄設定與金鑰配置 ---
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 20px;"><span style="font-size: 3rem;">📈</span></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #1E3A8A; margin-top: 0px;">台股基本面分析儀</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #64748B; font-size: 0.85rem;">由 Gemini 驅動的基本面智慧篩選平台</p>', unsafe_allow_html=True)
    st.write("---")
    
    # 導覽選單
    menu_options = [
        "📊 大盤營收概覽",
        "🔍 同業營收篩選",
        "📈 季報三率分析",
        "🔮 潛力轉盈股分析",
        "🎯 籌碼與技術糾結股",
        "🚀 當月異軍突起股",
        "📈 評等調整追蹤",
        "🤖 Gemini AI 投資顧問",
        "⚙️ 資料管理中心"
    ]
    choice = st.radio("選單導覽", menu_options)
    
    st.write("---")
    st.markdown("### 📅 資料庫最新期數")
    col_s_m = get_latest_month()
    col_s_pe = get_latest_pe_date()
    col_s_y, col_s_q = get_latest_quarter()
    st.caption(f"📅 最新營收月份: **{col_s_m or '無資料'}**")
    st.caption(f"📅 最新本益比日期: **{col_s_pe or '無資料'}**")
    st.caption(f"📅 最新財報季度: **{f'{col_s_y} Q{col_s_q}' if col_s_y else '無資料'}**")
    
    st.write("---")
    st.markdown("### 🔑 API 金鑰配置")
    
    # 優先嘗試從 env 取得
    env_api_key = os.environ.get('GEMINI_API_KEY', '')
    
    # 決定預設金鑰值
    default_key = st.session_state.get('api_key_input', env_api_key)
    
    api_key_input = st.text_input(
        "Gemini API Key", 
        value=default_key, 
        type="password",
        help="輸入您的 Gemini API 金鑰。金鑰將僅存在此 Session 中。"
    )
    
    # 儲存金鑰到 session 狀態與環境變數中
    st.session_state['api_key_input'] = api_key_input
    if api_key_input:
        os.environ['GEMINI_API_KEY'] = api_key_input
        st.success("API Key 已成功套用！")
    else:
        st.warning("請設定 API 金鑰以啟用 AI 功能。")

# --- 共用變數與資料庫載入狀態 ---
stats = get_db_stats()
db_is_empty = (stats.get('monthly_revenue', 0) == 0)

if db_is_empty:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #FFD60A; margin-top:0px;">⚠️ 資料庫中尚無資料</h2>', unsafe_allow_html=True)
    st.write("歡迎使用本系統！目前您的資料庫是空的，需要進行第一次爬蟲以初始化台股數據。")
    st.write("請依照以下步驟操作：")
    st.write("1. 點擊側邊欄的 **⚙️ 資料管理中心**")
    st.write("2. 點擊 **「更新/爬取最新台股數據」** 按鈕")
    st.write("爬蟲程式將會向證交所與櫃買中心拉取最新的月營收、每日估值（PE/PB/殖利率）及季度財報。抓取完成後即可解鎖所有分析頁面！")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 強制將非資料中心頁面跳轉至資料中心或僅允許停留在資料中心
    if choice != "⚙️ 資料管理中心":
        st.info("請先切換至「資料管理中心」爬取數據。")
        st.stop()

# --- 1. 頁面：大盤營收概覽 ---
if choice == "📊 大盤營收概覽":
    st.markdown('<h1 class="gradient-text">📊 大盤月度營收概覽</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">追蹤全台股上市櫃公司最新月份的營業收入表現，並提供 YoY 與 MoM 的宏觀趨勢分佈。</p>', unsafe_allow_html=True)
    
    # 取得最新月份
    latest_month = get_latest_month()
    
    # 月份選擇器
    # 找出資料庫中所有不重複月份
    conn = get_connection()
    df_months = pd.read_sql('SELECT DISTINCT date_month FROM monthly_revenue ORDER BY date_month DESC', conn)
    conn.close()
    
    selected_month = st.selectbox("選擇分析月份", df_months['date_month'].tolist(), index=0)
    
    if selected_month:
        # 載入當月資料
        raw_data = get_monthly_revenues_with_pe(selected_month)
        df_month = pd.DataFrame(raw_data)
        
        if df_month.empty:
            st.error(f"找不到 {selected_month} 的數據。")
        else:
            # 數據前處理：將空值填補
            df_month['yoy'] = df_month['yoy'].fillna(0)
            df_month['mom'] = df_month['mom'].fillna(0)
            df_month['pe'] = pd.to_numeric(df_month['pe'], errors='coerce')
            df_month['pb'] = pd.to_numeric(df_month['pb'], errors='coerce')
            df_month['dy'] = pd.to_numeric(df_month['dy'], errors='coerce')
            
            # 計算大盤指標
            total_rev = df_month['revenue'].sum()
            total_last_month = df_month['last_month_revenue'].sum()
            total_last_year = df_month['last_year_revenue'].sum()
            
            market_yoy = (total_rev - total_last_year) / total_last_year * 100 if total_last_year else 0
            market_mom = (total_rev - total_last_month) / total_last_month * 100 if total_last_month else 0
            
            # 篩選成長股數量 (YoY > 0)
            growth_stocks_count = len(df_month[df_month['yoy'] > 0])
            total_stocks = len(df_month)
            growth_ratio = (growth_stocks_count / total_stocks * 100) if total_stocks else 0
            
            # 顯示大盤 Metric 卡片
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="stCard">
                    <div style="color: #64748B; font-size: 0.9rem;">全市場總營收</div>
                    <div class="metric-value">{total_rev/100000.0:,.2f} 億元</div>
                    <div style="color: #64748B; font-size: 0.8rem;">已申報上市櫃公司加總<br><span style="color: #F59E0B; font-size: 0.75rem;">⚠️ 每月 10 號前為實時公告加總</span></div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                delta_class = "metric-delta-pos" if market_yoy >= 0 else "metric-delta-neg"
                sign = "+" if market_yoy >= 0 else ""
                st.markdown(f"""
                <div class="stCard">
                    <div style="color: #64748B; font-size: 0.9rem;">大盤營收 YoY (年增率)</div>
                    <div class="metric-value">{sign}{market_yoy:.2f}%</div>
                    <div class="{delta_class}">與去年同期相比</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                delta_class = "metric-delta-pos" if market_mom >= 0 else "metric-delta-neg"
                sign = "+" if market_mom >= 0 else ""
                st.markdown(f"""
                <div class="stCard">
                    <div style="color: #64748B; font-size: 0.9rem;">大盤營收 MoM (月增率)</div>
                    <div class="metric-value">{sign}{market_mom:.2f}%</div>
                    <div class="{delta_class}">與上月相比</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="stCard">
                    <div style="color: #64748B; font-size: 0.9rem;">營收成長股比例</div>
                    <div class="metric-value">{growth_ratio:.1f}%</div>
                    <div style="color: #16A34A; font-weight:600;">{growth_stocks_count:,} / {total_stocks:,} 檔個股</div>
                </div>
                """, unsafe_allow_html=True)
                
            # 大盤圖表
            st.write("---")
            st.markdown("### 📈 大盤與產業趨勢分析")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown('<div class="stCard">', unsafe_allow_html=True)
                st.markdown("#### 🏆 當月營收成長排行 (YoY Top 10)")
                exclude_construction = st.checkbox("🚫 排除營建股 (建材營造)", value=True, help="營建股常因建案一次性完工入帳導致單月營收爆發，勾選此處可將其篩除，以便觀察其他產業的實質營收成長。")
                
                df_rank_base = df_month[df_month['revenue'] >= 500000]
                if exclude_construction:
                    df_rank_base = df_rank_base[~df_rank_base['original_industry'].str.contains('營建|建材|營造', na=False, regex=True)]
                    
                df_filtered_rank = df_rank_base.sort_values(by='yoy', ascending=False).head(10)
                fig_rank = px.bar(
                    df_filtered_rank,
                    x='yoy',
                    y='stock_name',
                    orientation='h',
                    text='yoy',
                    labels={'yoy': 'YoY (%)', 'stock_name': '股票名稱'},
                    color='yoy',
                    color_continuous_scale='blues',
                    height=400
                )
                fig_rank.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#1E293B',
                    yaxis={'categoryorder': 'total ascending'},
                    coloraxis_showscale=False
                )
                fig_rank.update_xaxes(showgrid=True, gridcolor='#F1F5F9')
                fig_rank.update_yaxes(showgrid=True, gridcolor='#F1F5F9')
                fig_rank.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_rank, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c2:
                st.markdown('<div class="stCard">', unsafe_allow_html=True)
                st.markdown("#### 📊 YoY 營收年增率分佈直方圖")
                # 剔除極端值以利視覺化 (僅看 -100% ~ +200%)
                df_hist = df_month[(df_month['yoy'] >= -100) & (df_month['yoy'] <= 200)]
                fig_hist = px.histogram(
                    df_hist,
                    x='yoy',
                    nbins=50,
                    labels={'yoy': '營收年增率 YoY (%)', 'count': '個股數量'},
                    height=400,
                    color_discrete_sequence=['#1E3A8A']
                )
                fig_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#1E293B',
                    bargap=0.05
                )
                fig_hist.update_xaxes(showgrid=True, gridcolor='#F1F5F9')
                fig_hist.update_yaxes(showgrid=True, gridcolor='#F1F5F9')
                fig_hist.update_yaxes(title_text="個股數量")
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            # 產業別分析
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.markdown("#### 🏢 各產業營收年增率中位數與估值比較")
            # 依原始產業分組
            df_ind = df_month.groupby('original_industry').agg(
                count=('stock_code', 'count'),
                median_yoy=('yoy', 'median'),
                median_mom=('mom', 'median'),
                median_pe=('pe', lambda x: x.median(skipna=True))
            ).reset_index()
            # 過濾掉檔數太少(少於 4 檔)的產業
            df_ind = df_ind[df_ind['count'] >= 4].sort_values(by='median_yoy', ascending=False)
            
            fig_ind = px.bar(
                df_ind,
                x='original_industry',
                y='median_yoy',
                color='median_pe',
                labels={'median_yoy': 'YoY 中位數 (%)', 'original_industry': '產業別', 'median_pe': '本益比中位數'},
                color_continuous_scale='rdbu_r',
                height=500
            )
            fig_ind.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#1E293B',
                xaxis_tickangle=-45
            )
            fig_ind.update_xaxes(showgrid=True, gridcolor='#F1F5F9')
            fig_ind.update_yaxes(showgrid=True, gridcolor='#F1F5F9')
            st.plotly_chart(fig_ind, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- 2. 頁面：同業營收篩選 ---
elif choice == "🔍 同業營收篩選":
    st.markdown('<h1 class="gradient-text">🔍 產業同業營收篩選</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">在同一個產業別中，透過比較 YoY 與 MoM 中位數，篩選出顯著超越同業的優秀個股，並由 Gemini 評估其成長持續性。</p>', unsafe_allow_html=True)
    
    # 選擇月份
    conn = get_connection()
    df_months = pd.read_sql('SELECT DISTINCT date_month FROM monthly_revenue ORDER BY date_month DESC', conn)
    conn.close()
    selected_month = st.selectbox("選擇篩選月份", df_months['date_month'].tolist(), index=0)
    
    # 選擇產業別分類方式：原始產業別 vs. Gemini 精細分類產業別
    refined_map = get_refined_industries_map()
    
    industry_type = st.radio(
        "選擇同業分類類別方式",
        ["原始交易所產業別", "AI 智慧精細次產業別"],
        index=0,
        horizontal=True,
        help="選擇是要使用證交所/櫃買中心的傳統粗放產業分類，還是使用 Gemini AI 進行精確細分後的子產業分類來進行同業比較。"
    )
    use_refined = (industry_type == "AI 智慧精細次產業別")
    
    if selected_month:
        # 讀取當月資料
        raw_data = get_monthly_revenues_with_pe(selected_month)
        df_month = pd.DataFrame(raw_data)
        
        if df_month.empty:
            st.error("此月份無資料。")
        else:
            # 建立產業別欄位
            if use_refined:
                df_month['display_industry'] = df_month['refined_industry'].fillna(df_month['original_industry'])
            else:
                df_month['display_industry'] = df_month['original_industry']
                
            # 取得該月份所有的產業清單
            industries = sorted(df_month['display_industry'].dropna().unique().tolist())
            
            # 初始化 session state 中的選取值
            if 'selected_industry_val' not in st.session_state:
                st.session_state.selected_industry_val = industries[0] if industries else None
            elif st.session_state.selected_industry_val not in industries and industries:
                st.session_state.selected_industry_val = industries[0]
                
            if 'last_selected_stock' not in st.session_state:
                st.session_state.last_selected_stock = "-- 🔍 輸入或選擇個股查詢同業 --"
                
            col_search_stock, col_select_industry = st.columns(2)
            
            with col_search_stock:
                stock_list = ["-- 🔍 輸入或選擇個股查詢同業 --"] + sorted([f"{row['stock_code']} {row['stock_name']}" for _, row in df_month.iterrows()])
                selected_stock_query = st.selectbox(
                    "🔍 透過個股快速定位同業",
                    stock_list,
                    index=0,
                    help="選擇或輸入個股，系統將自動為您選取該個股所屬的同業別進行比較。"
                )
                
            # 當選取的個股與上次不同時，更新產業
            if selected_stock_query != st.session_state.last_selected_stock:
                st.session_state.last_selected_stock = selected_stock_query
                if selected_stock_query != "-- 🔍 輸入或選擇個股查詢同業 --":
                    code = selected_stock_query.split(" ")[0]
                    stock_row = df_month[df_month['stock_code'] == code]
                    if not stock_row.empty:
                        target_ind = stock_row.iloc[0]['display_industry']
                        if target_ind in industries:
                            st.session_state.selected_industry_val = target_ind
                            
            with col_select_industry:
                default_idx = 0
                if st.session_state.selected_industry_val in industries:
                    default_idx = industries.index(st.session_state.selected_industry_val)
                selected_industry = st.selectbox(
                    "選擇要比較的同業別",
                    industries,
                    index=default_idx
                )
                st.session_state.selected_industry_val = selected_industry
            
            if selected_industry:
                # 篩選該產業的股票
                df_ind = df_month[df_month['display_industry'] == selected_industry].copy()
                
                # 計算該同業中位數
                med_yoy = df_ind['yoy'].median()
                med_mom = df_ind['mom'].median()
                med_pe = df_ind['pe'].median()
                
                st.markdown(f"### 📊 同業表現中位數：**{selected_industry}** (共 {len(df_ind)} 檔個股)")
                c1, c2, c3 = st.columns(3)
                c1.metric("同業 YoY 年增率中位數", f"{med_yoy:.1f}%")
                c2.metric("同業 MoM 月增率中位數", f"{med_mom:.1f}%")
                c3.metric("同業 PE 本益比中位數", f"{med_pe:.1f}倍" if pd.notnull(med_pe) else "N/A")
                
                # 篩選出顯著超越同業的個股
                # 超越標準：YoY > 0，且 YoY > med_yoy，且 (YoY > med_yoy + 15 或 MoM > med_mom + 10)
                outliers = df_ind[
                    (df_ind['yoy'] > 0) & 
                    (df_ind['yoy'] > med_yoy) & 
                    (df_ind['mom'] > med_mom) & 
                    ((df_ind['yoy'] > med_yoy + 15) | (df_ind['mom'] > med_mom + 10))
                ].copy()
                
                st.write("---")
                st.markdown("### 🌟 異軍突起個股名單")
                if outliers.empty:
                    st.info("在此篩選條件下，當前月份本產業無顯著超越同業的個股。")
                else:
                    # 顯示亮眼個股的簡要資訊
                    cols = st.columns(min(len(outliers), 4))
                    for idx, (_, row) in enumerate(outliers.head(4).iterrows()):
                        with cols[idx % 4]:
                            pe_val = f"{row['pe']:.1f}x" if pd.notnull(row['pe']) else "N/A"
                            st.markdown(f"""
                            <div class="stCard" style="border: 1px solid rgba(30, 58, 138, 0.15) !important;">
                                <div style="color: #DC2626; font-weight: 700; font-size: 1.2rem;">{row['stock_code']} {row['stock_name']}</div>
                                <div style="color: #64748B; font-size: 0.8rem; margin-bottom: 10px;">{row['original_industry']}</div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span>營收 YoY</span><span style="color: #DC2626; font-weight:600;">+{row['yoy']:.1f}%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span>營收 MoM</span><span style="color: #DC2626; font-weight:600;">+{row['mom']:.1f}%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>PE (估值)</span><span style="color: #1E3A8A; font-weight: 600;">{pe_val}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.write("---")
                # 顯示該產業所有個股列表表單
                st.markdown("### 📋 完整同業營收與估值數據列表")
                
                # 計算個股營運/轉虧為盈狀態
                codes = df_ind['stock_code'].tolist()
                status_map = {}
                ai_turnaround_map = {}
                latest_eps_map = {}
                year_eps_map = {}
                
                if codes:
                    conn = get_connection()
                    placeholders = ','.join(['?'] * len(codes))
                    df_eps = pd.read_sql(f'''
                        SELECT stock_code, year, quarter, eps
                        FROM quarterly_financials
                        WHERE stock_code IN ({placeholders})
                        ORDER BY year DESC, quarter DESC
                    ''', conn, params=tuple(codes))
                    conn.close()
                    
                    # 彙整需要讓 Gemini 評估的虧損股清單
                    turnaround_candidates = []
                    
                    for code in codes:
                        df_s_eps = df_eps[df_eps['stock_code'] == code]
                        row_yoy_val = df_ind[df_ind['stock_code'] == code]['yoy'].values[0] if not df_ind[df_ind['stock_code'] == code].empty else 0
                        row_name_val = df_ind[df_ind['stock_code'] == code]['stock_name'].values[0] if not df_ind[df_ind['stock_code'] == code].empty else ''
                        
                        if df_s_eps.empty:
                            status_map[code] = '⚪ 穩定獲利'
                            latest_eps_map[code] = None
                            year_eps_map[code] = None
                            continue
                            
                        latest_rec = df_s_eps.iloc[0]
                        eps_latest = latest_rec['eps']
                        y_latest = latest_rec['year']
                        q_latest = latest_rec['quarter']
                        
                        latest_eps_map[code] = eps_latest
                        year_eps_map[code] = df_s_eps['eps'].head(4).sum()
                        
                        # 前一季
                        if q_latest == 1:
                            y_prev, q_prev = y_latest - 1, 4
                        else:
                            y_prev, q_prev = y_latest, q_latest - 1
                            
                        df_prev = df_s_eps[(df_s_eps['year'] == y_prev) & (df_s_eps['quarter'] == q_prev)]
                        eps_prev = df_prev.iloc[0]['eps'] if not df_prev.empty else None
                        
                        # 去年同季
                        y_same, q_same = y_latest - 1, q_latest
                        df_same = df_s_eps[(df_s_eps['year'] == y_same) & (df_s_eps['quarter'] == q_same)]
                        eps_same = df_same.iloc[0]['eps'] if not df_same.empty else None
                        
                        # 邏輯判定
                        if eps_latest is None:
                            status_map[code] = '⚪ 穩定獲利'
                        elif eps_latest > 0:
                            if (eps_prev is not None and eps_prev <= 0) or (eps_same is not None and eps_same <= 0):
                                status_map[code] = '🟢 已轉盈'
                            else:
                                status_map[code] = '⚪ 穩定獲利'
                        else:
                            if eps_prev is not None and eps_latest > eps_prev and row_yoy_val is not None and row_yoy_val > 0:
                                status_map[code] = '🟡 減虧中'
                            else:
                                status_map[code] = '🔴 持續虧損'
                                
                            # 針對目前仍處於虧損的個股，加入 Gemini 分析的候選名單
                            turnaround_candidates.append({
                                'code': code,
                                'name': row_name_val,
                                'eps_latest': eps_latest,
                                'eps_prev': eps_prev,
                                'yoy': row_yoy_val
                            })
                            
                    # 調用 Gemini 聯網/基本面評估有機會轉虧為盈的 Highlight Flag
                    api_key = st.session_state.get('api_key_input')
                    if turnaround_candidates and api_key:
                        cache_turnaround_key = f"turnaround_pred_{selected_month}_{selected_industry}"
                        if cache_turnaround_key not in st.session_state:
                            import json
                            from gemini_service import predict_turnarounds_with_gemini
                            candidates_json = json.dumps(turnaround_candidates)
                            with st.spinner("🔮 Gemini AI 正在分析本產業個股的轉盈潛力..."):
                                preds = predict_turnarounds_with_gemini(api_key, selected_industry, candidates_json)
                                st.session_state[cache_turnaround_key] = preds
                                
                        for pred in st.session_state.get(cache_turnaround_key, []):
                            c = str(pred.get('code')).strip()
                            ai_turnaround_map[c] = True
                            
                    # 將 Gemini 預測的結果疊加到 status_map 中
                    for code in codes:
                        if code in ai_turnaround_map:
                            status_map[code] = status_map.get(code, '') + " | 🔮 AI預估轉盈"
                else:
                    status_map = {}
                    latest_eps_map = {}
                    year_eps_map = {}
                    
                df_ind['turnaround_status'] = df_ind['stock_code'].map(status_map)
                df_ind['latest_eps'] = df_ind['stock_code'].map(latest_eps_map)
                df_ind['year_eps'] = df_ind['stock_code'].map(year_eps_map)
                
                # 整理 DataFrame 顯示欄位，保留數值型態以利升降序排序
                df_display = df_ind[[
                    'stock_code', 'stock_name', 'turnaround_status', 'original_industry', 'refined_industry',
                    'revenue', 'mom', 'yoy', 'pe', 'dy', 'pb', 'latest_eps', 'year_eps'
                ]].copy()
                
                # 處理行選取觸發彈窗 (在 dataframe 渲染前偵測並清空，避免 StreamlitAPIException)
                suffix = st.session_state.get('df_key_suffix', 0)
                df_key = f"peer_dataframe_{suffix}"
                if df_key in st.session_state and st.session_state[df_key].get('selection', {}).get('rows'):
                    selected_row_idx = st.session_state[df_key]['selection']['rows'][0]
                    if selected_row_idx < len(df_ind):
                        selected_stock_code = df_ind.iloc[selected_row_idx]['stock_code']
                        selected_stock_name = df_ind.iloc[selected_row_idx]['stock_name']
                        st.session_state.active_dialog_stock = (selected_stock_code, selected_stock_name)
                    
                    # 在渲染前清空狀態
                    st.session_state[df_key] = {"selection": {"rows": [], "columns": []}}
                
                # 重新命名欄位名稱以提供友善中文表頭
                df_styled_base = df_display.rename(columns={
                    "stock_code": "股票代號",
                    "stock_name": "股票名稱",
                    "turnaround_status": "營運狀態",
                    "original_industry": "原始產業",
                    "refined_industry": "AI精細產業",
                    "revenue": "當月營收(千元)",
                    "mom": "月增率 MoM (%)",
                    "yoy": "年增率 YoY (%)",
                    "pe": "本益比 PE",
                    "dy": "殖利率 DY (%)",
                    "pb": "淨值比 PB",
                    "latest_eps": "最近一季 EPS (元)",
                    "year_eps": "最近一年 EPS (元)"
                })
                
                # 套用正紅負綠與數值格式化 (保留原生浮點數以保證排序功能)
                styled_peer_df = df_styled_base.style.map(
                    style_positive_red_negative_green,
                    subset=["月增率 MoM (%)", "年增率 YoY (%)", "最近一季 EPS (元)", "最近一年 EPS (元)"]
                ).format({
                    "當月營收(千元)": "{:,.0f}",
                    "月增率 MoM (%)": "{:+.1f}%",
                    "年增率 YoY (%)": "{:+.1f}%",
                    "本益比 PE": "{:.2f}",
                    "殖利率 DY (%)": "{:.2f}%",
                    "淨值比 PB": "{:.2f}",
                    "最近一季 EPS (元)": "{:+.2f}",
                    "最近一年 EPS (元)": "{:+.2f}"
                }, na_rep="-")
                
                # 啟用單選行，渲染 Styled DataFrame
                event = st.dataframe(
                    styled_peer_df,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=df_key
                )
                
                st.caption("💡 提示：您可以直接**點擊表格中的任一列**，或者使用下方選單，來開啟該個股的「聯網 AI 深度解析與日 K 線圖」！")
                
                # 雙重觸發器：下拉選單選股
                stock_names_list = [f"{row['stock_code']} {row['stock_name']}" for _, row in df_ind.iterrows()]
                st.selectbox(
                    "🔍 選擇個股開啟深度透視視窗",
                    ["請選擇個股..."] + stock_names_list,
                    key="detail_stock_select",
                    on_change=on_stock_picker_change
                )
                
                if st.session_state.active_dialog_stock:
                    show_stock_detail_dialog(*st.session_state.active_dialog_stock)
                
                # Gemini AI 深度個股分析
                st.write("---")
                st.markdown("### 🤖 Gemini 同業異軍突起個股深度解析")
                
                # 檢查快取
                cached_report, report_time = get_gemini_report_details('monthly_industry', f"{selected_month}_{selected_industry}")
                report_placeholder = st.empty()
                if cached_report:
                    st.success(f"已載入快取的 AI 分析報告 (更新時間: {report_time})")
                    report_placeholder.markdown(cached_report)
                    st.write("---")
                    if st.button("🔄 重新分析並更新 (即時更新)", key=f"re_analyze_industry_{selected_month}_{selected_industry}"):
                        api_key = st.session_state.get('api_key_input')
                        if not api_key:
                            st.error("請先在側邊欄配置您的 Gemini API Key！")
                        else:
                            with st.spinner("Gemini 正在分析同業財報數據並編寫報告，請稍候..."):
                                report = analyze_industry_outliers(
                                    api_key=api_key,
                                    db_path=None,
                                    date_month=selected_month,
                                    industry_name=selected_industry,
                                    use_refined=use_refined
                                )
                                if "失敗" in report or "未設定" in report or "error" in report.lower():
                                    st.error(handle_gemini_error(report))
                                else:
                                    report_placeholder.markdown(report)
                                    st.success("✔ 報告已成功更新！")
                                    st.rerun()
                else:
                    st.info("此產業與月份尚未產生 AI 分析報告。")
                    if st.button("🔮 召喚 Gemini 進行同業深度分析"):
                        api_key = st.session_state.get('api_key_input')
                        if not api_key:
                            st.error("請先在側邊欄配置您的 Gemini API Key！")
                        else:
                            with st.spinner("Gemini 正在分析同業財報數據並編寫報告，請稍候..."):
                                report = analyze_industry_outliers(
                                    api_key=api_key,
                                    db_path=None,  # 預設路徑
                                    date_month=selected_month,
                                    industry_name=selected_industry,
                                    use_refined=use_refined
                                )
                                if "失敗" in report or "未設定" in report or "error" in report.lower():
                                    st.error(handle_gemini_error(report))
                                else:
                                    st.markdown(report)
                                    st.rerun()

# --- 3. 頁面：季報三率分析 ---
elif choice == "📈 季報三率分析":
    st.markdown('<h1 class="gradient-text">📈 季度財報利潤率分析</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">分析上市櫃公司季度損益表之毛利率、淨利率、營業利益率與 EPS 指標，並與月營收趨勢進行聯動對比。</p>', unsafe_allow_html=True)
    
    # 選擇季別
    conn = get_connection()
    df_seasons = pd.read_sql('SELECT DISTINCT year, quarter FROM quarterly_financials ORDER BY year DESC, quarter DESC', conn)
    conn.close()
    
    if df_seasons.empty:
        st.info("資料庫中尚無季報資料。")
    else:
        season_list = [f"{row['year']} Q{row['quarter']}" for _, row in df_seasons.iterrows()]
        
        col_s_drop, col_s_note = st.columns([2, 3])
        with col_s_drop:
            selected_season = st.selectbox("選擇季度", season_list, index=0)
        with col_s_note:
            st.markdown("""
            <div style="font-size: 0.8rem; color: #64748B; padding: 10px; border-left: 3px solid #1E3A8A; margin-top: 5px;">
                💡 <b>台股季報公告法定截止日</b>：<br>
                第一季 (Q1): <b>5/15</b> | 第二季 (Q2): <b>8/14</b> | 第三季 (Q3): <b>11/14</b> | 第四季 (Q4年報): <b>次年 3/31</b>
            </div>
            """, unsafe_allow_html=True)
            
        if selected_season:
            year, quarter = map(int, selected_season.replace("Q", "").split())
            
            # 讀取該季財務報告
            raw_data = get_quarterly_financials_list(year, quarter)
            df_q = pd.DataFrame(raw_data)
            
            if df_q.empty:
                st.error("無此季度財報資料。")
            else:
                # 填補空值
                df_q['gross_margin'] = df_q['gross_margin'].fillna(0)
                df_q['net_margin'] = df_q['net_margin'].fillna(0)
                df_q['eps'] = df_q['eps'].fillna(0)
                
                # 計算大盤本季中位數
                med_gm = df_q['gross_margin'].median()
                med_nm = df_q['net_margin'].median()
                med_eps = df_q['eps'].median()
                
                st.markdown(f"### 📊 全市場季度獲利中位數：**{selected_season}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("市場毛利率中位數", f"{med_gm:.2f}%")
                c2.metric("市場淨利率中位數", f"{med_nm:.2f}%")
                c3.metric("市場每股盈餘 (EPS) 中位數", f"{med_eps:.2f}元")
                
                # 利潤率排行 (前 10 名)
                st.write("---")
                st.markdown("### 🏆 季度 EPS 與利潤率排行榜 (Top 10)")
                c_rank1, c_rank2 = st.columns(2)
                
                with c_rank1:
                    st.markdown('<div class="stCard">', unsafe_allow_html=True)
                    st.markdown("#### 🥇 EPS 排行榜")
                    top_eps = df_q.sort_values(by='eps', ascending=False).head(10)[[
                        'stock_code', 'stock_name', 'eps', 'gross_margin', 'net_margin'
                    ]]
                    st.dataframe(
                        top_eps,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "stock_code": st.column_config.TextColumn("代號"),
                            "stock_name": st.column_config.TextColumn("公司名稱"),
                            "eps": st.column_config.NumberColumn("EPS (元)", format="%.2f"),
                            "gross_margin": st.column_config.NumberColumn("毛利率 (%)", format="%.2f%%"),
                            "net_margin": st.column_config.NumberColumn("淨利率 (%)", format="%.2f%%")
                        }
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with c_rank2:
                    st.markdown('<div class="stCard">', unsafe_allow_html=True)
                    st.markdown("#### 🥇 毛利率排行榜 (營業收入大於 5 億元)")
                    # 過濾低營收干擾項
                    top_gm = df_q[df_q['revenue'] >= 500000].sort_values(by='gross_margin', ascending=False).head(10)[[
                        'stock_code', 'stock_name', 'gross_margin', 'net_margin', 'eps'
                    ]]
                    st.dataframe(
                        top_gm,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "stock_code": st.column_config.TextColumn("代號"),
                            "stock_name": st.column_config.TextColumn("公司名稱"),
                            "gross_margin": st.column_config.NumberColumn("毛利率 (%)", format="%.2f%%"),
                            "net_margin": st.column_config.NumberColumn("淨利率 (%)", format="%.2f%%"),
                            "eps": st.column_config.NumberColumn("EPS (元)", format="%.2f")
                        }
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 個股營收與利潤率聯動趨勢 (領先與落後指標對比)
                st.write("---")
                st.markdown("### 🔍 個股「月營收年增率」與「季度三率」對比分析")
                
                # 選擇特定個股
                stock_list = [f"{row['stock_code']} {row['stock_name']}" for _, row in df_q.iterrows()]
                selected_stock_str = st.selectbox("選擇要分析的個股", stock_list, index=0)
                
                if selected_stock_str:
                    stock_code = selected_stock_str.split()[0]
                    
                    conn = get_connection()
                    # 撈取個股所有的歷史季度財報
                    df_stock_q = pd.read_sql(
                        'SELECT year, quarter, gross_margin, net_margin, eps, revenue FROM quarterly_financials WHERE stock_code = ? ORDER BY year, quarter',
                        conn, params=(stock_code,)
                    )
                    # 撈取個股所有的歷史月營收
                    df_stock_m = pd.read_sql(
                        'SELECT date_month, revenue, yoy, mom FROM monthly_revenue WHERE stock_code = ? ORDER BY date_month',
                        conn, params=(stock_code,)
                    )
                    conn.close()
                    
                    if df_stock_q.empty or df_stock_m.empty:
                        st.warning("此個股歷史數據不足，無法繪製聯動趨勢。")
                    else:
                        st.markdown(f"#### 📈 {selected_stock_str} 財務成長聯動曲線")
                        
                        # 繪製圖表：季度毛利率 vs 歷史月營收 YoY
                        fig = go.Figure()
                        
                        # 月度營收年增率
                        fig.add_trace(go.Scatter(
                            x=df_stock_m['date_month'],
                            y=df_stock_m['yoy'],
                            name='月營收年增率 YoY (%)',
                            line=dict(color='#1E3A8A', width=2.5),
                            yaxis='y1'
                        ))
                        
                        # 季度毛利率
                        # 為了將季度對齊到時間軸，我們將 "YYYY QX" 轉換為大致的月份，如 Q1 對齊到 3 月, Q2 到 6 月
                        q_dates = []
                        for _, r in df_stock_q.iterrows():
                            m = int(r['quarter'] * 3)
                            q_dates.append(f"{int(r['year'])}-{m:02d}")
                            
                        fig.add_trace(go.Scatter(
                            x=q_dates,
                            y=df_stock_q['gross_margin'],
                            name='季報毛利率 (%)',
                            line=dict(color='#DC2626', width=3, dash='dash'),
                            marker=dict(size=8),
                            yaxis='y2'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=q_dates,
                            y=df_stock_q['net_margin'],
                            name='季報淨利率 (%)',
                            line=dict(color='#475569', width=3, dash='dot'),
                            marker=dict(size=8),
                            yaxis='y2'
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_stock_str} 營收成長與獲利比率對照圖",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#1E293B',
                            xaxis=dict(
                                title='時間軸',
                                type='date',
                                tickformat="%Y-%m",
                                dtick="M1",
                                tickmode="linear",
                                showgrid=True,
                                gridcolor='#F1F5F9'
                            ),
                            yaxis=dict(title='月營收 YoY (%)', color='#1E3A8A', showgrid=True, gridcolor='#F1F5F9'),
                            yaxis2=dict(
                                title='毛利/淨利率 (%)',
                                color='#DC2626',
                                overlaying='y',
                                side='right'
                            ),
                            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("提示：當「營收年增率 (藍線)」持續走高，且「毛利率 (綠虛線)」同步向上時，通常代表公司正處於強烈的產品循環與營運槓桿擴張期。")
                        
                # 季度財報 AI 整體分析
                st.write("---")
                st.markdown("### 🤖 Gemini 季度財報總體獲利大解析")
                
                # 檢查快取
                cached_report, report_time = get_gemini_report_details('quarterly_market', f"{year}_Q{quarter}")
                report_placeholder = st.empty()
                if cached_report:
                    st.success(f"已載入快取的 AI 季報分析報告 (更新時間: {report_time})")
                    report_placeholder.markdown(cached_report)
                    st.write("---")
                    if st.button("🔄 重新分析並更新 (即時更新)", key=f"re_analyze_quarterly_{year}_Q{quarter}"):
                        api_key = st.session_state.get('api_key_input')
                        if not api_key:
                            st.error("請先在側邊欄配置您的 Gemini API Key！")
                        else:
                            with st.spinner("Gemini 正在彙總全市場財報並撰寫獲利分析報告，請稍候..."):
                                report = analyze_quarterly_financial_trends(
                                    api_key=api_key,
                                    db_path=None,
                                    year=year,
                                    quarter=quarter
                                )
                                if "失敗" in report or "未設定" in report or "error" in report.lower():
                                    st.error(handle_gemini_error(report))
                                else:
                                    report_placeholder.markdown(report)
                                    st.success("✔ 報告已成功更新！")
                                    st.rerun()
                else:
                    st.info("此季度財報尚未產生 AI 分析報告。")
                    if st.button("🔮 召喚 Gemini 進行季度財報深度大解析"):
                        api_key = st.session_state.get('api_key_input')
                        if not api_key:
                            st.error("請先在側邊欄配置您的 Gemini API Key！")
                        else:
                            with st.spinner("Gemini 正在彙總全市場財報並撰寫獲利分析報告，請稍候..."):
                                report = analyze_quarterly_financial_trends(
                                    api_key=api_key,
                                    db_path=None,
                                    year=year,
                                    quarter=quarter
                                )
                                if "失敗" in report or "未設定" in report or "error" in report.lower():
                                    st.error(handle_gemini_error(report))
                                else:
                                    st.markdown(report)
                                    st.rerun()

# --- 3.5. 頁面：潛力轉盈股分析 ---
elif choice == "🔮 潛力轉盈股分析":
    st.markdown('<h1 class="gradient-text">🔮 Gemini 智慧轉虧為盈潛力股大解析</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">由 Gemini AI 自動掃描近幾季處於虧損（或減虧中）但最新月營收 YoY 成長的潛力標的，並進行聯網深度原因分析。</p>', unsafe_allow_html=True)
    
    # 取得當前月份
    current_month = datetime.now().strftime("%Y-%m")
    
    # 檢查快取
    cached_turnaround_list, report_time = get_gemini_report_details('turnaround_list', current_month)
    report_placeholder = st.empty()
    if cached_turnaround_list:
        st.success(f"已載入當月的 AI 潛力轉虧為盈分析報告。 (更新時間: {report_time})")
        report_placeholder.markdown(cached_turnaround_list)
        st.write("---")
        render_verification_table(cached_turnaround_list, title="📊 轉盈候選股均線與 20MA 趨勢量化驗證表")
        st.write("---")
        if st.button("🔄 重新分析並更新 (即時更新)", key='re_run_turnaround_analysis'):
            api_key = st.session_state.get('api_key_input')
            if not api_key:
                st.error("請先在側邊欄配置您的 Gemini API Key！")
            else:
                with st.spinner("Gemini 正在從資料庫篩選虧損股、聯網搜尋最新動態並撰寫深度報告，請稍候... (這可能需要 20-30 秒)"):
                    report = analyze_turnaround_stocks(api_key, db_path=None)
                    if "失敗" in report or "未設定" in report or "error" in report.lower():
                        st.error(handle_gemini_error(report))
                    else:
                        report_placeholder.markdown(report)
                        st.success("✔ 報告已成功更新！")
                        st.rerun()
    else:
        st.info("尚未產生當月的潛力轉虧為盈分析報告。")
        if st.button("🔮 執行轉虧為盈潛力股大掃描與 AI 分析", key='run_turnaround_analysis'):
            api_key = st.session_state.get('api_key_input')
            if not api_key:
                st.error("請先在側邊欄配置您的 Gemini API Key！")
            else:
                with st.spinner("Gemini 正在從資料庫篩選虧損股、聯網搜尋最新動態並撰寫深度報告，請稍候... (這可能需要 20-30 秒)"):
                    report = analyze_turnaround_stocks(api_key, db_path=None)
                    if "失敗" in report or "未設定" in report or "error" in report.lower():
                        st.error(handle_gemini_error(report))
                    else:
                        st.markdown(report)
                        st.rerun()

# --- 3.6. 頁面：籌碼與技術糾結股 ---
elif choice == "🎯 籌碼與技術糾結股":
    st.markdown('<h1 class="gradient-text">🎯 主力籌碼集中與均線糾結股掃描</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">由 Gemini AI 聯網分析近期「特定分點持續買超/收購、籌碼集中度上升」的個股，並結合實時均線（5MA/10MA/20MA/60MA）糾結度進行量化計算與驗證。</p>', unsafe_allow_html=True)
    
    current_month = datetime.now().strftime("%Y-%m")
    
    # 檢查快取
    cached_chip_report, report_time = get_gemini_report_details('chip_and_ma_convergence', current_month)
    report_placeholder = st.empty()
    
    api_key = st.session_state.get('api_key_input')
    


    if cached_chip_report:
        st.success(f"已載入當月的籌碼與技術分析報告。 (更新時間: {report_time})")
        report_placeholder.markdown(cached_chip_report)
        st.write("---")
        render_verification_table(cached_chip_report)
        
        st.write("---")
        if st.button("🔄 重新分析並更新 (即時更新)", key='re_run_chip_analysis'):
            if not api_key:
                st.error("請先在側邊欄配置您的 Gemini API Key！")
            else:
                with st.spinner("Gemini 正在聯網搜尋最新分點籌碼動態，並計算均線狀態，請稍候... (這可能需要 20-30 秒)"):
                    from gemini_service import analyze_chip_and_ma_convergence
                    report = analyze_chip_and_ma_convergence(api_key, db_path=None)
                    if "失敗" in report or "未設定" in report or "error" in report.lower():
                        st.error(handle_gemini_error(report))
                    else:
                        report_placeholder.markdown(report)
                        st.success("✔ 報告已成功更新！")
                        st.rerun()
    else:
        st.info("尚未產生當月的籌碼與技術糾結分析報告。")
        if st.button("🎯 執行籌碼與技術面大掃描", key='run_chip_analysis'):
            if not api_key:
                st.error("請先在側邊欄配置您的 Gemini API Key！")
            else:
                with st.spinner("Gemini 正在聯網搜尋最新分點籌碼動態，並計算均線狀態，請稍候... (這可能需要 20-30 秒)"):
                    from gemini_service import analyze_chip_and_ma_convergence
                    report = analyze_chip_and_ma_convergence(api_key, db_path=None)
                    if "失敗" in report or "未設定" in report or "error" in report.lower():
                        st.error(handle_gemini_error(report))
                    else:
                        report_placeholder.markdown(report)
                        st.success("✔ 報告已成功掃描！")
                        st.rerun()


# --- 3.7. 頁面：當月異軍突起股 ---
elif choice == "🚀 當月異軍突起股":
    st.markdown('<h1 class="gradient-text">🚀 當月營收異軍突起股</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">篩選當月營收 YoY 巨大爆發、高成長且具備一定規模的標的，並透過 Gemini 聯網解析其營收翻倍的具體核心驅動原因。</p>', unsafe_allow_html=True)
    
    current_month = get_latest_month()
    st.markdown(f"##### 📅 目前分析期數: **{current_month}** 月營收數據")
    
    # 設置篩選過濾器
    st.markdown("### 🛠️ 篩選與過濾條件設定")
    col1, col2, col3 = st.columns(3)
    with col1:
        exclude_special = st.checkbox("排除建材營造與金融業 (營收波動多為一次性入帳)", value=True)
    with col2:
        min_yoy = st.slider("最低營收年增率 (YoY %)", min_value=10, max_value=200, value=25)
    with col3:
        min_rev = st.number_input("最低營收規模 (百萬元 TWD)", min_value=1, max_value=50000, value=20)
        
    conn = get_connection()
    sql = '''
        SELECT stock_code, stock_name, industry, revenue, last_year_revenue, last_month_revenue, yoy, mom 
        FROM monthly_revenue 
        WHERE date_month = (SELECT MAX(date_month) FROM monthly_revenue)
          AND yoy >= ? AND revenue >= ?
    '''
    params = [min_yoy, min_rev * 1000]
    if exclude_special:
        sql += " AND industry NOT IN ('建材營造', '金融業', '金融保險業', '證券業')"
    sql += " ORDER BY yoy DESC"
    
    df_surging = pd.read_sql(sql, conn, params=params)
    conn.close()
    
    if df_surging.empty:
        st.info("目前條件下無符合的異軍突起股，請放寬篩選標準。")
    else:
        st.markdown(f"### 📊 當月高成長與技術/籌碼篩選表 (共 {len(df_surging)} 檔)")
        st.caption("以下表格已串接即時技術指標。您可以點擊欄位標頭進行排序（例如點擊「接近52週高點 (%)」或「5-20MA糾結度 (%)」）以快速篩選最完美的起漲點型態。")
        
        # 批次查詢技術指標
        with st.spinner("正在計算候選股之即時均線糾結度與 52 週高點位置..."):
            from gemini_service import check_surging_technical_batch
            tech_results = check_surging_technical_batch(df_surging['stock_code'].tolist())
            
        df_display = df_surging.copy()
        df_display['revenue'] = df_display['revenue'] / 1000.0  # 轉為百萬
        
        # 初始化技術面欄位
        prices = []
        spreads_short = []
        trends_20ma = []
        proximities_52w = []
        
        for _, row in df_display.iterrows():
            code = row['stock_code']
            res_val = tech_results.get(code)
            if res_val and res_val[0]:  # success
                _, p, sp_s, sp_a, tr_20, h52, prox52 = res_val
                prices.append(p)
                spreads_short.append(sp_s)
                trends_20ma.append("📈 向上" if tr_20 else "📉 向下/持平")
                proximities_52w.append(prox52)
            else:
                prices.append(None)
                spreads_short.append(None)
                trends_20ma.append("未知")
                proximities_52w.append(None)
                
        df_display['即時股價'] = prices
        df_display['5-20MA糾結度 (%)'] = spreads_short
        df_display['20MA趨勢'] = trends_20ma
        df_display['接近52週高點 (%)'] = proximities_52w
        
        df_display = df_display.rename(columns={
            'stock_code': '股票代號',
            'stock_name': '股票名稱',
            'industry': '產業別',
            'revenue': '當月營收 (百萬元)',
            'yoy': '營收年增率 (YoY %)',
            'mom': '營收月增率 (MoM %)'
        })
        
        # 顯示包含技術篩選屬性的 dataframe
        st.dataframe(
            df_display[[
                '股票代號', '股票名稱', '產業別', '當月營收 (百萬元)', 
                '營收年增率 (YoY %)', '營收月增率 (MoM %)', '即時股價', 
                '5-20MA糾結度 (%)', '20MA趨勢', '接近52週高點 (%)'
            ]],
            use_container_width=True
        )
        
        # 個股技術與籌碼交互深度解析
        st.write("---")
        st.markdown("### 🔍 個股 K 線圖（包含 5/10/20/60MA 與 52W 高點）及 AI 即時籌碼分析")
        st.caption("選取下方高成長個股，即時查看其「均線糾結起漲 K 線」與「主力分點大戶吃貨」AI 深度聯網分析報告。")
        
        stock_options = [f"{row['股票代號']} {row['股票名稱']}" for _, row in df_display.iterrows()]
        selected_stock_str = st.selectbox("選取要進行深度剖析的個股", stock_options, key='surging_stock_select')
        
        if selected_stock_str:
            stock_code = selected_stock_str.split()[0]
            stock_name = selected_stock_str.split()[1]
            
            tab_chart, tab_chip = st.tabs(["📈 即時均線與 52W 高點 K 線圖", "🔮 該股籌碼與分點 AI 即時聯網大解析"])
            
            with tab_chart:
                with st.spinner(f"正在加載 {selected_stock_str} 的技術圖形與均線..."):
                    fig_52w = draw_k_line_chart_with_52w(stock_code)
                    if fig_52w:
                        st.plotly_chart(fig_52w, use_container_width=True)
                        st.caption("註：紅虛線為過去 52 週（約 1 年）盤中最高價位。橘線 5MA、淺藍線 10MA、粉紅線 20MA、綠線 60MA。均線越靠攏且接近 52W 高點突破，越符合起漲共振訊號。")
            
            with tab_chip:
                api_key = st.session_state.get('api_key_input')
                if not api_key:
                    st.warning("⚠️ 請先在側邊欄配置您的 Gemini API Key 以啟用個股聯網 AI 即時籌碼分析！")
                else:
                    # 檢查個股籌碼報告快取
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT report_content, updated_at FROM gemini_reports WHERE report_type = 'stock_chip_detail' AND report_key = ?",
                        (stock_code,)
                    )
                    row = cursor.fetchone()
                    conn.close()
                    
                    if row:
                        st.success(f"已載入 {selected_stock_str} 的籌碼分析報告。 (更新時間: {row['updated_at']})")
                        st.markdown(row['report_content'])
                        st.write("---")
                        if st.button(f"🔄 重新分析 {stock_name} 主力分點與籌碼", key=f"re_run_chip_detail_{stock_code}"):
                            with st.spinner(f"正在透過 AI 聯網查詢 {selected_stock_str} 最近分點囤貨與大戶鎖碼特徵..."):
                                from gemini_service import get_single_stock_chip_analysis
                                report = get_single_stock_chip_analysis(api_key, stock_code, stock_name)
                                if "失敗" not in report and "未設定" not in report:
                                    conn = get_connection()
                                    cursor = conn.cursor()
                                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    cursor.execute(
                                        "INSERT OR REPLACE INTO gemini_reports (report_type, report_key, report_content, updated_at) VALUES ('stock_chip_detail', ?, ?, ?)",
                                        (stock_code, report, current_time)
                                    )
                                    conn.commit()
                                    conn.close()
                                    st.rerun()
                                else:
                                    st.error(report)
                    else:
                        st.info("💡 該股目前尚無籌碼分析報告。點擊下方按鈕，AI 將即時進行聯網大數據搜尋，深入剖析特定主力券商分點囤貨動向。")
                        if st.button(f"🚀 啟動 {stock_name} AI 聯網籌碼與分點深度解析", key=f"run_chip_detail_{stock_code}"):
                            with st.spinner(f"正在透過 AI 聯網查詢 {selected_stock_str} 最近分點囤貨與大戶鎖碼特徵..."):
                                from gemini_service import get_single_stock_chip_analysis
                                report = get_single_stock_chip_analysis(api_key, stock_code, stock_name)
                                if "失敗" not in report and "未設定" not in report:
                                    conn = get_connection()
                                    cursor = conn.cursor()
                                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    cursor.execute(
                                        "INSERT OR REPLACE INTO gemini_reports (report_type, report_key, report_content, updated_at) VALUES ('stock_chip_detail', ?, ?, ?)",
                                        (stock_code, report, current_time)
                                    )
                                    conn.commit()
                                    conn.close()
                                    st.rerun()
                                else:
                                    st.error(report)
        
        st.write("---")
        st.markdown("### 🔮 Gemini AI 聯網異軍突起原因解析")
        
        cached_analysis, report_time = get_gemini_report_details('surging_stocks_analysis', current_month[:7])
        report_placeholder = st.empty()
        api_key = st.session_state.get('api_key_input')
        
        if cached_analysis:
            st.success(f"已載入當月的 AI 異軍突起分析報告。 (更新時間: {report_time})")
            report_placeholder.markdown(cached_analysis)
            st.write("---")
            render_verification_table(cached_analysis, title="📊 異軍突起股實時均線與 20MA 趨勢量化驗證表")
            st.write("---")
            if st.button("🔄 重新分析並更新 (即時更新)", key='re_run_surging_analysis'):
                if not api_key:
                    st.error("請先在側邊欄配置您的 Gemini API Key！")
                else:
                    with st.spinner("Gemini 正在聯網查詢這些個股營收暴增原因，並進行深度解析，請稍候... (這可能需要 20-30 秒)"):
                        from gemini_service import analyze_surging_stocks
                        report = analyze_surging_stocks(api_key, db_path=None)
                        if "失敗" in report or "未設定" in report or "error" in report.lower():
                            st.error(handle_gemini_error(report))
                        else:
                            report_placeholder.markdown(report)
                            st.success("✔ 報告已成功更新！")
                            st.rerun()
        else:
            st.info("尚未產生當月的異軍突起分析報告。")
            if st.button("🚀 執行高成長異軍突起股 AI 聯網大解析", key='run_surging_analysis'):
                if not api_key:
                    st.error("請先在側邊欄配置您的 Gemini API Key！")
                else:
                    with st.spinner("Gemini 正在聯網查詢這些個股營收暴增原因，並進行深度解析，請稍候... (這可能需要 20-30 秒)"):
                        from gemini_service import analyze_surging_stocks
                        report = analyze_surging_stocks(api_key, db_path=None)
                        if "失敗" in report or "未設定" in report or "error" in report.lower():
                            st.error(handle_gemini_error(report))
                        else:
                            st.markdown(report)
                            st.rerun()

# --- 3.8. 頁面：評等調整追蹤 ---
elif choice == "📈 評等調整追蹤":
    st.markdown('<h1 class="gradient-text">📈 外資與投信評等調整追蹤</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">追蹤各大外資與投信研究機構對台股個股評等的升降、目標價的調整，以及對應本益比變化的原因解析。</p>', unsafe_allow_html=True)
    
    tab_board, tab_manual, tab_ai = st.tabs(["📊 評等調整看板", "✍️ 手動新增評等", "🤖 AI 智慧聯網掃描"])
    
    with tab_board:
        st.markdown("### 🔍 評等變動看板與歷史紀錄")
        
        # 篩選條件
        col1, col2, col3 = st.columns(3)
        with col1:
            search_code = st.text_input("搜尋股票代號或名稱", "")
        with col2:
            search_broker = st.text_input("搜尋券商/研究機構", "")
        with col3:
            date_filter = st.date_input("篩選調整日期", value=None)
            
        # 查詢資料庫
        conn = get_connection()
        # 防禦性檢查：確保資料表一定存在，避免部署延遲或資料庫檔案不同步導致的 pandas read_sql 報錯
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rating_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,                    -- 調整日期 (YYYY-MM-DD)
                stock_code TEXT,              -- 股票代號
                stock_name TEXT,              -- 股票名稱
                broker TEXT,                  -- 報告券商/研究機構
                original_rating TEXT,         -- 原評等
                new_rating TEXT,              -- 新評等
                target_price REAL,            -- 目標價
                reason TEXT,                  -- 調整原因與分析
                current_pe REAL,              -- 現行 PE
                adjusted_pe REAL,             -- 調整後 PE (目標價對應 PE)
                created_at TEXT
            )
        ''')
        conn.commit()
        
        sql = '''
            SELECT id, date, stock_code, stock_name, broker, original_rating, 
                   new_rating, target_price, current_pe, adjusted_pe, reason
            FROM rating_adjustments
            WHERE 1=1
        '''
        params = []
        if search_code:
            sql += " AND (stock_code LIKE ? OR stock_name LIKE ?)"
            params.append(f"%{search_code}%")
            params.append(f"%{search_code}%")
        if search_broker:
            sql += " AND broker LIKE ?"
            params.append(f"%{search_broker}%")
        if date_filter:
            sql += " AND date = ?"
            params.append(str(date_filter))
            
        sql += " ORDER BY date DESC, id DESC"
        df_ratings = pd.read_sql(sql, conn, params=params)
        conn.close()
        
        if df_ratings.empty:
            st.info("目前尚無符合條件的評等調整紀錄。您可以在其他分頁手動新增或啟用 AI 掃描！")
        else:
            # 複製一份用於美觀顯示
            df_disp = df_ratings.copy()
            df_disp = df_disp.rename(columns={
                'date': '日期',
                'stock_code': '股票代號',
                'stock_name': '股票名稱',
                'broker': '研究機構',
                'original_rating': '原先評等',
                'new_rating': '調整後評等',
                'target_price': '目標價 (TWD)',
                'current_pe': '現行 PE',
                'adjusted_pe': '目標價 PE'
            })
            
            # 使用 st.dataframe 呈現表格
            st.dataframe(
                df_disp[['日期', '股票代號', '股票名稱', '研究機構', '原先評等', '調整後評等', '目標價 (TWD)', '現行 PE', '目標價 PE']],
                use_container_width=True
            )
            
            # 展開詳細原因
            st.write("---")
            st.markdown("### 💬 調整理由與深度解析")
            rating_options = [f"{row['date']} | {row['stock_code']} {row['stock_name']} ({row['broker']})" for _, row in df_ratings.iterrows()]
            selected_idx = st.selectbox("選擇一筆記錄以查看詳細調整理由：", range(len(rating_options)), format_func=lambda x: rating_options[x])
            
            if selected_idx is not None:
                record = df_ratings.iloc[selected_idx]
                st.markdown(f"#### 🎯 **{record['stock_code']} {record['stock_name']}** — **{record['broker']}** 調整評等點評")
                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.write(f"📅 **日期**：{record['date']}")
                    st.write(f"🏷️ **評等變動**：`{record['original_rating']}` ➔ `{record['new_rating']}`")
                    st.write(f"🎯 **目標價**：`{record['target_price']} 元`")
                with col_det2:
                    st.write(f"📊 **現行本益比 (PE)**：`{record['current_pe']} 倍`")
                    st.write(f"📈 **調整後(目標價)本益比**：`{record['adjusted_pe']} 倍`")
                    
                st.markdown("##### 💡 **調評核心原因與展望**：")
                st.info(record['reason'] if record['reason'] else "無提供具體理由。")
                
                # 顯示該股 K 線與均線圖
                st.markdown("##### 📈 **即時均線與 52W 高點 K 線對齊**：")
                fig_rating = draw_k_line_chart_with_52w(record['stock_code'])
                if fig_rating:
                    st.plotly_chart(fig_rating, use_container_width=True)
                    st.caption("註：紅色虛線為 52 週最高價。橘線 5MA、淺藍線 10MA、粉紅線 20MA、綠線 60MA。")
                    
    with tab_manual:
        st.markdown("### ✍️ 手動新增或每日更新個股評等調整")
        st.write("手動輸入今天或近期調整評等的台股個股資訊：")
        
        with st.form("manual_rating_form", clear_on_submit=True):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                f_date = st.date_input("評等調整日期", datetime.now())
                f_code = st.text_input("股票代號 (如 2330)", "")
                f_name = st.text_input("股票名稱 (如 台積電)", "")
                f_broker = st.text_input("研究機構/券商 (如 摩根大通)", "")
            with col_f2:
                f_orig = st.text_input("原先評等 (如 中立)", "")
                f_new = st.text_input("調整後評等 (如 買進)", "")
                f_target = st.number_input("目標價 (TWD)", min_value=0.0, value=0.0, step=0.5)
                f_curr_pe = st.number_input("現行本益比 (PE)", min_value=0.0, value=0.0, step=0.1)
                f_adj_pe = st.number_input("調整後本益比 (目標價對應 PE)", min_value=0.0, value=0.0, step=0.1)
                
            f_reason = st.text_area("評等調整原因與核心邏輯", height=120)
            
            submit_btn = st.form_submit_button("➕ 提交並儲存至資料庫")
            
            if submit_btn:
                if not f_code or not f_name or not f_broker:
                    st.error("❌ 股票代號、公司名稱與研究機構為必填欄位！")
                else:
                    conn = get_connection()
                    cursor = conn.cursor()
                    # 確保資料表存在，避免部署不同步問題
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS rating_adjustments (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            date TEXT,
                            stock_code TEXT,
                            stock_name TEXT,
                            broker TEXT,
                            original_rating TEXT,
                            new_rating TEXT,
                            target_price REAL,
                            reason TEXT,
                            current_pe REAL,
                            adjusted_pe REAL,
                            created_at TEXT
                        )
                    ''')
                    conn.commit()
                    created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute('''
                        INSERT INTO rating_adjustments (
                            date, stock_code, stock_name, broker, original_rating, 
                            new_rating, target_price, reason, current_pe, adjusted_pe, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        str(f_date), f_code, f_name, f_broker, f_orig or '未知',
                        f_new or '未知', f_target, f_reason, f_curr_pe, f_adj_pe, created_time
                    ))
                    conn.commit()
                    conn.close()
                    st.success(f"✔ 成功手動新增 {f_code} {f_name} 的評等變動紀錄！")
                    st.rerun()
                    
    with tab_ai:
        st.markdown("### 🤖 AI 智慧聯網全自動掃描今日評等")
        st.write("點擊下方按鈕，AI 將自動聯網（透過 Google Search Grounding）檢索近期台灣股市上有關各大外資與投信最新出爐的評等調整與目標價報告，並自動解析與寫入您的資料庫。")
        
        # 顯示最近一次的掃描狀態與更新時間
        if 'rating_scan_success' in st.session_state:
            st.success(st.session_state.rating_scan_success)
        if 'rating_scan_error' in st.session_state:
            st.error(st.session_state.rating_scan_error)
            
        api_key = st.session_state.get('api_key_input')
        if not api_key:
            st.warning("⚠️ 請先在側邊欄配置您的 Gemini API Key 以啟用 AI 聯網掃描！")
        else:
            if st.button("🚀 啟動 AI 聯網自動更新今日評等調整", key='run_ai_rating_scan'):
                with st.spinner("AI 正在聯網搜集今日最新外資/投信評等調整報告，並自動提取 PE 與原因，請稍候... (這大約需要 20-30 秒)"):
                    from gemini_service import scan_broker_ratings
                    res_msg = scan_broker_ratings(api_key, db_path=None)
                    
                    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if "失敗" in res_msg or "error" in res_msg.lower():
                        st.session_state.rating_scan_error = f"❌ {res_msg} | 執行時間: {current_time_str}"
                        st.session_state.pop('rating_scan_success', None)
                    else:
                        st.session_state.rating_scan_success = f"✔ {res_msg} | 執行時間: {current_time_str}"
                        st.session_state.pop('rating_scan_error', None)
                    st.rerun()

# --- 4. 頁面：Gemini AI 投資顧問 ---
elif choice == "🤖 Gemini AI 投資顧問":
    st.markdown('<h1 class="gradient-text">🤖 Gemini AI 智慧基本面投顧</h1>', unsafe_allow_html=True)
    # 選擇月份
    conn = get_connection()
    df_months = pd.read_sql('SELECT DISTINCT date_month FROM monthly_revenue ORDER BY date_month DESC', conn)
    conn.close()
    
    tab1, tab2, tab3 = st.tabs(["📝 月度大盤趨勢報告", "💬 個股與基本面智慧問答", "📢 當年度法說會智慧分析"])
    
    with tab1:
        selected_month = st.selectbox("選擇報告月份", df_months['date_month'].tolist(), index=0, key='advisor_month')
        
        if selected_month:
            # 檢查快取
            cached_report, report_time = get_gemini_report_details('monthly_market', selected_month)
            report_placeholder = st.empty()
            if cached_report:
                st.success(f"已載入快取的 AI 月度營收策略報告 (更新時間: {report_time})")
                report_placeholder.markdown(cached_report)
                st.write("---")
                if st.button("🔄 重新分析並更新 (即時更新)", key=f"re_analyze_monthly_market_{selected_month}"):
                    api_key = st.session_state.get('api_key_input')
                    if not api_key:
                        st.error("請先在側邊欄配置您的 Gemini API Key！")
                    else:
                        with st.spinner("Gemini 正在運算數據並撰寫策略報告，這可能需要 15-20 秒..."):
                            report = analyze_monthly_market_trends(
                                api_key=api_key,
                                db_path=None,
                                date_month=selected_month
                            )
                            if "失敗" in report or "未設定" in report or "error" in report.lower():
                                st.error(handle_gemini_error(report))
                            else:
                                report_placeholder.markdown(report)
                                st.success("✔ 報告已成功更新！")
            else:
                st.info("此月份營收尚未產生 AI 策略報告。")
                if st.button("🔮 產生大盤營收趨勢策略報告"):
                    api_key = st.session_state.get('api_key_input')
                    if not api_key:
                        st.error("請先在側邊欄配置您的 Gemini API Key！")
                    else:
                        with st.spinner("Gemini 正在運算數據並撰寫策略報告，這可能需要 15-20 秒..."):
                            report = analyze_monthly_market_trends(
                                api_key=api_key,
                                db_path=None,
                                date_month=selected_month
                            )
                            if "失敗" in report or "未設定" in report or "error" in report.lower():
                                st.error(handle_gemini_error(report))
                            else:
                                st.markdown(report)
                                st.rerun()
                            
    with tab2:
        st.markdown("#### 💡 基本面智慧問答")
        st.write("您可以針對資料庫中的個股營收、本益比、毛利率或產業趨勢進行詢問，AI 投資顧問將根據當前資料庫數據為您提供客觀的解析。")
        
        # 使用者提問輸入框
        user_query = st.text_input("輸入您的基本面問題：", placeholder="例如：2330 台積電近期的營收趨勢如何？與同業相比表現好嗎？")
        
        if st.button("💬 詢問 AI 顧問"):
            api_key = st.session_state.get('api_key_input')
            if not api_key:
                st.error("請先在側邊欄配置您的 Gemini API Key！")
            else:
                model = get_gemini_model(api_key)
                if not model:
                    st.error("初始化 Gemini 失敗，請確認 API Key。")
                else:
                    # 撈取個股代號
                    # 嘗試從使用者的輸入中匹配 4 位數的股票代碼
                    code_match = re.search(r'\b\d{4}\b', user_query)
                    
                    stock_context = ""
                    if code_match:
                        code = code_match.group(0)
                        # 撈取該股的財務數據
                        conn = get_connection()
                        df_rev = pd.read_sql('SELECT date_month, revenue, yoy, mom FROM monthly_revenue WHERE stock_code = ? ORDER BY date_month DESC LIMIT 6', conn, params=(code,))
                        df_fin = pd.read_sql('SELECT year, quarter, gross_margin, net_margin, eps FROM quarterly_financials WHERE stock_code = ? ORDER BY year DESC, quarter DESC LIMIT 4', conn, params=(code,))
                        df_pe = pd.read_sql('SELECT pe, pb, dy FROM daily_pe WHERE stock_code = ? ORDER BY date DESC LIMIT 1', conn, params=(code,))
                        conn.close()
                        
                        if not df_rev.empty:
                            stock_context = f"""
【資料庫中有關股票 {code} 的最新數據資訊】:
1. 月營收趨勢 (最近6個月):
{df_rev.to_string(index=False)}

2. 季報毛利率與獲利 (最近4季):
{df_fin.to_string(index=False)}

3. 最新估值 (本益比、殖利率、股價淨值比):
{df_pe.to_string(index=False)}
"""
                    
                    # 建立 Prompt
                    full_prompt = f"""
你是一位專業且客觀的台股基本面分析師。
使用者想問你一個問題：
"{user_query}"

{stock_context}

請根據上述提供的資料庫最新財務資訊（如果有的話）以及你對台灣股市基本面的了解，為使用者解答。
回答時請注意：
- 若有提供最新財務數據，請務必精準引用並進行分析（如計算營收複合成長、毛利率三率走勢、本益比位階評估）。
- 給予客觀的觀點，說明當前公司面臨的機遇（如新產品、高毛利產品比重提升）與風險（如估值偏高、同業競爭）。
- 回答請使用繁體中文，文風要專業、語氣穩健。
"""
                    with st.spinner("AI 顧問正在研讀財報數據並編寫回覆中..."):
                        try:
                            response = model.generate_content(full_prompt)
                            st.write("---")
                            st.markdown("### 📝 AI 投資顧問的回覆：")
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"分析失敗: {e}")
                            
    with tab3:
        st.markdown("#### 📢 當年度法說會智慧分析")
        st.write("結合證交所重訊、公司官網及 [AlphaMemo](https://www.alphamemo.ai/free-transcripts) 法說逐字稿，使用 Gemini 聯網搜尋當年度法說會內容，挖掘瓶頸缺貨、營運動能好轉的產業與公司。")
        
        # 取得當前月份
        current_month = datetime.now().strftime("%Y-%m")
        current_year = current_month[:4]
        
        # 檢查快取
        cached_conf_report, report_time = get_gemini_report_details('investor_conferences', current_month)
        
        report_placeholder = st.empty()
        
        if cached_conf_report:
            st.success(f"已載入 {current_year} 年度的 AI 法說會智慧分析報告 (更新時間: {report_time})")
            report_placeholder.markdown(cached_conf_report)
            st.write("---")
            if st.button("🔄 重新分析並更新 (即時更新)", key='re_conf_report_button'):
                api_key = st.session_state.get('api_key_input')
                if not api_key:
                    st.error("請先在側邊欄配置您的 Gemini API Key！")
                else:
                    with st.spinner("Gemini 正在搜尋最新法說紀錄、逐字稿及重訊並編寫報告，這可能需要 20-30 秒..."):
                        report = analyze_investor_conferences(api_key, db_path=None)
                        if "失敗" in report or "未設定" in report or "error" in report.lower():
                            st.error(handle_gemini_error(report))
                        else:
                            report_placeholder.markdown(report)
                            st.success("✔ 報告已成功更新！")
        else:
            st.info(f"尚未產生 {current_year} 年度的法說會智慧分析報告。")
            if st.button("🔮 執行當年度法說會與產業展望分析", key='conf_report_button'):
                api_key = st.session_state.get('api_key_input')
                if not api_key:
                    st.error("請先在側邊欄配置您的 Gemini API Key！")
                else:
                    with st.spinner("Gemini 正在搜尋最新法說紀錄、逐字稿及重訊並編寫報告，這可能需要 20-30 秒..."):
                        report = analyze_investor_conferences(api_key, db_path=None)
                        if "失敗" in report or "未設定" in report or "error" in report.lower():
                            st.error(handle_gemini_error(report))
                        else:
                            report_placeholder.markdown(report)
                            st.success("✔ 報告已成功產生！")

# --- 5. 頁面：資料管理中心 ---
elif choice == "⚙️ 資料管理中心":
    st.markdown('<h1 class="gradient-text">⚙️ 資料管理中心</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtext">管理本地 SQLite 資料庫，執行資料爬蟲以獲取交易所最新數據，並對個股進行 AI 精細產業別的更新。</p>', unsafe_allow_html=True)
    
    # 顯示資料庫狀態
    st.markdown("### 📊 本地資料庫統計資訊")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("月營收資料筆數", f"{stats.get('monthly_revenue', 0):,} 筆")
    c2.metric("日估值 (PE) 資料筆數", f"{stats.get('daily_pe', 0):,} 筆")
    c3.metric("季度財務資料筆數", f"{stats.get('quarterly_financials', 0):,} 筆")
    
    c4, c5 = st.columns(2)
    c4.metric("AI 精細產業快取筆數", f"{stats.get('gemini_industry', 0):,} 筆")
    c5.metric("AI 報告快取筆數", f"{stats.get('gemini_reports', 0):,} 筆")
    
    # 顯示最新數據日期
    st.write("---")
    st.markdown("### 📅 目前資料庫最新期數")
    col_date1, col_date2, col_date3 = st.columns(3)
    col_date1.info(f"最新營收月份: **{get_latest_month() or '無資料'}**")
    col_date2.info(f"最新本益比日期: **{get_latest_pe_date() or '無資料'}**")
    latest_y, latest_q = get_latest_quarter()
    col_date3.info(f"最新財報季度: **{f'{latest_y} Q{latest_q}' if latest_y else '無資料'}**")
    
    st.markdown("""
    > 💡 **台灣上市櫃公司財報公告法定截止日提醒**：
    > *   **第一季季報 (Q1)**：當年度 **5/15** 前
    > *   **第二季半年報 (Q2)**：當年度 **8/14** 前
    > *   **第三季季報 (Q3)**：當年度 **11/14** 前
    > *   **第四季年度年報 (Q4)**：次年度 **3/31** 前
    > *(如遇假日，申報截止日將順延至下一個工作天。金控公司申報日可能不同)*
    """)
    
    # 資料爬蟲更新按鈕
    st.write("---")
    st.markdown("### 📥 資料更新與爬取操作")
    st.write("此操作將同時透過證交所與櫃買中心之 OpenAPI，以及**公開資訊觀測站 (MOPS) 實時 HTML 網頁**，抓取最新交易日的本益比與**當月每天最新公佈的實時月營收**。當天剛公佈的個股最新營收（如台積電等）皆能即時同步抓取更新！")
    
    if st.button("🔄 更新/爬取最新台股數據", key='crawl_button'):
        with st.spinner("正在執行資料爬蟲，請稍候... (這大約需要 30-60 秒)"):
            try:
                # 1. 爬取 PE 估值
                pe_records = fetch_daily_pe()
                if pe_records:
                    save_daily_pes(pe_records)
                    st.success(f"成功更新 {len(pe_records)} 筆每日 PE/PB/殖利率資料。")
                
                # 2. 爬取月營收
                revenue_records = fetch_monthly_revenue()
                if revenue_records:
                    save_monthly_revenues(revenue_records)
                    st.success(f"成功更新 {len(revenue_records)} 筆月營收資料。")
                    
                # 3. 爬取季報
                fin_records = fetch_quarterly_financials()
                if fin_records:
                    save_quarterly_financials(fin_records)
                    st.success(f"成功更新 {len(fin_records)} 筆季度財務損益資料。")
                
                st.success("🎉 所有數據已成功更新完成！")
                st.rerun()
            except Exception as e:
                st.error(f"資料更新失敗: {e}")
                
    # 歷史營收回補中心
    st.write("---")
    st.markdown("### 📅 歷史營收數據回補中心")
    st.write("您可以手動選擇時間範圍來回補過去月份的月營收數據。系統將會調用公開資訊觀測站的 HTML 歷史數據進行下載與解析，並儲存至本地資料庫中。")
    
    # 選擇年分與月份範圍
    c_bf1, c_bf2 = st.columns(2)
    with c_bf1:
        bf_start_date = st.date_input("開始年月", datetime(2026, 1, 1), min_value=datetime(2020, 1, 1), max_value=datetime.now())
    with c_bf2:
        bf_end_date = st.date_input("結束年月", datetime(2026, 5, 1), min_value=datetime(2020, 1, 1), max_value=datetime.now())
        
    if st.button("🚀 執行歷史營收回補", key='backfill_button'):
        start_year, start_month = bf_start_date.year, bf_start_date.month
        end_year, end_month = bf_end_date.year, bf_end_date.month
        
        # 產生所有年月對
        months_to_run = []
        curr_y, curr_m = start_year, start_month
        while (curr_y < end_year) or (curr_y == end_year and curr_m <= end_month):
            months_to_run.append((curr_y, curr_m))
            curr_m += 1
            if curr_m > 12:
                curr_m = 1
                curr_y += 1
                
        if not months_to_run:
            st.warning("開始時間不能晚於結束時間。")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            total_months = len(months_to_run)
            success_count = 0
            total_records_saved = 0
            
            for idx, (y, m) in enumerate(months_to_run):
                month_str = f"{y}-{m:02d}"
                status_text.markdown(f"⏳ 正在下載與解析 **{month_str}** 的月營收資料（請耐心等待，這需要連接公開資訊觀測站靜態網頁）...")
                
                try:
                    # 載入 crawler 歷史月度營收抓取函數
                    from crawler import fetch_historical_monthly_revenue
                    records = fetch_historical_monthly_revenue(y, m)
                    
                    if records:
                        save_monthly_revenues(records)
                        total_records_saved += len(records)
                        success_count += 1
                        st.write(f"✅ 成功寫入 **{month_str}** 共 {len(records)} 筆資料。")
                    else:
                        st.write(f"⚠️ **{month_str}** 爬取結果為空或無該月份網頁。")
                except Exception as ex:
                    st.write(f"❌ **{month_str}** 處理失敗: {ex}")
                    
                # 更新進度條
                progress_bar.progress((idx + 1) / total_months)
                
            status_text.markdown(f"🎉 回補完成！成功回補 **{success_count}** 個月份，共寫入 **{total_records_saved}** 筆營收資料。")
            st.rerun()
            
    # AI 產業分類觸發器
    st.write("---")
    st.markdown("### 🔮 AI 智慧次產業分類更新")
    st.write("本系統使用 Gemini API 將寬泛的行業類別進行次產業精細分類。分類完成後，可於「同業營收篩選」中開啟智慧次產業分類。")
    
    # 統計尚未進行分類的股票數量
    conn = get_connection()
    df_unrefined = pd.read_sql('''
        SELECT COUNT(DISTINCT stock_code) AS count 
        FROM monthly_revenue 
        WHERE stock_code NOT IN (SELECT stock_code FROM gemini_industry)
    ''', conn)
    conn.close()
    unrefined_count = df_unrefined.loc[0, 'count']
    
    st.write(f"目前還有 **{unrefined_count}** 檔股票尚未進行 Gemini 次產業精細分類。")
    
    # 輸入一次要跑幾檔
    batch_run_size = st.number_input("每次處理的股票數量（建議 30-50 檔，避免 API 超限）", min_value=10, max_value=100, value=30)
    
    if st.button("🔮 執行 AI 精細產業分類", key='refine_button'):
        api_key = st.session_state.get('api_key_input')
        if not api_key:
            st.error("請先在側邊欄配置您的 Gemini API Key！")
        else:
            with st.spinner("Gemini 正在分析並分類股票核心業務，請稍候..."):
                msg = refine_stock_industries(
                    api_key=api_key,
                    db_path=None,
                    batch_size=int(batch_run_size)
                )
                st.info(msg)
                st.rerun()
                

