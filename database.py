import sqlite3
import os
from datetime import datetime

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stocks.db')

def get_connection(db_path=DEFAULT_DB_PATH):
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DEFAULT_DB_PATH):
    """初始化 SQLite 資料庫，建立所有需要的資料表"""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 1. 月營收資料表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS monthly_revenue (
            date_month TEXT,              -- 格式: YYYY-MM
            stock_code TEXT,              -- 股票代號
            stock_name TEXT,              -- 股票名稱
            industry TEXT,                -- 原始產業別
            revenue REAL,                 -- 當月營收
            last_month_revenue REAL,      -- 上月營收
            last_year_revenue REAL,       -- 去年當月營收
            mom REAL,                     -- MoM %
            yoy REAL,                     -- YoY %
            cum_revenue REAL,             -- 當月累計營收
            cum_yoy REAL,                 -- 累計營收比較增減 %
            notes TEXT,                   -- 備註
            PRIMARY KEY (date_month, stock_code)
        )
    ''')
    
    # 2. 每日本益比、殖利率、股價淨值比資料表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_pe (
            date TEXT,                    -- 格式: YYYY-MM-DD
            stock_code TEXT,              -- 股票代號
            stock_name TEXT,              -- 股票名稱
            pe REAL,                      -- 本益比
            dy REAL,                      -- 殖利率
            pb REAL,                      -- 股價淨值比
            PRIMARY KEY (date, stock_code)
        )
    ''')
    
    # 3. 季損益表資料表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quarterly_financials (
            year INTEGER,                 -- 西元年度 (YYYY)
            quarter INTEGER,              -- 季別 (1, 2, 3, 4)
            stock_code TEXT,              -- 股票代號
            stock_name TEXT,              -- 股票名稱
            revenue REAL,                 -- 營業收入
            gross_profit REAL,            -- 營業毛利
            net_profit REAL,              -- 本期淨利 (歸屬母公司淨利或本期淨利)
            eps REAL,                     -- 基本每股盈餘 (EPS)
            gross_margin REAL,            -- 毛利率 (營業毛利 / 營業收入 * 100)
            net_margin REAL,              -- 淨利率 (本期淨利 / 營業收入 * 100)
            PRIMARY KEY (year, quarter, stock_code)
        )
    ''')
    
    # 4. Gemini 精細化產業別快取表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gemini_industry (
            stock_code TEXT PRIMARY KEY,  -- 股票代號
            stock_name TEXT,              -- 股票名稱
            refined_industry TEXT,        -- AI 精細分類產業別
            reason TEXT,                  -- 分類理由
            updated_at TEXT               -- 更新時間 (YYYY-MM-DD HH:MM:SS)
        )
    ''')
    
    # 5. Gemini 分析報告快取表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gemini_reports (
            report_type TEXT,             -- 報告類型: 'monthly_market', 'monthly_industry', 'quarterly_market'
            report_key TEXT,              -- 報告 Key (例如: '2026-04' 或 '2026-04_IC設計')
            report_content TEXT,          -- 報告 Markdown 內容
            updated_at TEXT,              -- 更新時間
            PRIMARY KEY (report_type, report_key)
        )
    ''')
    
    # 6. 外資與投信評等調整紀錄表
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
    conn.close()
    print(f"Database initialized at: {db_path}")

# --- 資料寫入 Helper 函數 ---

def save_monthly_revenues(records, db_path=DEFAULT_DB_PATH):
    """批次儲存/更新月營收資料"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.executemany('''
        INSERT INTO monthly_revenue (
            date_month, stock_code, stock_name, industry, revenue, 
            last_month_revenue, last_year_revenue, mom, yoy, 
            cum_revenue, cum_yoy, notes
        ) VALUES (
            :date_month, :stock_code, :stock_name, :industry, :revenue,
            :last_month_revenue, :last_year_revenue, :mom, :yoy,
            :cum_revenue, :cum_yoy, :notes
        ) ON CONFLICT(date_month, stock_code) DO UPDATE SET
            stock_name=excluded.stock_name,
            industry=excluded.industry,
            revenue=excluded.revenue,
            last_month_revenue=excluded.last_month_revenue,
            last_year_revenue=excluded.last_year_revenue,
            mom=excluded.mom,
            yoy=excluded.yoy,
            cum_revenue=excluded.cum_revenue,
            cum_yoy=excluded.cum_yoy,
            notes=excluded.notes
    ''', records)
    
    conn.commit()
    conn.close()

def save_daily_pes(records, db_path=DEFAULT_DB_PATH):
    """批次儲存/更新每日本益比基本面資料"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.executemany('''
        INSERT INTO daily_pe (
            date, stock_code, stock_name, pe, dy, pb
        ) VALUES (
            :date, :stock_code, :stock_name, :pe, :dy, :pb
        ) ON CONFLICT(date, stock_code) DO UPDATE SET
            stock_name=excluded.stock_name,
            pe=excluded.pe,
            dy=excluded.dy,
            pb=excluded.pb
    ''', records)
    
    conn.commit()
    conn.close()

def save_quarterly_financials(records, db_path=DEFAULT_DB_PATH):
    """批次儲存/更新季損益表資料"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.executemany('''
        INSERT INTO quarterly_financials (
            year, quarter, stock_code, stock_name, revenue, 
            gross_profit, net_profit, eps, gross_margin, net_margin
        ) VALUES (
            :year, :quarter, :stock_code, :stock_name, :revenue,
            :gross_profit, :net_profit, :eps, :gross_margin, :net_margin
        ) ON CONFLICT(year, quarter, stock_code) DO UPDATE SET
            stock_name=excluded.stock_name,
            revenue=excluded.revenue,
            gross_profit=excluded.gross_profit,
            net_profit=excluded.net_profit,
            eps=excluded.eps,
            gross_margin=excluded.gross_margin,
            net_margin=excluded.net_margin
    ''', records)
    
    conn.commit()
    conn.close()

def backfill_quarterly_financials_yfinance(stock_code, db_path=DEFAULT_DB_PATH):
    """
    從 yfinance 下載個股的歷史季度綜合損益表，並儲存到資料庫中。
    以此來補全資料庫中缺失的歷史季度數據（例如 2025Q4, 2025Q3 等）。
    """
    import yfinance as yf
    import pandas as pd
    
    ticker_code = f"{stock_code}.TW"
    try:
        t = yf.Ticker(ticker_code)
        df = t.quarterly_income_stmt
        if df.empty or len(df) == 0:
            ticker_code = f"{stock_code}.TWO"
            t = yf.Ticker(ticker_code)
            df = t.quarterly_income_stmt
    except Exception as e:
        print(f"Error fetching yfinance ticker {stock_code}: {e}")
        return False
        
    if df.empty or len(df) == 0:
        return False
        
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 撈取該股名稱
    cursor.execute("SELECT DISTINCT stock_name FROM monthly_revenue WHERE stock_code = ? LIMIT 1", (stock_code,))
    row_name = cursor.fetchone()
    stock_name = row_name['stock_name'] if row_name else ""
    
    eps_field = 'Basic EPS'
    if eps_field not in df.index and 'Diluted EPS' in df.index:
        eps_field = 'Diluted EPS'
        
    records_saved = 0
    
    for col in df.columns:
        try:
            dt = pd.to_datetime(col)
            year = dt.year
            quarter = (dt.month - 1) // 3 + 1
            
            revenue_val = df.loc['Total Revenue', col] if 'Total Revenue' in df.index else None
            gross_profit_val = df.loc['Gross Profit', col] if 'Gross Profit' in df.index else None
            net_profit_val = df.loc['Net Income', col] if 'Net Income' in df.index else None
            
            if pd.isna(revenue_val) or revenue_val is None:
                continue
                
            revenue = float(revenue_val) / 1000.0 if not pd.isna(revenue_val) else 0.0
            gross_profit = float(gross_profit_val) / 1000.0 if not pd.isna(gross_profit_val) else 0.0
            net_profit = float(net_profit_val) / 1000.0 if not pd.isna(net_profit_val) else 0.0
            
            eps = None
            if eps_field in df.index:
                eps_val = df.loc[eps_field, col]
                if not pd.isna(eps_val) and eps_val is not None:
                    eps = float(eps_val)
                    
            gross_margin = (gross_profit / revenue * 100) if revenue else 0.0
            net_margin = (net_profit / revenue * 100) if revenue else 0.0
            
            cursor.execute('''
                INSERT INTO quarterly_financials 
                (year, quarter, stock_code, stock_name, revenue, gross_profit, net_profit, eps, gross_margin, net_margin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(year, quarter, stock_code) DO UPDATE SET
                    stock_name=excluded.stock_name,
                    revenue=excluded.revenue,
                    gross_profit=excluded.gross_profit,
                    net_profit=excluded.net_profit,
                    eps=COALESCE(excluded.eps, quarterly_financials.eps),
                    gross_margin=excluded.gross_margin,
                    net_margin=excluded.net_margin
            ''', (
                year, quarter, stock_code, stock_name, 
                revenue, gross_profit, net_profit, eps, gross_margin, net_margin
            ))
            records_saved += 1
        except Exception as e:
            print(f"Error parsing quarter {col} for {stock_code}: {e}")
            
    conn.commit()
    conn.close()
    return records_saved > 0

def save_gemini_industry(stock_code, stock_name, refined_industry, reason, db_path=DEFAULT_DB_PATH):
    """儲存/更新單筆 Gemini 精細化產業分類"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
        INSERT INTO gemini_industry (
            stock_code, stock_name, refined_industry, reason, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(stock_code) DO UPDATE SET
            stock_name=excluded.stock_name,
            refined_industry=excluded.refined_industry,
            reason=excluded.reason,
            updated_at=excluded.updated_at
    ''', (stock_code, stock_name, refined_industry, reason, now_str))
    
    conn.commit()
    conn.close()

def save_gemini_report(report_type, report_key, report_content, db_path=DEFAULT_DB_PATH):
    """儲存/更新 Gemini 產出的 Markdown 分析報告"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
        INSERT INTO gemini_reports (
            report_type, report_key, report_content, updated_at
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT(report_type, report_key) DO UPDATE SET
            report_content=excluded.report_content,
            updated_at=excluded.updated_at
    ''', (report_type, report_key, report_content, now_str))
    
    conn.commit()
    conn.close()

# --- 資料查詢 Helper 函數 ---

def get_latest_month(db_path=DEFAULT_DB_PATH):
    """獲取資料庫中最新的營收月份 (格式: YYYY-MM)"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(date_month) FROM monthly_revenue')
    res = cursor.fetchone()
    conn.close()
    return res[0] if res else None

def get_latest_pe_date(db_path=DEFAULT_DB_PATH):
    """獲取資料庫中最新的本益比日期 (格式: YYYY-MM-DD)"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(date) FROM daily_pe')
    res = cursor.fetchone()
    conn.close()
    return res[0] if res else None

def get_latest_quarter(db_path=DEFAULT_DB_PATH):
    """獲取資料庫中最新的季報季度 (回傳: (year, quarter))"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT year, quarter FROM quarterly_financials ORDER BY year DESC, quarter DESC LIMIT 1')
    res = cursor.fetchone()
    conn.close()
    return (res['year'], res['quarter']) if res else (None, None)

def get_monthly_revenues_with_pe(date_month, db_path=DEFAULT_DB_PATH):
    """
    查詢特定月份的月營收資料，並合併該月最新的本益比 (PE)、殖利率 (DY)、股價淨值比 (PB) 
    以及 Gemini 的精細化產業分類
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 獲取資料庫中最新的營收月份
    cursor.execute('SELECT MAX(date_month) FROM monthly_revenue')
    max_month_res = cursor.fetchone()
    max_month = max_month_res[0] if max_month_res else None
    
    if date_month == max_month:
        # 如果是最新月份，不限制本益比日期，直接取得最新的估值資料
        limit_date = "9999-12-31"
    else:
        # 檢查該月份月底或之前是否有本益比記錄
        month_end_date = f"{date_month}-31"
        cursor.execute('''
            SELECT COUNT(1) FROM daily_pe 
            WHERE date <= ?
        ''', (month_end_date,))
        has_pe_before = cursor.fetchone()[0] > 0
        limit_date = month_end_date if has_pe_before else "9999-12-31"
    
    # 使用 Window Function ROW_NUMBER() 取得各個個股在限制日期前的最新一筆估值數據
    # 這樣可以解決上市 (TWSE) 與上櫃 (TPEx) 日本益比日期不對齊、以及歷史月份無當月 PE 數據的問題
    query = '''
        SELECT 
            m.date_month,
            m.stock_code,
            m.stock_name,
            m.industry AS original_industry,
            g.refined_industry,
            g.reason AS refined_reason,
            m.revenue,
            m.last_month_revenue,
            m.last_year_revenue,
            m.mom,
            m.yoy,
            m.cum_revenue,
            m.cum_yoy,
            m.notes,
            p.pe,
            p.dy,
            p.pb
        FROM monthly_revenue m
        LEFT JOIN gemini_industry g ON m.stock_code = g.stock_code
        LEFT JOIN (
            SELECT stock_code, pe, dy, pb, date,
                   ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY date DESC) as rn
            FROM daily_pe
            WHERE date <= ?
        ) p ON m.stock_code = p.stock_code AND p.rn = 1
        WHERE m.date_month = ?
    '''
    
    cursor.execute(query, (limit_date, date_month))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_quarterly_financials_list(year=None, quarter=None, db_path=DEFAULT_DB_PATH):
    """查詢季度財務資料，可選擇特定年份與季度"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    if year and quarter:
        cursor.execute('''
            SELECT * FROM quarterly_financials 
            WHERE year = ? AND quarter = ?
            ORDER BY stock_code
        ''', (year, quarter))
    else:
        cursor.execute('''
            SELECT * FROM quarterly_financials 
            ORDER BY year DESC, quarter DESC, stock_code
        ''')
        
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_refined_industries_map(db_path=DEFAULT_DB_PATH):
    """獲取所有已快取的 Gemini 精細化產業對照表 {stock_code: refined_industry}"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT stock_code, refined_industry FROM gemini_industry')
    res = {row['stock_code']: row['refined_industry'] for row in cursor.fetchall()}
    conn.close()
    return res

def get_gemini_report(report_type, report_key, db_path=DEFAULT_DB_PATH):
    """獲取快取的 AI 分析報告"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT report_content FROM gemini_reports 
        WHERE report_type = ? AND report_key = ?
    ''', (report_type, report_key))
    res = cursor.fetchone()
    conn.close()
    return res['report_content'] if res else None

def get_gemini_report_details(report_type, report_key, db_path=DEFAULT_DB_PATH):
    """獲取快取的 AI 分析報告及更新時間"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT report_content, updated_at FROM gemini_reports 
        WHERE report_type = ? AND report_key = ?
    ''', (report_type, report_key))
    res = cursor.fetchone()
    conn.close()
    return (res['report_content'], res['updated_at']) if res else (None, None)

def get_db_stats(db_path=DEFAULT_DB_PATH):
    """獲取資料庫各資料表筆數統計"""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if not os.path.exists(db_path):
        return {}
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    stats = {}
    tables = ['monthly_revenue', 'daily_pe', 'quarterly_financials', 'gemini_industry', 'gemini_reports']
    for t in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {t}")
            stats[t] = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            stats[t] = 0
            
    conn.close()
    return stats
