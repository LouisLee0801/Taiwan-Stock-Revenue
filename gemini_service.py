import os
import json
from datetime import datetime
import pandas as pd
import google.generativeai as genai
from google.generativeai import protos
from database import (
    get_connection, save_gemini_industry, save_gemini_report, get_gemini_report
)

def write_gemini_debug(msg):
    try:
        log_path = r"C:\Users\a0919\.gemini\antigravity\scratch\tw-stock-fundamental-analyzer\gemini_debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception as e:
        print(f"Failed to write debug log: {e}")

def get_gemini_model(api_key=None, model_name="gemini-2.5-flash", enable_search=False):
    """初始化並傳回適合的 Gemini 模式，以靜態且高效的方式選取最新可用型態"""
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        write_gemini_debug("get_gemini_model failed: No API Key provided.")
        return None
        
    # 主動映射已退役或不存在的模型到最新的穩定版模型 (gemini-2.5-flash)
    # 避免發起 list_models 網路請求以防止連線掛起
    if model_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-3.5-flash"]:
        model_name = "gemini-2.5-flash"
        
    key_preview = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "invalid_key"
    write_gemini_debug(f"Configuring Gemini Client. Key: {key_preview}, Target model: {model_name}, Search: {enable_search}")
    try:
        genai.configure(api_key=api_key)
        write_gemini_debug(f"Selected model name: {model_name}")
        # 如果需要啟用 Google Search Grounding
        if enable_search:
            return genai.GenerativeModel(
                model_name=model_name,
                tools=[protos.Tool(google_search=protos.Tool.GoogleSearch())]
            )
        else:
            return genai.GenerativeModel(model_name)
    except Exception as e:
        write_gemini_debug(f"Error configuring Gemini client: {e}")
        return None

def get_vertex_model(project_id=None, model_name="gemini-2.5-flash", enable_search=False):
    """相容性別名包裝，供現有函式呼叫使用"""
    return get_gemini_model(api_key=project_id, model_name=model_name, enable_search=enable_search)



# --- 1. 產業精細化分類 ---

def refine_stock_industries(api_key, db_path, batch_size=40):
    """
    從資料庫中找出尚未進行 Gemini 產業精細分類的個股，並分批發送給 Gemini 進行精細分類。
    將分類後的結果寫入 gemini_industry 資料表中。
    """
    model = get_vertex_model(api_key)
    if not model:
        return "Gemini API 金鑰未設定或初始化失敗，無法進行產業精細分類。"
    
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 找出 monthly_revenue 中有出現，但在 gemini_industry 中沒有紀錄的個股
    cursor.execute('''
        SELECT DISTINCT stock_code, stock_name, industry 
        FROM monthly_revenue 
        WHERE stock_code NOT IN (SELECT stock_code FROM gemini_industry)
        ORDER BY stock_code
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "所有個股都已完成產業精細分類，無需更新。"
        
    stocks_to_process = [{'code': r['stock_code'], 'name': r['stock_name'], 'org_industry': r['industry']} for r in rows]
    total_stocks = len(stocks_to_process)
    print(f"Total stocks to refine: {total_stocks}")
    
    refined_count = 0
    # 分批處理
    for i in range(0, total_stocks, batch_size):
        batch = stocks_to_process[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} stocks)...")
        
        # 建立 Prompt
        input_data = []
        for s in batch:
            input_data.append(f"代號: {s['code']} | 名稱: {s['name']} | 原產業別: {s['org_industry']}")
        
        input_text = "\n".join(input_data)
        
        prompt = f"""
你是一位熟悉台灣股市個股業務的基本面專家。
標準的證交所產業分類（如「電子零組件」、「半導體業」）往往過於籠統，無法精準區分公司核心業務。
請依據下面這批台灣個股的名稱與原分類，將它們精細地重新分類到現代且精準的「子產業別」（例如：晶圓代工、IC設計、AI伺服器、散熱模組、工業電腦、綠能/重電、光通訊、生技製藥、內需餐飲、金融控股等）。

請以繁體中文回答，並且**嚴格遵守 JSON 格式輸出**，回傳一個 JSON 陣列，每個物件包含以下欄位：
1. "code": 股票代號（字串）
2. "refined_industry": 精細分類後的子產業別名稱（字串，精簡有力，1-3個詞，如「IC設計」）
3. "reason": 精細分類的理由（字串，簡短說明核心業務，15-20字內，如「主要設計電源管理IC與USB傳輸晶片」）

輸入個股列表：
{input_text}
"""
        try:
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # 解析 JSON 內容
            result_list = json.loads(response.text)
            
            for item in result_list:
                code = str(item.get('code')).strip()
                refined_ind = str(item.get('refined_industry')).strip()
                reason = str(item.get('reason')).strip()
                
                # 尋找對應的名稱
                name = ""
                for s in batch:
                    if s['code'] == code:
                        name = s['name']
                        break
                
                if code and refined_ind:
                    save_gemini_industry(code, name, refined_ind, reason, db_path=db_path)
                    refined_count += 1
                    
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # 如果失敗，記錄錯誤but繼續處理下一批
            continue
            
    return f"產業精細分類完成！成功分類 {refined_count} / {total_stocks} 檔個股。"

# --- 2. 篩選顯著超越同業的個股與 AI 分析 ---

def analyze_industry_outliers(api_key, db_path, date_month, industry_name, use_refined=True):
    """
    篩選出在該月份中，YoY 與 MoM 顯著超越同產業中位數的個股，
    並將這些數據發送給 Gemini 分析是否具有持續性。
    """
    model = get_vertex_model(api_key)
    if not model:
        return "Gemini API 金鑰未設定或初始化失敗，無法產生同業篩選分析。"
        
    conn = get_connection(db_path)
    
    # 讀取該月營收，並合併本益比與精細產業別
    from database import get_monthly_revenues_with_pe
    raw_data = get_monthly_revenues_with_pe(date_month, db_path=db_path)
    
    if not raw_data:
        conn.close()
        return f"找不到 {date_month} 月份的營收資料，無法分析。"
        
    df = pd.DataFrame(raw_data)
    conn.close()
    
    # 決定使用原始產業別還是精細產業別
    ind_col = 'refined_industry' if use_refined else 'original_industry'
    
    # 過濾出特定產業的股票
    df_ind = df[df[ind_col] == industry_name].copy()
    if df_ind.empty:
        # 若精細產業為空，嘗試原始產業
        df_ind = df[df['original_industry'] == industry_name].copy()
        
    if len(df_ind) < 2:
        return f"產業 '{industry_name}' 個股數量過少，無法進行同業比較分析。"
        
    # 計算中位數與平均數
    median_yoy = df_ind['yoy'].median()
    median_mom = df_ind['mom'].median()
    
    # 篩選出同時大於中位數，且 YoY 或 MoM 超過中位數至少 15% 的優秀個股（且 YoY > 0）
    outliers = df_ind[
        (df_ind['yoy'] > 0) & 
        (df_ind['yoy'] > median_yoy) & 
        (df_ind['mom'] > median_mom) & 
        ((df_ind['yoy'] > median_yoy + 15) | (df_ind['mom'] > median_mom + 10))
    ].copy()
    
    if outliers.empty:
        # 如果篩選過於嚴格，退而求其次：YoY 大於中位數且 YoY > 10%
        outliers = df_ind[
            (df_ind['yoy'] > median_yoy) & (df_ind['yoy'] > 10)
        ].copy()
        
    if outliers.empty:
        return f"在 {date_month} 的 '{industry_name}' 產業中，沒有篩選出顯著超越同業中位數的個股。"
        
    # 準備發送給 Gemini 的資料
    # 取營收前 15 大或表現最好的前 10 檔做深析，避免 token 過長
    outliers = outliers.sort_values(by='yoy', ascending=False).head(10)
    
    outliers_text = []
    for _, row in outliers.iterrows():
        pe_str = f"{row['pe']:.1f}" if pd.notnull(row['pe']) else "N/A"
        outliers_text.append(
            f"- {row['stock_code']} {row['stock_name']}: YoY: {row['yoy']:.1f}%, MoM: {row['mom']:.1f}%, PE: {pe_str}, 當月營收: {row['revenue']/1000:.1f}百萬"
        )
        
    outliers_list_str = "\n".join(outliers_text)
    
    # 建立 Prompt
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
你是一位台股基本面分析師。
當前系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。
請分析在 **{date_month}** 期間，**{industry_name}** 產業中，以下幾檔營收成長顯著超出同業的「異軍突起個股」：

【同業中位數指標】
- 產業同業 YoY 中位數: {median_yoy:.1f}%
- 產業同業 MoM 中位數: {median_mom:.1f}%

【異軍突起個股名單】
{outliers_list_str}

請針對這些個股，撰寫一份**同業篩選分析報告**，包含：
1. **成長動能解析**：這幾檔個股的 YoY/MoM 顯著超越同業的可能原因（如新產能投產、獲得大客戶訂單、產業供應鏈位置等）。
2. **成長持續性評估**：區分哪些公司是「短期入帳高峰」（例如工程案完工、一次性認列），哪些是「結構性趨勢向上」（如 AI 需求爆發、新平台滲透率提升），並給予持續性評估（高/中/低）。
3. **本益比與評價合理性**：結合 PE (本益比) 資訊，分析哪些個股目前性價比較高，哪些可能有追高風險。
4. **結論與操作建議**：簡短的總結。

請以繁體中文撰寫，內容要專業、客觀，並使用 Markdown 格式呈現，多使用表格或列表來提升易讀性。
注意：請以當前時間視角來分析，避免提及過時的舊分析，專注於當前的實際狀況。
"""
    try:
        response = model.generate_content(prompt)
        report_content = response.text
        
        # 快取報告
        report_key = f"{date_month}_{industry_name}"
        save_gemini_report('monthly_industry', report_key, report_content, db_path=db_path)
        
        return report_content
    except Exception as e:
        return f"Gemini 產生報告失敗: {e}"

def analyze_monthly_market_trends(api_key, db_path, date_month):
    """
    彙整整個市場當月的營收表現（總額、成長排行、產業分布），
    由 Gemini 撰寫大盤整體營收解析與未來趨勢預測。
    """
    model = get_vertex_model(api_key)
    if not model:
        return "Gemini API 金鑰未設定或初始化失敗，無法產生大盤營收趨勢報告。"
        
    conn = get_connection(db_path)
    from database import get_monthly_revenues_with_pe
    raw_data = get_monthly_revenues_with_pe(date_month, db_path=db_path)
    
    if not raw_data:
        conn.close()
        return f"找不到 {date_month} 月份的營收資料，無法分析。"
        
    df = pd.DataFrame(raw_data)
    conn.close()
    
    # 計算大盤整體數據
    total_revenue = df['revenue'].sum()
    total_last_year_revenue = df['last_year_revenue'].sum()
    total_last_month_revenue = df['last_month_revenue'].sum()
    
    market_yoy = (total_revenue - total_last_year_revenue) / total_last_year_revenue * 100 if total_last_year_revenue else 0
    market_mom = (total_revenue - total_last_month_revenue) / total_last_month_revenue * 100 if total_last_month_revenue else 0
    
    # 以原始產業別計算各產業的 YoY 中位數，找出最強與最弱的產業
    ind_stats = df.groupby('original_industry').agg(
        median_yoy=('yoy', 'median'),
        median_mom=('mom', 'median'),
        count=('stock_code', 'count')
    )
    # 過濾出檔數大於 3 的產業才具代表性
    ind_stats_filtered = ind_stats[ind_stats['count'] >= 4]
    
    top_industries = ind_stats_filtered.sort_values(by='median_yoy', ascending=False).head(5)
    bottom_industries = ind_stats_filtered.sort_values(by='median_yoy', ascending=True).head(5)
    
    # 找出當月營收 MoM 成長最強與最弱的產業
    top_industries_mom = ind_stats_filtered.sort_values(by='median_mom', ascending=False).head(5)
    bottom_industries_mom = ind_stats_filtered.sort_values(by='median_mom', ascending=True).head(5)
    
    # 找出當月營收 YoY 成長最強的權值股
    large_caps = df[df['revenue'] >= 5000000].sort_values(by='yoy', ascending=False).head(10)
    
    # 格式化統計數據文字
    top_ind_text = []
    for ind, row in top_industries.iterrows():
        top_ind_text.append(f"- {ind}: YoY中位數 {row['median_yoy']:.1f}%, MoM中位數 {row['median_mom']:.1f}% (共{int(row['count'])}檔)")
    
    bottom_ind_text = []
    for ind, row in bottom_industries.iterrows():
        bottom_ind_text.append(f"- {ind}: YoY中位數 {row['median_yoy']:.1f}%, MoM中位數 {row['median_mom']:.1f}% (共{int(row['count'])}檔)")
        
    top_ind_mom_text = []
    for ind, row in top_industries_mom.iterrows():
        top_ind_mom_text.append(f"- {ind}: MoM中位數 {row['median_mom']:.1f}%, YoY中位數 {row['median_yoy']:.1f}% (共{int(row['count'])}檔)")
        
    bottom_ind_mom_text = []
    for ind, row in bottom_industries_mom.iterrows():
        bottom_ind_mom_text.append(f"- {ind}: MoM中位數 {row['median_mom']:.1f}%, YoY中位數 {row['median_yoy']:.1f}% (共{int(row['count'])}檔)")
        
    large_caps_text = []
    for _, row in large_caps.iterrows():
        large_caps_text.append(f"- {row['stock_code']} {row['stock_name']} ({row['original_industry']}): 當月營收 {row['revenue']/1000000:.1f}十億, YoY: {row['yoy']:.1f}%, MoM: {row['mom']:.1f}%")
        
    top_ind_str = "\n".join(top_ind_text)
    bottom_ind_str = "\n".join(bottom_ind_text)
    top_ind_mom_str = "\n".join(top_ind_mom_text)
    bottom_ind_mom_str = "\n".join(bottom_ind_mom_text)
    large_caps_str = "\n".join(large_caps_text)
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
你是一位專業的台股策略分析師與基本面專家。
當前系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。
請針對 **{date_month}** 月份台股全體上市櫃公司的營業收入彙總結果，撰寫一份專業的**台股月度營收大盤趨勢與展望報告**。

【大盤整體營收數據】
- 全市場總營收: {total_revenue/1000000:.1f} 十億元新台幣
- 全市場營收 YoY: {market_yoy:.2f}%
- 全市場營收 MoM: {market_mom:.2f}%

【當月最亮眼產業（YoY 年增率中位數前 5 名）】
{top_ind_str}

【當月最疲弱產業（YoY 年增率中位數後 5 名）】
{bottom_ind_str}

【當月月成長最強產業（MoM 月增率中位數前 5 名）】
{top_ind_mom_str}

【當月月成長最弱產業（MoM 月增率中位數後 5 名）】
{bottom_ind_mom_str}

【高營收大型權值股表現優異者】
{large_caps_str}

請撰寫報告，結構如下：
1. **大盤月度營收評論**：點評整體上市櫃營收的 YoY 與 MoM 成長狀況，說明當前台灣出口與製造業的景氣位階（擴張、復甦、或衰退放緩）。
2. **產業強弱對比與排行榜（YoY & MoM）**：
   - 請明確以 Markdown 表格或條列清單，完整列出當月最亮眼與最疲弱的產業排行榜（YoY 年增率前 5 名與後 5 名，含中位數數據）。
   - 請明確以 Markdown 表格或條列清單，完整列出當月月成長最強與最弱的產業排行榜（MoM 月增率前 5 名與後 5 名，含中位數數據）。
   - 詳細解析為何這些強勢產業（如 AI 供應鏈、先進封裝、半導體設備等）能維持高成長，以及最疲弱的產業面臨何種瓶頸（如庫存去化、需求不振）。
3. **領頭羊權值股評估**：分析大型權值股的營收暴增對大盤的指引意義。
4. **未來趨勢展望與投資操作指引**：展望未來 1 到 2 季，哪些板塊具有結構性趨勢（如 AI 供應鏈擴散、電子傳統旺季、新技術導入），哪些板塊需要避開？給予投資人具體策略指引。

請以繁體中文撰寫，文風要像投顧機構的首席策略報告，結構清晰、邏輯嚴密，並採用 Markdown 格式。
注意：請以當前時間視角來分析，避免提及過時的舊分析，專注於當前的實際狀況。
"""
    try:
        response = model.generate_content(prompt)
        report_content = response.text
        
        # 快取報告
        save_gemini_report('monthly_market', date_month, report_content, db_path=db_path)
        
        return report_content
    except Exception as e:
        return f"Gemini 產生大盤報告失敗: {e}"

# --- 4. 季度財務報告 AI 深度分析 ---

def analyze_quarterly_financial_trends(api_key, db_path, year, quarter):
    """
    彙總特定季度的公司財務數據（毛利率、淨利率、EPS 成長等），
    由 Gemini 撰寫季度財報獲利能力與毛利率分析報告。
    """
    model = get_vertex_model(api_key)
    if not model:
        return "Gemini API 金鑰未設定或初始化失敗，無法產生季度財報分析。"
        
    conn = get_connection(db_path)
    from database import get_quarterly_financials_list
    raw_data = get_quarterly_financials_list(year, quarter, db_path=db_path)
    
    if not raw_data:
        conn.close()
        return f"找不到 {year} 年 Q{quarter} 的財報資料，無法分析。"
        
    df = pd.DataFrame(raw_data)
    
    # 為了知道產業分類，我們從 monthly_revenue 找最新的產業別對照
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT stock_code, industry FROM monthly_revenue')
    ind_map = {row['stock_code']: row['industry'] for row in cursor.fetchall()}
    conn.close()
    
    df['industry'] = df['stock_code'].map(ind_map)
    df = df[df['industry'].notnull()].copy()
    
    # 計算產業毛利率、淨利率的中位數
    ind_margin = df.groupby('industry').agg(
        median_gm=('gross_margin', 'median'),
        median_nm=('net_margin', 'median'),
        median_eps=('eps', 'median'),
        count=('stock_code', 'count')
    )
    
    ind_margin_filtered = ind_margin[ind_margin['count'] >= 4]
    
    top_gm_ind = ind_margin_filtered.sort_values(by='median_gm', ascending=False).head(5)
    
    # 篩選獲利能力最強的前 10 檔個股
    top_eps_stocks = df.sort_values(by='eps', ascending=False).head(10)
    
    top_gm_text = []
    for ind, row in top_gm_ind.iterrows():
        top_gm_text.append(f"- {ind}: 毛利率中位數 {row['median_gm']:.1f}%, 淨利率中位數 {row['median_nm']:.1f}%")
        
    top_eps_text = []
    for _, row in top_eps_stocks.iterrows():
        top_eps_text.append(f"- {row['stock_code']} {row['stock_name']} ({row['industry']}): EPS {row['eps']:.2f}元, 毛利率 {row['gross_margin']:.1f}%, 淨利率 {row['net_margin']:.1f}%")
        
    top_gm_str = "\n".join(top_gm_text)
    top_eps_str = "\n".join(top_eps_text)
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    你是一位專精台股個股基本面與財務分析的專家。
    當前系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。
    請針對 **{year} 年第 {quarter} 季 (Q{quarter})** 全體上市櫃公司的季度財務報告結果，撰寫一份**台股季度財報獲利分析與三率展望報告**。
    
    【Q{quarter} 全體毛利率最高的前 5 個產業】
    {top_gm_str}
    
    【Q{quarter} 每股盈餘 (EPS) 最亮眼的前 10 名指標股】
    {top_eps_str}
    
    請撰寫報告，結構如下：
    1. **季度獲利能力綜述**：點評本季台股整體獲利表現，說明毛利率與淨利率的變化反映出企業面臨何種大環境（如台幣匯率波動、原物料價格、產能利用率）。
    2. **高毛利率/淨利率產業解析**：深度剖析為什麼這幾個產業的利潤率能居冠，其護城河在哪裡（例如技術門檻、訂價權、供應鏈稀缺性）。
    3. **營運槓桿與 EPS 異軍突起者評估**：分析 EPS 指標股的利潤成長來源（是營收規模擴大帶來的營運槓桿，還是業外收益，亦或是毛利率顯著提升）。
    4. **季報對月營收的指引與投資啟示**：投資人如何將季度報告的利潤率（落後指標）與最新的月度營業收入（領先指標）相結合，來篩選出「營收與毛利雙升」的成長股？展望下個季度，哪些板塊的毛利率有望持續擴張？
    
    請在報告中特別挑選幾檔你分析提到的具體指標個股，利用你的搜尋引擎工具即時檢索並在報告中明確列出它們當前的「估值狀況」，包含：目前本益比 (PE)、預估本益比 (Forward PE) 與股價淨值比 (PB)，並點評其合理性。
    
    請以繁體中文撰寫，內容要具備高度的財務專業度，使用 Markdown 格式呈現。
    注意：請以當前時間視角來分析，避免提及過時的舊分析，專注於當前的實際狀況。
    """
    try:
        model_with_search = get_vertex_model(api_key, enable_search=True)
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        
        # 快取報告
        report_key = f"{year}_Q{quarter}"
        save_gemini_report('quarterly_market', report_key, report_content, db_path=db_path)
        
        return report_content
    except Exception as e:
        return f"Gemini 產生季報報告失敗: {e}"

def get_latest_stock_price(stock_code):
    """從 yfinance 取得最新股價"""
    import yfinance as yf
    try:
        ticker = f"{stock_code}.TW"
        df = yf.download(ticker, period="5d", progress=False, timeout=5)
        if df.empty or len(df) == 0:
            ticker = f"{stock_code}.TWO"
            df = yf.download(ticker, period="5d", progress=False, timeout=5)
        if not df.empty:
            df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
            for val in reversed(df['Close'].tolist()):
                if val is not None and not pd.isna(val):
                    return float(val)
    except Exception as e:
        print(f"Error fetching stock price for {stock_code}: {e}")
    return None

def get_stock_details_from_gemini(api_key, stock_code, stock_name, db_path=None):
    """
    使用 Gemini (啟用 Google Search Grounding) 重新查詢個股的詳細資訊。
    包括：個股介紹、最近題材、小作文、法說會資訊、新聞，以及 Forward PE 估值分析。
    """
    error_msg = "未知聯網錯誤"
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定或初始化失敗，無法查詢個股詳細資訊。"
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
你是一位專業的台股投資顧問與產業分析師。
當前系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。
請針對個股 **{stock_code} {stock_name}**，使用搜尋引擎查詢最新（截至當前時間）的相關資訊，並為我撰寫一份「個股深度解析報告」。
 
請務必包含以下項目，不要套用千篇一律的模板，必須結合你搜尋到的真實具體資訊：
1. **個股介紹**：簡述該公司的核心業務、主要產品、以及在產業鏈中的角色與市佔率。
2. **最近題材**：分析該股近期最受市場矚目的題材（例如 AI、半導體先進製程、光通訊、重電等最新技術或訂單趨勢）。
3. **業務發展與市場討論**：彙整市場上針對該個股近期業務與接單情況的焦點討論（例如新技術進展、新客戶打入、產能產量變化等討論），並給予客觀的分析與評估。
4. **法說會與重要會議重點**：請明確指出最近一次公開法說會發生的具體時間（年月或日期），並整理法說會重點（包含營收展望、資本支出、毛利率預測、技術節點進度）。
5. **最新新聞與事件**：摘要過去數個月內對公司股價或營運有重大影響的媒體報導或重訊。
6. **財務與估值分析 (Forward PE)**：
   - 估計該公司過去半年（截至當前時間）的營收、毛利、淨利與 EPS 概況。
   - 根據市場目前的最新共識與展望，預估其 Forward PE（預估本益比），並點評目前估值水準是否合理、偏高或偏低。

請以繁體中文撰寫，字數約 800 - 1500 字，要求內容扎實、細緻、條理分明。使用 Markdown 格式呈現，多使用子標題、列表或對比表格來增強可讀性。
注意：請以當前時間視角來回答，搜尋最新資訊，避免提及「記憶體復甦已確立」等已在2024-2025年完成的陳舊分析（除非當前有最新數據），專注於當前的實際狀況。
"""
    try:
        write_gemini_debug(f"Sending stock details prompt to search-enabled model for {stock_code} {stock_name}...")
        response = model_with_search.generate_content(prompt)
        write_gemini_debug(f"Response received. Candidates count: {len(response.candidates) if response.candidates else 0}")
        if response.candidates:
            c = response.candidates[0]
            write_gemini_debug(f"Candidate finish reason: {c.finish_reason}")
            if hasattr(c, 'safety_ratings'):
                write_gemini_debug(f"Safety ratings: {c.safety_ratings}")
            if hasattr(c, 'grounding_metadata') and c.grounding_metadata:
                write_gemini_debug(f"Grounding metadata present: True")
        
        report_content = response.text
        if not report_content or not report_content.strip():
            raise ValueError("API returned empty content")
        if len(report_content.strip()) < 300 or "系統提示" in report_content:
            raise ValueError(f"API returned incomplete status: {report_content[:150]}")
        write_gemini_debug("Response text successfully retrieved.")
        return report_content
    except Exception as e:
        error_msg = str(e)
        write_gemini_debug(f"Search grounding failed for stock details: {error_msg}")
        
    # 如果聯網搜尋失敗或空內容，自動退回到無聯網的標準 Gemini 模式 (以本地資料庫數據進行分析)
    try:
        model_no_search = get_vertex_model(api_key, enable_search=False)
        if not model_no_search:
            return "Gemini API 金鑰未設定或初始化失敗，無法查詢個股詳細資訊。"
            
        # 從資料庫與 yfinance 中讀取即時數據作為 Context 傳給 Gemini
        conn = get_connection(db_path)
        df_rev = pd.read_sql('SELECT date_month, revenue, yoy, mom FROM monthly_revenue WHERE stock_code = ? ORDER BY date_month DESC LIMIT 6', conn, params=(stock_code,))
        df_fin = pd.read_sql('SELECT year, quarter, gross_margin, net_margin, eps FROM quarterly_financials WHERE stock_code = ? ORDER BY year DESC, quarter DESC LIMIT 4', conn, params=(stock_code,))
        df_pe = pd.read_sql('SELECT pe, pb, dy FROM daily_pe WHERE stock_code = ? ORDER BY date DESC LIMIT 1', conn, params=(stock_code,))
        conn.close()
        
        realtime_price = get_latest_stock_price(stock_code)
        price_str = f"{realtime_price} 元新台幣" if realtime_price else "目前無法取得即時市價"
        
        local_context = f"""
【本地資料庫與即時市價有關 {stock_code} {stock_name} 的財務數據】
1. 目前即時股價：{price_str} (由 yfinance 取得)
2. 最近 6 個月營收：
{df_rev.to_string(index=False)}

3. 最近 4 季獲利與利潤率：
{df_fin.to_string(index=False)}

4. 最新估值指標 (PE/PB/DY)：
{df_pe.to_string(index=False)}
"""
        fallback_prompt = f"""
你是一位專業的台股投資顧問與基本面分析師。
目前我們無法使用 Google 搜尋引擎聯網工具（可能是金鑰沒有 Search Grounding 聯網權限或餘額不足），我們將退回到基於本地資料庫數據的分析模式。
當前系統時間是：{current_date_str}。
請針對個股 **{stock_code} {stock_name}**，根據以下提供的本地資料庫數據，為我撰寫一份「個股基本面分析報告」：

{local_context}

請務必包含以下項目，要求內容扎實、條理分明：
1. **個股介紹與主要業務**：說明該公司屬於什麼產業，其核心業務是什麼。
2. **財務數據分析**：分析最近 6 個月營收年增與增減趨勢，以及最近 4 季 EPS、毛利率與淨利率走勢（說明是轉好、惡化還是持平）。
3. **估值評估 (PE/PB/DY)**：根據其目前的本益比 PE、淨值比 PB 與殖利率 DY，並結合即時股價 {price_str}，評估其目前的估值位階（偏高、合理或偏低）。
4. **結論與操作建議**：給予客觀的操作與基本面佈局建議。

請以繁體中文撰寫，以 Markdown 格式呈現。
"""
        response_fallback = model_no_search.generate_content(fallback_prompt)
        fallback_content = response_fallback.text
        if fallback_content and fallback_content.strip():
            search_model_used = getattr(model_with_search, 'model_name', 'unknown-model')
            notice = f"⚠️ **提示：API 聯網搜尋失敗（使用模型：`{search_model_used}`，詳細原因：`{error_msg}`），已自動退回使用「本地資料庫數據與即時市價」進行分析。若您的金鑰是付費版，請確認您的 Google Cloud 專案已開啟 Google Search Grounding API 並重新分析。**\n\n"
            return notice + fallback_content
        else:
            return "Gemini 查詢個股詳細資訊失敗: 聯網搜尋與本地基本面分析均未傳回內容。"
    except Exception as fallback_err:
        return f"Gemini 查詢個股詳細資訊失敗: 聯網搜尋失敗，且本地分析亦失敗: {fallback_err}"

def predict_turnarounds_with_gemini(api_key, industry_name, stock_financials_json):
    """
    傳入同業財務數據，讓 Gemini 分析預測其中即將或有潛力轉虧為盈的個股，並說明理由。
    傳回為 JSON 列表。
    """
    model = get_vertex_model(api_key)
    if not model:
        return []
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
你是一位熟悉台灣股市的基本面分析專家。
當前系統時間是：{current_date_str}。
請分析以下 **{industry_name}** 產業個股最近幾季的季度 EPS 以及當月營收 YoY 成長數據。
請從中篩選出「目前處於虧損（最新季度 EPS 負值或接近零），但未來極有潛力、即將、或已經開始轉虧為盈」的個股。

分析時請注意：
1. 觀察最新一季相較前幾季，虧損是否持續收斂（減虧中）。
2. 觀察月營收 YoY 是否轉正或高速增長，通常營收會領先獲利反映。

請回傳一個 JSON 陣列，每個物件包含以下欄位，**不要輸出額外的 Markdown 包裝或解釋**：
1. "code": 股票代號（字串）
2. "reason": AI 判斷該股有機會轉虧為盈的理由（字串，簡短 15-20 字內，如「最新季虧損收斂且營收年增率大於50%」）

輸入個股財務數據（JSON格式）：
{stock_financials_json}
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error predicting turnarounds: {e}")
        return []

def get_latest_stock_prices_batch(stock_codes):
    """批次從 yfinance 取得最新股價，回傳 {stock_code: price} 字典"""
    import yfinance as yf
    import pandas as pd
    res = {}
    if not stock_codes:
        return res
    tickers = []
    for c in stock_codes:
        tickers.append(f"{c}.TW")
        tickers.append(f"{c}.TWO")
    try:
        df = yf.download(tickers, period="5d", progress=False, timeout=10)
        if df.empty:
            return res
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]:
                close_df = df['Close']
            elif 'Close' in df.columns.levels[1]:
                close_df = df.xs('Close', axis=1, level=1)
            else:
                close_df = pd.DataFrame()
        else:
            close_df = df
            
        for code in stock_codes:
            price = None
            for t_suffix in [".TW", ".TWO"]:
                ticker = f"{code}{t_suffix}"
                if ticker in close_df.columns:
                    col_data = close_df[ticker]
                    for val in reversed(col_data.dropna().tolist()):
                         price = float(val)
                         break
                if price is not None:
                    break
            if price is not None:
                res[code] = price
    except Exception as e:
        print(f"Error fetching batch stock prices: {e}")
        # DO NOT fall back to individual yfinance queries to prevent hanging.
    return res

def analyze_turnaround_stocks(api_key, db_path=None):
    """
    找出資料庫中最近一季虧損（EPS <= 0.2）或股價偏低，但最新月營收 YoY 成長的潛力標的，
    結合過去 6 個月的月營收趨勢、法說會、新聞、TrendForce、經濟日報與工商時報等因素，
    調用 Gemini (啟用 Google Search Grounding) 聯網分析判斷下一季最有可能 EPS 虧轉盈或回到 10 元票面價值的個股。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定或初始化失敗，無法進行轉盈股分析。"
        
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 查詢候選股 (放寬 EPS 限制，並拉取更多候選以透過即時股價過濾)
    cursor.execute('''
        SELECT 
            m.stock_code, 
            m.stock_name, 
            m.industry AS original_industry, 
            m.yoy AS latest_yoy, 
            m.revenue AS latest_revenue, 
            q.eps AS latest_eps,
            q.year,
            q.quarter
        FROM (
            SELECT stock_code, stock_name, industry, yoy, revenue,
                   ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY date_month DESC) as rn
            FROM monthly_revenue
        ) m
        JOIN (
            SELECT stock_code, eps, year, quarter,
                   ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY year DESC, quarter DESC) as rn
            FROM quarterly_financials
        ) q ON m.stock_code = q.stock_code AND q.rn = 1
        WHERE m.rn = 1 AND (q.eps <= 0.2 OR m.stock_code IN (
            SELECT DISTINCT stock_code FROM daily_pe WHERE pb < 1.0 OR pe <= 0 OR pe IS NULL
        )) AND m.yoy > 5
        ORDER BY m.yoy DESC
        LIMIT 35
    ''')
    rows = cursor.fetchall()
    
    if not rows:
        conn.close()
        return "目前資料庫中沒有符合篩選條件的潛力轉盈個股。"
        
    # 批次查詢均線糾結度、即時股價與 20MA 趨勢
    tech_map = check_ma_convergence_batch([r['stock_code'] for r in rows])
    
    filtered_rows = []
    for r in rows:
        code = r['stock_code']
        tech_val = tech_map.get(code)
        
        success = False
        spread = 999.0
        mas = {}
        price = None
        is_20ma_rising = False
        
        if tech_val:
            if len(tech_val) == 4:
                success, spread, mas, price = tech_val
                is_20ma_rising = False
            else:
                success, spread, mas, price, is_20ma_rising = tech_val
            
        # 為了避免在 yfinance 被鎖時造成長時間掛起，只有在已篩選的數量小於 3 且沒拿到價格時才嘗試單獨查詢
        if price is None and len(filtered_rows) < 3:
            price = get_latest_stock_price(code)
            if price is not None:
                ind_res = check_ma_convergence(code)
                if len(ind_res) == 4:
                    ind_success, ind_spread, ind_mas, _ = ind_res
                    ind_is_20ma_rising = False
                else:
                    ind_success, ind_spread, ind_mas, _, ind_is_20ma_rising = ind_res
                if ind_success:
                    spread = ind_spread
                    is_20ma_rising = ind_is_20ma_rising
        
        # 篩選條件：1. 最新一季 EPS <= 0.2 或是 2. 股價低於 15 元
        if price is not None:
            if r['latest_eps'] <= 0.2 or price < 15:
                filtered_rows.append((r, price, spread, is_20ma_rising))
        else:
            # 如果依然沒有取得價格，只要滿足最新一季 EPS <= 0.2 依然保留
            if r['latest_eps'] <= 0.2:
                filtered_rows.append((r, None, 999.0, False))
                
    filtered_rows = filtered_rows[:12]
    
    candidates = []
    for r, price, spread, is_20ma_rising in filtered_rows:
        code = r['stock_code']
        # 查詢該股過去 6 個月營收趨勢
        cursor.execute('''
            SELECT date_month, revenue, yoy, mom 
            FROM monthly_revenue 
            WHERE stock_code = ? 
            ORDER BY date_month DESC 
            LIMIT 6
        ''', (code,))
        rev_history = cursor.fetchall()
        
        rev_history_str_list = []
        for rh in rev_history:
            rev_val = rh['revenue']
            yoy_val = rh['yoy']
            mom_val = rh['mom']
            
            rev_str = f"{rev_val/1000:.1f}" if rev_val is not None else "N/A"
            yoy_str = f"{yoy_val:.1f}" if yoy_val is not None else "N/A"
            mom_str = f"{mom_val:.1f}" if mom_val is not None else "N/A"
            
            rev_history_str_list.append(
                f"    - {rh['date_month']}: 營收 {rev_str}百萬 (YoY: {yoy_str}%, MoM: {mom_str}%)"
            )
        rev_history_str = "\n".join(rev_history_str_list)
        
        price_str = f"目前即時股價: {price} 元" if price is not None else "目前即時股價: N/A"
        
        tech_str = ""
        if price is not None and spread != 999.0:
            tech_str += f"  - 技術面現況：均線糾結度 (Spread) 為 {spread}%"
            if spread < 3.0:
                tech_str += " (🟢 均線強烈糾結整理中)"
            elif spread <= 5.0:
                tech_str += " (🟡 均線輕微收斂中)"
            else:
                tech_str += " (⚪ 均線尚未糾結)"
            tech_str += f"；20MA 趨勢為 {'📈 向上 (20MA呈上升通道)' if is_20ma_rising else '📉 向下或持平'}\n"
        else:
            tech_str += "  - 技術面現況：暫無即時均線與股價數據\n"
            
        candidates.append(
            f"- {r['stock_code']} {r['stock_name']} ({price_str}):\n"
            f"  - 最新一季報 EPS: {r['latest_eps']} 元\n"
            f"  - 過去 6 個月月營收趨勢:\n{rev_history_str}\n"
            f"{tech_str}"
        )
    conn.close()
    candidates_str = "\n".join(candidates)
    
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
你是一位專業的台股投資顧問與基本面策略分析師。
當前系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。

我們從資料庫與即時均線數據中篩選出了可能具有轉盈潛力（最新一季 EPS 處於微利或虧損狀態且營收 YoY 成長，或者股價低於票面價值 10 元）的個股名單，以及它們過去 6 個月的月營收趨勢與技術指標現況：
{candidates_str}

請利用你的 Google 搜尋引擎聯網工具，深入查詢這幾檔個股的最新業務現況、產業動態、媒體報導與法說會內容，並撰寫一份專業的**台股潛力轉虧為盈個股深度解析報告**。

⚠️ 重要特別規定（請嚴格遵守）：
1. 請確保搜尋並輸出之「股票代號與公司名稱必須完全精確對應」，絕不可對錯或胡亂拼湊（例如：3321 是同泰，不是穎漢科技；穎漢科技是 4562；台積電是 2330）。如果發生代碼張冠李戴，後端程式將在量化比對時發生嚴重的邏輯衝突。
2. 為了能讓程式系統自動提取代碼進行實時技術分析，請「務必」以方括號包圍 4 位數台股代碼（例如 `[2330]`、`[3321]` 等）。請「絕對不要」遮蔽、替代或隱藏股票代號（例如：不要寫成 [23XX]、[3XXX]、[XXXX] 或是 [xxxx]）。

報告撰寫要求與重點：
1. **月營收趨勢分析**：結合各別公司過去數月的**月營收變化趨勢**（是正在加速成長、築底向上還是波段起伏）。
2. **多重來源比對**：指定研判與比對 **TrendForce 研報（特別是涉及半導體/面板/記憶體等產業）、法說會指引、重大新聞，以及《經濟日報》、《工商時報》的最新報導與評論**。
3. **技術面結合評估**：請對比分析這些個股的**均線糾結狀態**（是否已在底部高度糾結，代表籌碼洗乾淨）與 **20MA 趨勢是否已經向上**（代表短中期趨勢正式走揚、主力開始拉抬）。說明技術面均線收斂且 20MA 向上是否能與營收回溫形成「基本面+技術面雙重起漲點」。
4. **虧轉盈與票面價判斷**：判斷下一季哪些股票最有可能實現 **EPS 虧轉盈**（EPS 由負轉正），以及哪些目前股價低於 10 元票面價值的股票最有可能**回到/站上 10 元票面價值**。

報告內容應包括：
1. **整體轉盈趨勢與技術/基本雙共振評估**：簡述營收領先獲利反映的商業邏輯，並說明均線糾結與 20MA 向上如何代表籌碼面和技術面與基本面的同步共振。
2. **轉盈與技術面評估綜合表**：請以 Markdown 表格列出所有分析的個股，欄位包含：
   - 股票代號與名稱（必須對應正確，如 `[3321] 同泰`）
   - 目前股價
   - 最新一季 EPS
   - 均線糾結狀態 (強烈糾結/輕微收斂/未糾結)
   - 20MA 趨勢 (向上/向下/持平)
   - 預估下一季是否能 EPS 虧轉盈 (預估 EPS)
   - 預估是否能回到/站上票面價值 10 元 (是/否/已高於 10 元)
   - 潛力評級 (高/中/低)
   - 主要評估與技術依據摘要
3. **個股逐一深度剖析**：針對表格中潛力評級為「高」或「中」的 4-6 檔核心個股進行深入檢索：
   - 說明其核心業務、營收暴增原因，並融合技術面（如均線糾結、20MA向上）的走勢點評。
   - 分析其最近一次公開法說會的重點與管理層對轉盈與股價重回票面的時程展望，明確標示法說會時間。
   - **估值點評**：利用搜尋檢索其當前的預估本益比 (Forward PE)、PB 淨值比，評估目前股價是否已過度反映轉盈預期。
4. **風險提示與操作建議**：操作此類基本+技術面共振轉盈股的策略與防範陷阱。
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if not report_content or not report_content.strip():
            raise ValueError("API returned empty content")
        if len(report_content.strip()) < 300 or "系統提示" in report_content:
            raise ValueError(f"API returned incomplete status: {report_content[:150]}")
        # 快取報告
        save_gemini_report('turnaround_list', current_date_str[:7], report_content, db_path=db_path)
        return report_content
    except Exception as e:
        error_msg = str(e)
        print(f"Search grounding failed for turnaround stocks: {error_msg}. Falling back...")
        try:
            model_no_search = get_vertex_model(api_key, enable_search=False)
            if not model_no_search:
                return f"Gemini 產生轉盈股分析報告失敗: {e}"
            fallback_prompt = prompt + "\n\n⚠️ 提示：由於聯網搜尋工具目前不可用，請直接根據上述提供的本地資料庫數據（EPS、月營收 YoY 等）與您對這些公司的認知進行基本面深度分析。"
            response_fallback = model_no_search.generate_content(fallback_prompt)
            fallback_content = response_fallback.text
            if fallback_content and fallback_content.strip():
                save_gemini_report('turnaround_list', current_date_str[:7], fallback_content, db_path=db_path)
                notice = f"⚠️ **提示：API 聯網搜尋失敗（詳細原因：`{error_msg}`），已自動退回使用「本地資料庫數據與即時市價」進行分析。若您的金鑰是付費版，請確認您的 Google Cloud 專案已開啟 Google Search Grounding API 並重新分析。**\n\n"
                return notice + fallback_content
            else:
                return f"Gemini 產生轉盈股分析報告失敗: {e}，且備用本地分析亦無回應。"
        except Exception as fallback_err:
            return f"Gemini 產生轉盈股分析報告失敗: {e}，且備用本地分析發生錯誤: {fallback_err}"

def check_ma_convergence(stock_code):
    """
    計算特定個股的均線糾結程度。
    回傳: (success: bool, spread: float, ma_dict: dict, current_price: float)
    """
    import yfinance as yf
    import pandas as pd
    try:
        # 下載過去 90 天以確保能計算出 60MA
        ticker = f"{stock_code}.TW"
        df = yf.download(ticker, period="90d", progress=False, timeout=5)
        if df.empty or len(df) == 0:
            ticker = f"{stock_code}.TWO"
            df = yf.download(ticker, period="90d", progress=False, timeout=5)
            
        if df.empty or len(df) < 60:
            return False, 999.0, {}, 0.0, False
            
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        close_series = df['Close']
        
        # 計算均線，並轉成一般 float 避免 serialization 錯誤
        ma5 = float(close_series.rolling(5).mean().iloc[-1])
        ma10 = float(close_series.rolling(10).mean().iloc[-1])
        ma20 = float(close_series.rolling(20).mean().iloc[-1])
        ma60 = float(close_series.rolling(60).mean().iloc[-1])
        
        current_price = float(close_series.iloc[-1])
        
        if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20) or pd.isna(ma60) or pd.isna(current_price):
            return False, 999.0, {}, 0.0, False
            
        # 計算 20MA 是否向上 (比較今日與 5 天前的 20MA)
        is_20ma_rising = False
        if len(close_series) >= 25:
            ma20_prev = float(close_series.rolling(20).mean().iloc[-5])
            ma20_curr = float(close_series.rolling(20).mean().iloc[-1])
            is_20ma_rising = ma20_curr > ma20_prev
            
        # 計算糾結度 (百分比價差比)
        mas = [ma5, ma10, ma20, ma60]
        max_ma = max(mas)
        min_ma = min(mas)
        spread = ((max_ma - min_ma) / current_price) * 100
        
        ma_dict = {
            '5MA': round(ma5, 2),
            '10MA': round(ma10, 2),
            '20MA': round(ma20, 2),
            '60MA': round(ma60, 2)
        }
        return True, round(spread, 2), ma_dict, round(current_price, 2), is_20ma_rising
    except Exception as e:
        print(f"Error checking MA convergence for {stock_code}: {e}")
        return False, 999.0, {}, 0.0, False

def check_ma_convergence_batch(stock_codes):
    """
    批次計算多個個股的均線糾結程度。
    回傳: {stock_code: (success, spread, ma_dict, price)}
    """
    import yfinance as yf
    import pandas as pd
    res = {}
    if not stock_codes:
        return res
    tickers = []
    for c in stock_codes:
        tickers.append(f"{c}.TW")
        tickers.append(f"{c}.TWO")
    try:
        df = yf.download(tickers, period="90d", progress=False, timeout=10)
        if df.empty:
            return res
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]:
                close_df = df['Close']
            elif 'Close' in df.columns.levels[1]:
                close_df = df.xs('Close', axis=1, level=1)
            else:
                close_df = pd.DataFrame()
        else:
            close_df = df[['Close']] if 'Close' in df.columns else pd.DataFrame()
            
        for code in stock_codes:
            success = False
            spread = 999.0
            ma_dict = {}
            current_price = 0.0
            is_20ma_rising = False
            
            for t_suffix in [".TW", ".TWO"]:
                ticker = f"{code}{t_suffix}"
                if ticker in close_df.columns:
                     series = close_df[ticker].dropna()
                     if len(series) >= 60:
                         try:
                             ma5 = float(series.rolling(5).mean().iloc[-1])
                             ma10 = float(series.rolling(10).mean().iloc[-1])
                             ma20 = float(series.rolling(20).mean().iloc[-1])
                             ma60 = float(series.rolling(60).mean().iloc[-1])
                             price = float(series.iloc[-1])
                             
                             mas = [ma5, ma10, ma20, ma60]
                             max_ma = max(mas)
                             min_ma = min(mas)
                             spread_val = ((max_ma - min_ma) / price) * 100
                             
                             # 計算 20MA 是否向上 (比較今日與 5 天前的 20MA)
                             is_20ma_rising_val = False
                             if len(series) >= 25:
                                 ma20_prev = float(series.rolling(20).mean().iloc[-5])
                                 ma20_curr = float(series.rolling(20).mean().iloc[-1])
                                 is_20ma_rising_val = ma20_curr > ma20_prev
                             
                             ma_dict = {
                                 '5MA': round(ma5, 2),
                                 '10MA': round(ma10, 2),
                                 '20MA': round(ma20, 2),
                                 '60MA': round(ma60, 2)
                             }
                             current_price = round(price, 2)
                             spread = round(spread_val, 2)
                             is_20ma_rising = is_20ma_rising_val
                             success = True
                             break
                         except Exception as e:
                             print(f"Error calculating MA for {code}: {e}")
            res[code] = (success, spread, ma_dict, current_price, is_20ma_rising)
    except Exception as e:
        print(f"Error checking batch MA convergence: {e}")
        # DO NOT fall back to individual yfinance queries to prevent hanging.
        for code in stock_codes:
            res[code] = (False, 999.0, {}, 0.0, False)
    return res

def analyze_chip_and_ma_convergence(api_key, db_path=None):
    """
    使用 Gemini (啟用 Google Search Grounding) 聯網搜尋近期有「特定分點買超/收購、籌碼集中度上升」的台股標的。
    回傳一個包含分析報告與候選股代碼的結果。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定或初始化失敗，無法進行籌碼分析。"

    current_date_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
你是一位專業的台股籌碼面與技術面分析專家。
當前系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。

請利用你的 Google 搜尋引擎聯網工具，搜尋近期一個月內台灣股市中，關於「特定分點買超」、「主力分點囤貨」、「籌碼集中度上升」或「主力鎖碼」的相關財經新聞、分析報導與市場討論。

你可以搜尋如下主題：
1. 近期有特定證券商分點（如：關鍵大戶、主力分點或外資/投信特定券商）在過去數週內持續買進囤貨的股票。
2. 近期籌碼集中度上升，且技術面上短期與中長期均線（5MA、10MA、20MA、60MA）開始糾結整理（價差比收斂）的個股。

請根據搜尋到的最新網路資訊（如財經新聞、媒體報導、研究評論等），撰寫一份**【台股籌碼集中與均線糾結股】深度解析報告**，包含：
1. **籌碼面大局觀**：說明主力分點囤貨與籌碼集中度上升對股價後市的意義。
2. **籌碼囤貨指標股分析**：請列出至少 6-8 檔近期被提及「特定分點買超、籌碼集中、股價趨於穩定」的具體股票。針對每檔股票，必須說明：
   - 股票代號（4位數字）與名稱。
   - 買超該股的**特定分點名稱**（例如：台灣摩根士丹利、元大松山、凱基台北等，若搜尋到具體分點請明確指出，否則可說明主力券商）。
   - 主力收購的動機或近期利多題材（如新接單、大戶鎖碼等）。
   - 技術面整理狀況（均線是否已糾結、成交量是否萎縮）。
3. **操作建議與風險提示**：如何利用主力成本線佈局，以及防範主力出貨風險。

⚠️ 重要特別規定（請嚴格遵守）：
1. 為了能讓程式系統自動提取代碼進行實時均線與股價的量化計算，請「務必」提供真實且正確的 4 位數字台股代號（例如 `[2330]`、`[2061]`、`[3231]` 等），請務必以方括號包圍。
2. 請「絕對不要」出於合規、隱私或任何保護目的而遮蔽、替代或隱藏股票代號（例如：不要寫成 [23XX]、[3XXX]、[XXXX]、或使用英文代號如 [xxxx]）。一旦遮蔽代碼，後端程式將無法在台灣證券交易所與櫃買中心進行即時報價與技術均線比對，這會導致整個系統報表顯示空白。
3. 報告內容應注重客觀客觀資訊的彙整與分析，但必須提供真實股票的代碼以便對接程式做技術分析。

請以繁體中文撰寫，內容要專業、客觀，並使用 Markdown 格式。
注意：請以當前時間視角來分析，避免提及陳舊分析，專注於當前的實際狀況。
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if not report_content or not report_content.strip():
            raise ValueError("API returned empty content")
        if len(report_content.strip()) < 300 or "系統提示" in report_content:
            raise ValueError(f"API returned incomplete status: {report_content[:150]}")
        # 快取報告
        save_gemini_report('chip_and_ma_convergence', current_date_str[:7], report_content, db_path=db_path)
        return report_content
    except Exception as e:
        error_msg = str(e)
        print(f"Search grounding failed for chip analysis: {error_msg}. Falling back...")
        try:
            model_no_search = get_vertex_model(api_key, enable_search=False)
            if not model_no_search:
                return f"Gemini 產生籌碼分析報告失敗: {e}"
                
            fallback_prompt = prompt + "\n\n⚠️ 提示：由於聯網搜尋工具目前不可用，請根據您對近期台股（如最近幾季）籌碼集中、主力分點囤貨的認知，並結合均線糾結的技術特徵，為我撰寫這份報告。"
            response_fallback = model_no_search.generate_content(fallback_prompt)
            fallback_content = response_fallback.text
            if fallback_content and fallback_content.strip():
                save_gemini_report('chip_and_ma_convergence', current_date_str[:7], fallback_content, db_path=db_path)
                notice = f"⚠️ **提示：API 聯網搜尋失敗（詳細原因：`{error_msg}`），已自動退回使用「備用 AI 模型知識庫」進行分析。若您的金鑰是付費版，請確認您的 Google Cloud 專案已開啟 Google Search Grounding API 並重新分析。**\n\n"
                return notice + fallback_content
            else:
                return f"Gemini 產生籌碼分析報告失敗: {e}，且備用本地分析亦無回應。"
        except Exception as fallback_err:
            return f"Gemini 產生籌碼分析報告失敗: {e}，且備用本地分析發生錯誤: {fallback_err}"

def extract_valid_stock_codes(text, db_path=None):
    """
    從分析報告文字中利用正則表達式尋找 [股票代號] 或四位數字，並比對資料庫是否為有效個股。
    """
    import re
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT stock_code FROM monthly_revenue")
    valid_codes = {row['stock_code'] for row in cursor.fetchall()}
    conn.close()
    
    # 尋找所有 4 位數字
    candidates = re.findall(r'\b\d{4}\b', text)
    # 尋找所有帶方括號的 [2330] 格式
    candidates_bracket = re.findall(r'\[(\d{4})\]', text)
    
    all_found = set(candidates + candidates_bracket)
    
    # 過濾出在資料庫中真正存在的個股代碼 (避開年份如 2024, 2025, 2026 等)
    valid_found = [code for code in all_found if code in valid_codes]
    return sorted(list(valid_found))


def analyze_investor_conferences(api_key, db_path=None):
    """
    使用 Gemini (啟用 Google Search Grounding) 聯網查詢並整理所有上市上櫃公司當年度的法說內容。
    分析接下來的看點，包括哪些公司可能面臨產能瓶頸、缺貨，或哪些公司營收會轉好。
    數據源除證交所與公司官網外，亦包含指定查詢：https://www.alphamemo.ai/free-transcripts
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定或初始化失敗，無法產生法說會分析。"
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
你是一位專業的台股宏觀產業與個股研究員。
當前系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。
請使用搜尋引擎，檢索並彙整台灣上市上櫃公司在當年度的最新「法說會 (Investor Conference) 內容與產業展望紀錄」。

請特別針對以下幾點進行深度整合分析：
1. **主要看點與亮點**：今年度最受市場矚目的產業板塊（如 AI 伺服器、先進封裝 CoWoS、光通訊 CPO、重電與綠能、散熱等）的最新法說展望要點與成長指引。
2. **產能瓶頸與缺貨公司**：哪些公司在法說會中明確提到面臨「產能瓶頸」、「缺貨/供不應求」或「設備交期過長」，導致後續可能漲價或營收受限（如台積電先進封裝、特定先進材料或零組件廠商）？
3. **營運轉好與谷底復甦公司**：哪些公司或產業在法說會中明確指出「庫存去化結束」、「需求回溫」或「即將轉虧為盈/谷底翻揚」，營收有望迎來爆發？
4. **具體個股展望點評**：結合法說會內容，點評至少 5-8 檔核心指標個股（例如台積電、聯發科、廣達、信驊、鴻海等或其他中小型關鍵廠商），並給出其當前的 PE/Forward PE 估值與 PB 概況。

請在搜尋與彙整時，**指定查詢包括「臺灣證券交易所 (TWSE) 重訊、各家公司官網法說會簡報，以及 https://www.alphamemo.ai/free-transcripts 上的法說逐字稿與摘要資訊」**。

請使用繁體中文撰寫一份專業的**當年度法說會總體產業與個股大解析報告**。內容要充實、有邏輯，以 Markdown 格式呈現，多使用子標題、加粗與表格展示。
注意：請以當前時間視角來分析，避免提及過時的舊分析，專注於當前的實際狀況。
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if not report_content or not report_content.strip():
            raise ValueError("API returned empty content")
        if len(report_content.strip()) < 300 or "系統提示" in report_content:
            raise ValueError(f"API returned incomplete status: {report_content[:150]}")
        save_gemini_report('investor_conferences', current_date_str[:7], report_content, db_path=db_path)
        return report_content
    except Exception as e:
        error_msg = str(e)
        print(f"Search grounding failed for investor conferences: {error_msg}. Falling back...")
        try:
            model_no_search = get_vertex_model(api_key, enable_search=False)
            if not model_no_search:
                return f"Gemini 產生法說會分析報告失敗: {e}"
                
            # 取得主要指標股即時市價以利估值計算
            indicator_stocks = {
                '2330': '台積電',
                '2454': '聯發科',
                '2317': '鴻海',
                '2382': '廣達',
                '6669': '緯穎',
                '5274': '信驊'
            }
            prices_str_list = []
            for code, name in indicator_stocks.items():
                p = get_latest_stock_price(code)
                if p:
                    prices_str_list.append(f"- {code} {name}: 目前即時股價約 {p} 元")
            prices_context = "\n".join(prices_str_list)
            
            fallback_prompt = prompt + f"\n\n【最新指標股即時股價資訊 (供 Forward PE 估值參考)】:\n{prices_context}\n\n⚠️ 提示：由於聯網搜尋工具目前不可用，請直接根據上述提供的最新即時股價，以及您對這些公司法說會與估值展望的知識，為我撰寫這份報告。"
            response_fallback = model_no_search.generate_content(fallback_prompt)
            fallback_content = response_fallback.text
            if fallback_content and fallback_content.strip():
                save_gemini_report('investor_conferences', current_date_str[:7], fallback_content, db_path=db_path)
                notice = f"⚠️ **提示：API 聯網搜尋失敗（詳細原因：`{error_msg}`），已自動退回使用「AI 產業模型預訓練知識與指標股即時市價」進行分析。若您的金鑰是付費版，請確認您的 Google Cloud 專案已開啟 Google Search Grounding API 並重新分析。**\n\n"
                return notice + fallback_content
            else:
                return f"Gemini 產生法說會分析報告失敗: {e}，且備用本地分析亦無回應。"
        except Exception as fallback_err:
            return f"Gemini 產生法說會分析報告失敗: {e}，且備用本地分析發生錯誤: {fallback_err}"

def analyze_surging_stocks(api_key, db_path=None):
    """
    從資料庫找出當月營收 YoY/MoM 異軍突起的個股，
    並由 Gemini AI 聯網查詢其爆發原因，產出深度分析報告。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定或初始化失敗，無法進行分析。"
        
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 撈取當月最新營收且 YoY 成長強勁的個股 (包括大部分行業，篩選 YoY > 30% 且具備一定規模)
    cursor.execute('''
        SELECT stock_code, stock_name, industry, revenue, yoy, mom 
        FROM monthly_revenue 
        WHERE date_month = (SELECT MAX(date_month) FROM monthly_revenue)
          AND yoy > 25 AND revenue > 20000
        ORDER BY yoy DESC 
        LIMIT 15
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "目前資料庫中沒有符合篩選條件的異軍突起個股。"
        
    # 批次查詢均線糾結度、即時股價與 20MA 趨勢
    tech_map = check_ma_convergence_batch([r['stock_code'] for r in rows])
    
    candidates = []
    for r in rows:
        code = r['stock_code']
        rev_val = r['revenue']
        yoy_val = r['yoy']
        mom_val = r['mom']
        rev_str = f"{rev_val/1000:.1f}" if rev_val is not None else "N/A"
        yoy_str = f"{yoy_val:.1f}" if yoy_val is not None else "N/A"
        mom_str = f"{mom_val:.1f}" if mom_val is not None else "N/A"
        
        tech_val = tech_map.get(code)
        tech_str = "無法取得即時技術指標"
        if tech_val:
            if len(tech_val) == 4:
                success, spread, mas, price = tech_val
                is_20ma_rising = False
            else:
                success, spread, mas, price, is_20ma_rising = tech_val
            
            if success:
                trend_str = "20MA趨勢向上" if is_20ma_rising else "20MA趨勢向下/持平"
                tech_str = f"實時股價 {price}元, 均線糾結度 {spread}%, {trend_str}"
                
        candidates.append(
            f"- {code} {r['stock_name']} ({r['industry']}): 當月營收 {rev_str}百萬 (YoY: {yoy_str}%, MoM: {mom_str}%) | 技術狀態: {tech_str}"
        )
    candidates_str = "\n".join(candidates)
    
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
你是一位專業的台股投資策略分析師與產業研究員，專精於結合基本面營收暴增、籌碼面主力囤貨與技術面均線收斂起漲的「三合一共振」選股策略。
當前系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。

我們從資料庫中篩選出了當前月份營收表現「異軍突起」（營收金額較高且年成長率 YoY 非常巨大）的台股候選名單：
{candidates_str}

請利用你的 Google 搜尋引擎聯網工具，深入查詢這幾檔個股近期營收暴增（飆升）的具體核心原因，並特別結合**【籌碼面主力分點買超/集中度上升】與【技術面均線高度糾結收斂後起漲/突破】的圖形特徵**進行綜合研判，撰寫一份專業的**【台股當月營收異軍突起與籌碼技術三合一深度解析報告】**。

特別注意（請嚴格遵守）：
1. 為了能讓程式系統自動提取代碼進行後續處理，請「務必」提供真實且正確的 4 位數字台股代號（例如 `[2330]`、`[2061]`、`[3231]` 等），請以方括號包圍。請確保代碼與公司名稱精確對應。
2. 請「絕對不要」遮蔽、替代或隱藏股票代號（例如：不要寫成 [23XX]、[3XXX]、[XXXX] 或是 [xxxx]）。
3. 區分「建材營造業」與「金融業」的一次性入帳或季節性變動，與「製造業/電子業/生技業」等實體產品銷售的「結構性爆發」。

報告內容應包括：
1. **異軍突起與籌碼/技術共振大解析**：簡述本月整體市場營收爆發的宏觀產業背景，並說明當「營收結構性暴增」伴隨「主力分點悄悄囤貨（籌碼集中）」且技術面上「短期/中期/長期均線高度糾結整理收斂後，股價帶量突破/20MA上揚起漲」時（即底部成形、長黑/盤整後帶量突破的完美圖形），對於後市噴發幅度的加乘效應。
2. **營收爆發股核心原因與籌碼技術剖析表**：以 Markdown 表格列出，包含：
   - 股票代號與名稱（真實 4 位數字代號，如 `[3535] 晶彩科`）
   - 產業別
   - 本月營收與年增率 (YoY)
   - 營收爆發核心具體原因摘要
   - 籌碼與技術面現況（有無特定主力分點囤貨、均線是否糾結/呈起漲噴發型態）
   - 成長持續性與圖形爆發評級（高/中/低）
3. **重點突破個股逐一深度解析**：挑選 3-5 檔非單純建案一次性認列、最符合「營收爆發+籌碼集中+均線糾結起漲」圖形的個股進行深度點評：
   - 分析其產品線爆發與訂單核心動能。
   - **籌碼面分析**：指出是否有特定主力券商分點（如：台灣摩根士丹利、元大松山、凱基台北、特定關鍵大戶分點）在過去數週持續買進，且籌碼集中度顯著上升。
   - **技術面圖形分析**：分析其日K線圖形，是否正處於均線（5MA/10MA/20MA/60MA）極度糾結盤整、20MA 正式向上翻揚、或突破下降軌道/箱型整理的關鍵買點圖形。
4. **風險與操作建議**：警告投資人一次性授權金認列、營收假突破、以及主力短線分點假鎖碼出貨的風險。

請以繁體中文撰寫，內容要具備高度專業度，使用 Markdown 格式呈現，多使用標題與加粗字體。
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if not report_content or not report_content.strip():
            raise ValueError("API returned empty content")
        save_gemini_report('surging_stocks_analysis', current_date_str[:7], report_content, db_path=db_path)
        return report_content
    except Exception as e:
        error_msg = str(e)
        print(f"Search grounding failed for surging stocks: {error_msg}. Falling back...")
        try:
            model_no_search = get_vertex_model(api_key, enable_search=False)
            if not model_no_search:
                return f"Gemini 產生異軍突起分析報告失敗: {e}"
            fallback_prompt = prompt + "\n\n⚠️ 提示：由於聯網搜尋工具目前不可用，請直接根據上述提供的數據進行產業特性分析與爆發原因推理。"
            response_fallback = model_no_search.generate_content(fallback_prompt)
            fallback_content = response_fallback.text
            if fallback_content and fallback_content.strip():
                save_gemini_report('surging_stocks_analysis', current_date_str[:7], fallback_content, db_path=db_path)
                notice = f"⚠️ **提示：API 聯網搜尋失敗（詳細原因：`{error_msg}`），已自動退回使用「本地資料庫數據」進行分析。**\n\n"
                return notice + fallback_content
            else:
                return f"Gemini 產生報告失敗: {e}，且備用本地分析亦無回應。"
        except Exception as fallback_err:
            return f"Gemini 產生報告失敗: {e}，且備用本地分析發生錯誤: {fallback_err}"


def check_surging_technical_batch(stock_codes):
    """
    批次計算多個個股的技術指標，包含：
    1. 實時股價
    2. 短期均線糾結度 (5MA, 10MA, 20MA)
    3. 全期均線糾結度 (5MA, 10MA, 20MA, 60MA)
    4. 20MA 趨勢 (是否向上)
    5. 52週高點與接近程度
    回傳: {stock_code: (success, price, spread_short, spread_all, is_20ma_rising, high_52w, proximity_52w)}
    """
    import yfinance as yf
    import pandas as pd
    res = {}
    if not stock_codes:
        return res
    tickers = []
    for c in stock_codes:
        tickers.append(f"{c}.TW")
        tickers.append(f"{c}.TWO")
    try:
        # 下載過去 1 年 (52週) 的數據來計算 52W 新高與 60MA
        df = yf.download(tickers, period="1y", progress=False, timeout=12)
        if df.empty:
            return res
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.levels[0]:
                close_df = df['Close']
            elif 'Close' in df.columns.levels[1]:
                close_df = df.xs('Close', axis=1, level=1)
            else:
                close_df = pd.DataFrame()
                
            if 'High' in df.columns.levels[0]:
                high_df = df['High']
            elif 'High' in df.columns.levels[1]:
                high_df = df.xs('High', axis=1, level=1)
            else:
                high_df = pd.DataFrame()
        else:
            close_df = df[['Close']] if 'Close' in df.columns else pd.DataFrame()
            high_df = df[['High']] if 'High' in df.columns else pd.DataFrame()
            
        for code in stock_codes:
            success = False
            price = 0.0
            spread_short = 999.0
            spread_all = 999.0
            is_20ma_rising = False
            high_52w = 0.0
            proximity_52w = 0.0
            
            for t_suffix in [".TW", ".TWO"]:
                ticker = f"{code}{t_suffix}"
                if ticker in close_df.columns:
                    series = close_df[ticker].dropna()
                    high_series = high_df[ticker].dropna() if ticker in high_df.columns else series
                    if len(series) >= 60:
                        try:
                            ma5 = float(series.rolling(5).mean().iloc[-1])
                            ma10 = float(series.rolling(10).mean().iloc[-1])
                            ma20 = float(series.rolling(20).mean().iloc[-1])
                            ma60 = float(series.rolling(60).mean().iloc[-1])
                            curr_price = float(series.iloc[-1])
                            
                            # 52週高點 (以過去 250 天最大值計算)
                            h52w = float(high_series.max())
                            
                            # 糾結度
                            mas_short = [ma5, ma10, ma20]
                            spread_s = ((max(mas_short) - min(mas_short)) / curr_price) * 100
                            
                            mas_all = [ma5, ma10, ma20, ma60]
                            spread_a = ((max(mas_all) - min(mas_all)) / curr_price) * 100
                            
                            # 20MA 趨勢
                            is_20ma_rising_val = False
                            if len(series) >= 25:
                                ma20_prev = float(series.rolling(20).mean().iloc[-5])
                                ma20_curr = float(series.rolling(20).mean().iloc[-1])
                                is_20ma_rising_val = ma20_curr > ma20_prev
                            
                            price = round(curr_price, 2)
                            spread_short = round(spread_s, 2)
                            spread_all = round(spread_a, 2)
                            is_20ma_rising = is_20ma_rising_val
                            high_52w = round(h52w, 2)
                            proximity_52w = round((curr_price / h52w) * 100, 2) if h52w > 0 else 0.0
                            success = True
                            break
                        except Exception as e:
                            print(f"Error calculating surging tech for {code}: {e}")
            res[code] = (success, price, spread_short, spread_all, is_20ma_rising, high_52w, proximity_52w)
    except Exception as e:
        print(f"Error checking batch surging technicals: {e}")
        for code in stock_codes:
            res[code] = (False, 0.0, 999.0, 999.0, False, 0.0, 0.0)
    return res


def get_single_stock_chip_analysis(api_key, stock_code, stock_name, db_path=None):
    """
    針對特定個股進行即時「特定分點買超、主力囤貨、籌碼集中度」的 AI 聯網分析。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定或初始化失敗，無法進行個股籌碼分析。"
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
你是一位專業的台股籌碼分析專家。
當前系統時間是：{current_date_str}。

請利用你的 Google 搜尋引擎聯網工具，搜尋近期一個月內關於台股個股 {stock_code} {stock_name} 的特定券商分點買賣超動態、主力進出、大戶鎖碼、以及籌碼集中度變化的財經新聞與討論。

請針對 {stock_code} {stock_name} 提供一份**【個股即時籌碼面與分點動態解析】**：
1. **主力與特定分點進出**：指出過去數週是否有特定證券商分點（例如：外資、投信的特定分點，或是主力大戶如凱基台北、美商高盛、台灣摩根士丹利、元大松山等）在持續囤貨、大量買超或賣出。如果搜尋到具體買超分點，請明確寫出。
2. **籌碼集中度現況**：說明該股目前的籌碼結構，是趨於集中還是分散？大戶持股比例是否增加？
3. **題材與主力囤貨動機**：主力在吸貨/收購背後的近期利多題材或潛在驅動因素（如新單、技術突破、法說會展望良好等）。
4. **綜合評分與操作建議**：給予該股「籌碼集中度評級」（高/中/低），並從籌碼面角度給予操作上的具體防守點或買賣操作建議。

請以繁體中文撰寫，內容要具備高度專業度，使用 Markdown 格式呈現，多使用標題與加粗字體。
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if not report_content or not report_content.strip():
            raise ValueError("API returned empty content")
        return report_content
    except Exception as e:
        error_msg = str(e)
        print(f"Search grounding failed for single stock chip analysis: {error_msg}. Falling back...")
        try:
            model_no_search = get_vertex_model(api_key, enable_search=False)
            if not model_no_search:
                return f"Gemini 產生個股籌碼分析報告失敗: {e}"
            fallback_prompt = prompt + "\n\n⚠️ 提示：由於聯網搜尋工具目前不可用，請根據您對該個股近期籌碼趨向、分點行為及市場題材的歷史知識進行深度解析。"
            response_fallback = model_no_search.generate_content(fallback_prompt)
            fallback_content = response_fallback.text
            if fallback_content and fallback_content.strip():
                return f"⚠️ **提示：API 聯網搜尋失敗（詳細原因：`{error_msg}`），已自動退回使用「本地 AI 知識庫」分析。**\n\n" + fallback_content
            else:
                return f"Gemini 產生報告失敗: {e}，且備用本地分析亦無回應。"
        except Exception as fallback_err:
            return f"Gemini 產生報告失敗: {e}，且備用本地分析發生錯誤: {fallback_err}"


def scan_broker_ratings(api_key, db_path=None):
    """
    使用 Gemini 聯網搜尋近期（今日或最近幾天）台灣股市中，外資、投信等研究機構調整個股評等與目標價的資訊。
    回傳解析出的評等調整，並寫入 rating_adjustments 資料表中。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定或初始化失敗，無法進行評等掃描。"
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
你是一位專業的台股研究分析助理。當前系統時間是 {current_date_str}。

請利用你的 Google 搜尋引擎聯網工具，搜尋近期（特別是接近 {current_date_str} 的這幾天）台灣股市中，各大券商、外資（如大摩、小摩、野村、瑞銀、麥格理、美林等）或本土投信/研究機構「調整個股評等（升評/降評/維持）」與「目標價變更」的相關新聞、研究報告精要與財經網站資料。

請將搜尋到的評等調整資訊彙整，並「嚴格且只以 JSON 陣列格式」輸出。每個物件代表一次評等調整，欄位如下：
- `date`: 調整日期，格式必須為 "YYYY-MM-DD"（請根據報導推估具體日期，必須在此期間內）。
- `stock_code`: 4位數字股票代碼，例如 "2330"。如果無法確定，請略過該筆。
- `stock_name`: 公司名稱，例如 "台積電"。
- `broker`: 券商或研究機構名稱，例如 "摩根士丹利"、"瑞銀證券"、"統一投顧" 等。
- `original_rating`: 調整前的評等（如：中立、減碼、無、或買進）。如果未知請填 "未知"。
- `new_rating`: 調整後的評等（如：買進、優於大盤、中立、減碼等）。
- `target_price`: 調整後的目標價（數值，例如 1200.0。若沒有具體數值請填 0）。
- `current_pe`: 現行本益比（數值，例如 24.5。如果報導中沒有提及，請根據最新股價與預估/過往EPS進行合理估計，若真的無法估計則填 0）。
- `adjusted_pe`: 調整後目標價對應的本益比（數值，即目標價除以預估EPS，例如 30.2。如果報導沒提及，請以目標價合理估計，若真的無法估算則填 0）。
- `reason`: 調整評等或目標價的具體理由，請簡短摘錄核心要點（例如：新產能開出、CoWoS訂單爆發、AI伺服器拉貨超預期、或是庫存去化慢等），字數在 50-150 字內，以繁體中文撰寫。

請注意：
1. 僅回傳一個 JSON Array，不要有任何 Markdown 包裹字元（不要 ```json），也不要任何前後贅字。如果真的沒有找到任何評等調整，請回傳空陣列 `[]`。
2. 確保股票代碼是真實且存在的台股代碼。

回傳格式範例：
[
  {{
    "date": "2026-06-25",
    "stock_code": "3535",
    "stock_name": "晶彩科",
    "broker": "摩根士丹利",
    "original_rating": "中立",
    "new_rating": "優於大盤",
    "target_price": 32.5,
    "current_pe": 18.2,
    "adjusted_pe": 23.4,
    "reason": "受惠於新一代面板檢測設備出貨增加，且特定大客戶拉貨動能強勁。預期本季營收YoY將有結構性暴增，因此將評等調升至優於大盤，目標價升至 32.5 元。"
  }}
]
"""
    try:
        response = model_with_search.generate_content(prompt)
        content = response.text.strip()
        
        # 移除可能存在的 Markdown 標記
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        import json
        try:
            records = json.loads(content)
        except Exception as json_err:
            import re
            match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
            if match:
                records = json.loads(match.group(0))
            else:
                raise json_err
                
        if not isinstance(records, list):
            return "Gemini 回傳格式非陣列，無法解析。"
            
        conn = get_connection(db_path)
        cursor = conn.cursor()
        # 確保資料表一定存在，避免部署/快取不同步導致查詢出錯
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
        
        added_count = 0
        for r in records:
            code = r.get('stock_code')
            name = r.get('stock_name')
            broker = r.get('broker')
            if not code or not name or not broker:
                continue
                
            cursor.execute(
                "SELECT id FROM rating_adjustments WHERE date = ? AND stock_code = ? AND broker = ?",
                (r.get('date'), code, broker)
            )
            if cursor.fetchone():
                continue
                
            created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
                INSERT INTO rating_adjustments (
                    date, stock_code, stock_name, broker, original_rating, 
                    new_rating, target_price, reason, current_pe, adjusted_pe, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                r.get('date'), code, name, broker, r.get('original_rating', '未知'),
                r.get('new_rating', '未知'), float(r.get('target_price', 0.0) or 0.0),
                r.get('reason', ''), float(r.get('current_pe', 0.0) or 0.0),
                float(r.get('adjusted_pe', 0.0) or 0.0), created_time
            ))
            added_count += 1
            
        conn.commit()
        conn.close()
        return f"✔ AI 聯網掃描完成！本次共新增了 {added_count} 筆評等調整記錄。"
    except Exception as e:
        return f"Gemini 評等掃描失敗: {e}"

