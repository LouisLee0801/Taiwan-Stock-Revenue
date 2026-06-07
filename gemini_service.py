import os
import json
from datetime import datetime
import pandas as pd
import google.generativeai as genai
from google.generativeai import protos
from database import (
    get_connection, save_gemini_industry, save_gemini_report, get_gemini_report
)

def get_gemini_model(api_key=None, model_name="gemini-3.5-flash", enable_search=False):
    """初始化並傳回適合的 Gemini 模式，若型態已棄用則動態選取最新可用型態"""
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return None
        
    # 主動阻斷已棄用的 gemini-1.5-flash 與 gemini-2.0-flash，升級為最新的 3.5-flash
    if model_name in ["gemini-1.5-flash", "gemini-2.0-flash"]:
        model_name = "gemini-3.5-flash"
        
    try:
        genai.configure(api_key=api_key)
        
        # 進行動態模型選取
        selected_model_name = model_name
        try:
            available_models = [m.name for m in genai.list_models()]
            candidates = [
                'models/gemini-3.5-flash',
                'models/gemini-3.1-flash-lite',
                'models/gemini-2.5-flash'
            ]
            # 如果預設傳入的 model_name 不在可用清單中，從候選清單中選取一個可用的
            if f"models/{model_name}" not in available_models and model_name not in available_models:
                for candidate in candidates:
                    if candidate in available_models:
                        selected_model_name = candidate.replace('models/', '')
                        break
        except Exception as list_err:
            print(f"Could not list models: {list_err}. Falling back to default model_name.")
            # 如果列表失敗，且預設是已退役的 1.5-flash / 2.0-flash，則直接升級為 3.5-flash
            if selected_model_name in ["gemini-1.5-flash", "gemini-2.0-flash"]:
                selected_model_name = "gemini-3.5-flash"
        
        # 如果需要啟用 Google Search Grounding
        if enable_search:
            return genai.GenerativeModel(
                model_name=selected_model_name,
                tools=[protos.Tool(google_search=protos.Tool.GoogleSearch())]
            )
        else:
            return genai.GenerativeModel(selected_model_name)
    except Exception as e:
        print(f"Error configuring Gemini client: {e}")
        return None

def get_vertex_model(project_id=None, model_name="gemini-3.5-flash", enable_search=False):
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
    
    # 找出當月營收 YoY 成長最強的權值股
    large_caps = df[df['revenue'] >= 5000000].sort_values(by='yoy', ascending=False).head(10)
    
    # 格式化統計數據文字
    top_ind_text = []
    for ind, row in top_industries.iterrows():
        top_ind_text.append(f"- {ind}: YoY中位數 {row['median_yoy']:.1f}%, MoM中位數 {row['median_mom']:.1f} % (共{int(row['count'])}檔)")
    
    bottom_ind_text = []
    for ind, row in bottom_industries.iterrows():
        bottom_ind_text.append(f"- {ind}: YoY中位數 {row['median_yoy']:.1f}%, MoM中位數 {row['median_mom']:.1f} % (共{int(row['count'])}檔)")
        
    large_caps_text = []
    for _, row in large_caps.iterrows():
        large_caps_text.append(f"- {row['stock_code']} {row['stock_name']} ({row['original_industry']}): 當月營收 {row['revenue']/1000000:.1f}十億, YoY: {row['yoy']:.1f}%, MoM: {row['mom']:.1f}%")
        
    top_ind_str = "\n".join(top_ind_text)
    bottom_ind_str = "\n".join(bottom_ind_text)
    large_caps_str = "\n".join(large_caps_text)
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
幕僚策略分析師與基本面專家。
當前系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。
請針對 **{date_month}** 月份台股全體上市櫃公司的營業收入彙總結果，撰寫一份專業的**台股月度營收大盤趨勢與展望報告**。

【大盤整體營收數據】
- 全市場總營收: {total_revenue/1000000:.1f} 十億元新台幣
- 全市場營收 YoY: {market_yoy:.2f}%
- 全市場營收 MoM: {market_mom:.2f}%

【當月最亮眼產業（YoY 中位數前 5 名）】
{top_ind_str}

【當月最疲弱產業（YoY 中位數後 5 名）】
{bottom_ind_str}

【高營收大型權值股表現優異者】
{large_caps_str}

請撰寫報告，結構如下：
1. **大盤月度營收評論**：點評整體上市櫃營收的 YoY 與 MoM 成長狀況，說明當前台灣出口與製造業的景氣位階（擴張、復甦、或衰退放緩）。
2. **產業強弱勢解析與類股輪動**：詳細解析為何上述最強勢的產業能維持高成長，以及最疲弱的產業面臨何種瓶頸（如庫存去化、需求不振）。
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

def get_stock_details_from_gemini(api_key, stock_code, stock_name, db_path=None):
    """
    使用 Gemini (啟用 Google Search Grounding) 重新查詢個股的詳細資訊。
    包括：個股介紹、最近題材、小作文、法說會資訊、新聞，以及 Forward PE 估值分析。
    """
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
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if report_content and report_content.strip():
            return report_content
    except Exception as e:
        print(f"Search grounding failed for stock details: {e}")
        
    # 如果聯網搜尋失敗或空內容，自動退回到無聯網的標準 Gemini 模式 (以本地資料庫數據進行分析)
    try:
        model_no_search = get_vertex_model(api_key, enable_search=False)
        if not model_no_search:
            return "Gemini API 金鑰未設定或初始化失敗，無法查詢個股詳細資訊。"
            
        # 從資料庫中讀取該個股的本地數據作為 Context 傳給 Gemini，使其能寫出基本面分析
        conn = get_connection(db_path)
        df_rev = pd.read_sql('SELECT date_month, revenue, yoy, mom FROM monthly_revenue WHERE stock_code = ? ORDER BY date_month DESC LIMIT 6', conn, params=(stock_code,))
        df_fin = pd.read_sql('SELECT year, quarter, gross_margin, net_margin, eps FROM quarterly_financials WHERE stock_code = ? ORDER BY year DESC, quarter DESC LIMIT 4', conn, params=(stock_code,))
        df_pe = pd.read_sql('SELECT pe, pb, dy FROM daily_pe WHERE stock_code = ? ORDER BY date DESC LIMIT 1', conn, params=(stock_code,))
        conn.close()
        
        local_context = f"""
【本地資料庫有關 {stock_code} {stock_name} 的財務數據】
1. 最近 6 個月營收：
{df_rev.to_string(index=False)}

2. 最近 4 季獲利與利潤率：
{df_fin.to_string(index=False)}

3. 最新估值指標 (PE/PB/DY)：
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
2. **財務數據分析**：分析最近 6 個月營收年增與月增趨勢，以及最近 4 季 EPS、毛利率與淨利率走勢（說明是轉好、惡化還是持平）。
3. **估值評估 (PE/PB/DY)**：根據其目前的本益比 PE、淨值比 PB 與殖利率 DY，評估其目前的估值位階（偏高、合理或偏低）。
4. **結論與操作建議**：給予客觀的操作與基本面佈局建議。

請以繁體中文撰寫，以 Markdown 格式呈現。
並在報告開頭加上友善提示：`⚠️ 提示：由於 API 金鑰不支援聯網搜尋工具，本報告已自動退回使用「本地資料庫數據」進行基本面分析。`
"""
        response_fallback = model_no_search.generate_content(fallback_prompt)
        fallback_content = response_fallback.text
        if fallback_content and fallback_content.strip():
            return fallback_content
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
            return "Gemini 產生法說會分析報告失敗: API 傳回空內容。這可能是因為安全過濾器阻擋、搜尋失敗，或金鑰餘額不足。請確認 API 金鑰狀態。"
        save_gemini_report('investor_conferences', current_date_str[:7], report_content, db_path=db_path)
        return report_content
    except Exception as e:
        return f"Gemini 產生法說會分析報告失敗: {e}"

def analyze_turnaround_stocks(api_key, db_path=None):
    """
    找出資料庫中最近一季虧損（EPS <= 0.1），但最新月營收 YoY 顯著成長的潛力標的，
    調用 Gemini (啟用 Google Search Grounding) 聯網分析並撰寫「潛力轉盈股大解析報告」，
    告訴用戶為什麼這些個股有機會轉虧為盈。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定或初始化失敗，無法進行轉盈股分析。"
        
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 查詢候選股
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
        WHERE m.rn = 1 AND q.eps <= 0.1 AND m.yoy > 15
        ORDER BY m.yoy DESC
        LIMIT 10
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "目前資料庫中沒有符合「最近一季虧損且最新月份營收 YoY 成長」的潛力轉盈個股。"
        
    candidates = []
    for r in rows:
        candidates.append(
            f"- {r['stock_code']} {r['stock_name']}: 最新季報 EPS: {r['latest_eps']} 元, 最新月營收 YoY: {r['latest_yoy']:.1f}%"
        )
    candidates_str = "\n".join(candidates)
    
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
你是一位專業的台股投資顧問與基本面策略分析師。
當前系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。

我們從資料庫中篩選出了最新一季仍處於微利或虧損狀態（EPS <= 0.1），但最新月度營收 YoY 呈顯著爆發成長的「潛力轉虧為盈個股」名單：
{candidates_str}

請利用你的 Google 搜尋引擎聯網工具，深入查詢這幾檔個股的最新業務現況、訂單能見度、以及營收暴增原因，並撰寫一份專業的**台股潛力轉虧為盈個股深度解析報告**。

報告內容應包括：
1. **整體轉盈趨勢評估**：簡述營收領先獲利反映的商業邏輯，並說明這批公司目前所處的產業轉折點。
2. **個股逐一深度剖析**：針對上述個股，挑選其中最具代表性與實質基本面支撐的 5-6 檔進行深入檢索：
   - 說明其核心業務與近期營收暴增的具體原因（如：新產品認證通過、取得特定大廠訂單、缺料緩解、產品結構優化等）。
   - 分析其最近一次公開法說會的重點與管理層對轉盈時程的展望。
   - **法說會時間**：請明確寫出最近一次法說會發生的具體時間（年月或日期）。
   - **估值點評**：利用搜尋檢索其當前的預估本益比 (Forward PE)、PB 淨值比，評估目前股價是否已過度反映轉盈預期。
3. **風險提示**：列出投資這類轉盈股的常見陷阱（如：營收認列不具持續性、一次性處分利益、本業仍疲弱等）。
4. **操作策略與結論**：如何透過分批佈局或確認季報利潤率轉正來進行安全操作。

請以繁體中文撰寫，內容要具備高度專業度，使用 Markdown 格式呈現，多使用標題與加粗字體。
注意：請以當前時間視角來分析，避免提及陳舊分析，專注於當前的實際狀況。
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if not report_content or not report_content.strip():
            return "Gemini 產生轉盈股分析報告失敗: API 傳回空內容。這可能是因為安全過濾器阻擋、搜尋失敗，或金鑰餘額不足。請確認 API 金鑰狀態。"
        # 快取報告
        save_gemini_report('turnaround_list', current_date_str[:7], report_content, db_path=db_path)
        return report_content
    except Exception as e:
        return f"Gemini 產生轉盈股分析報告失敗: {e}"
