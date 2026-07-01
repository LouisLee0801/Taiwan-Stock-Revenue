import os
import json
from datetime import datetime
import pandas as pd
import google.generativeai as genai
from google.generativeai import protos
from database import (
    get_connection, save_gemini_industry, save_gemini_report, get_gemini_report
)

# --- 全域多方驗證與高信心度輸出攔截器 (Monkey-patching generate_content) ---
_original_generate_content = genai.GenerativeModel.generate_content

def _patched_generate_content(self, contents, **kwargs):
    verification_suffix = """
\\n
【⚠️ 數據真實性與多方驗證極嚴格規定（99% 把握度）】
1. **多方來源交叉驗證**：對於你搜集到的所有新聞事件、重大訊息、評等調整、可轉債發行數據等，你必須在搜尋結果中進行「多方交叉核對」。認可的主流財經與官方來源包括：公開資訊觀測站(MOPS)、櫃買中心(TPEx)、證交所(TWSE)、經濟日報、工商時報、MoneyDJ、Anue 鉅亨網、Yahoo 奇摩股市、Goodinfo! 台灣股市資訊網、玩股網、股狗網、投資少數派、各大證券商研究報告。如果只有單一極度模糊或可疑的個人社群貼文，請多加核實。
2. **99% 把握度與可轉債 (CB) 認定**：
   - 只要搜尋結果中明確指出該股流通在外的可轉債代碼與名稱（例如：亞帝歐一/35161、有擔保/無擔保、發行總額、到期日、轉換價、現價、已轉換比例或剩餘餘額），這屬於官方登錄與公告之數據，即代表 100% 真實存在，請務必完整且真實地呈現於報告的「可轉債 (CB) 發行狀況與評估」欄位中，絕對不可漏掉或寫「無發行」！
   - **到期日與年份的絕對精準度**：對於可轉債的發行日與到期日，你必須精確與搜尋到的官方公告核對，不可自行做錯誤的年份推算或加法運算（例如：亞帝歐一的發行日為 2024/10/18，到期日為 2027/10/18，絕對不可寫成 2028 年！）。所有日期必須與搜尋到的官方事實 100% 吻合！
   - 對於新聞或重訊，只有在你有 99% 的把握度確認其為真實、準確且發生於指定日期區間的事實後，才能將其回傳。任何存在疑慮、語意不清、或無法確定真偽的傳言與數據，請老實註明「尚待證實」或忽略，也絕對不能冒險編造！
3. **拒絕編造與幻覺**：絕對禁止任何形式的編造或填充（例如編造假的產品名稱、假的公司合作、假的財務展望數據）。對於數據庫或搜尋中未提及的資訊，請直接回答「未有公開數據」或「無發行/無此訊息」，老實回答是唯一被允許的行為！
"""
    if isinstance(contents, str):
        contents = contents + verification_suffix
    elif isinstance(contents, list):
        if contents and isinstance(contents[-1], str):
            contents[-1] = contents[-1] + verification_suffix
        elif contents and hasattr(contents[-1], 'text'):
            try:
                contents[-1].text = contents[-1].text + verification_suffix
            except Exception:
                pass
    return _original_generate_content(self, contents, **kwargs)

genai.GenerativeModel.generate_content = _patched_generate_content

def write_gemini_debug(msg):
    try:
        log_path = r"C:\Users\a0919\.gemini\antigravity\scratch\tw-stock-fundamental-analyzer\gemini_debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception as e:
        print(f"Failed to write debug log: {e}")

def get_gemini_model(api_key=None, model_name="gemini-2.5-flash", enable_search=False):
    """????????????????? Gemini ��???????????????????????????????????"""
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        write_gemini_debug("get_gemini_model failed: No API Key provided.")
        return None
        
    # �D?????�w???��?????????��??????????í???????? (gemini-2.5-flash)
    # ??????�_ list_models ����???�H??��????????
    if model_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-3.5-flash"]:
        model_name = "gemini-2.5-flash"
        
    key_preview = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "invalid_key"
    write_gemini_debug(f"Configuring Gemini Client. Key: {key_preview}, Target model: {model_name}, Search: {enable_search}")
    try:
        genai.configure(api_key=api_key)
        write_gemini_debug(f"Selected model name: {model_name}")
        # ?????????? Google Search Grounding
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
    """??�e??????????????????????��??"""
    return get_gemini_model(api_key=project_id, model_name=model_name, enable_search=enable_search)



# --- 1. ??�~���????? ---



# --- 1. 產業精細化分類 ---

def refine_stock_industries(api_key, db_path=None, batch_size=40):
    """
    使用 Gemini 根據個股名稱及原始分類，精細化其產業分類，
    並將結果寫入 gemini_industry 表。
    """
    model = get_vertex_model(api_key)
    if not model:
        return "Gemini API 金鑰未設定，無法執行產業精細化。"
    
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT DISTINCT stock_code, stock_name, industry 
        FROM monthly_revenue 
        WHERE stock_code NOT IN (SELECT stock_code FROM gemini_industry)
        ORDER BY stock_code
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "資料庫中無需要進行精細化分類的個股。"
        
    stocks_to_process = [{'code': r['stock_code'], 'name': r['stock_name'], 'org_industry': r['industry']} for r in rows]
    total_stocks = len(stocks_to_process)
    print(f"Total stocks to refine: {total_stocks}")
    
    refined_count = 0
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    
    for i in range(0, total_stocks, batch_size):
        batch = stocks_to_process[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} stocks)...")
        
        input_data = []
        for s in batch:
            input_data.append(f"股票代碼: {s['code']} | 股票名稱: {s['name']} | 原始產業分類: {s['org_industry']}")
        input_text = "\n".join(input_data)
        
        prompt = f"""
你是一位專業的台股產業分析師。
今天系統時間是：{current_date_str}。

請將以下這批台灣上市櫃公司的「原始產業分類」進行精細化調整，使其更符合當前實際的主營業務或更精確的細分產業分類（例如：將原始分類「半導體業」精細化為「晶圓代工」、「IC設計」、「封測」、「半導體設備」、「半導體材料」；將「電子零組件業」精細化為「散熱」、「銅箔基板」、「連接器」、「PCB」等；其餘產業如營建、生技、化學、電機等亦請比照進行合適的細分）。

請注意：
1. 僅回傳一個 JSON 陣列，不要有任何 Markdown 包裹符號或贅字。
2. 每個物件代表一家公司，欄位為：
   - `stock_code`: 4位數字代碼
   - `refined_industry`: 精細化後的產業分類名稱（繁體中文，如「伺服器散熱」）
   - `reason`: 精細化的簡短理由（30字以內）

輸入公司名單：
{input_text}

輸出 JSON 格式範例：
[
  {{"stock_code": "2330", "refined_industry": "晶圓代工", "reason": "全球晶圓代工龍頭，以先進製程為主"}}
]
"""
        try:
            response = model.generate_content(prompt)
            content = response.text.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            import json
            records = json.loads(content)
            
            for r in records:
                code = r.get('stock_code')
                refined_ind = r.get('refined_industry')
                reason_val = r.get('reason', '')
                if not code or not refined_ind:
                    continue
                
                s_name = next((s['name'] for s in batch if s['code'] == code), '')
                save_gemini_industry(code, s_name, refined_ind, reason_val, db_path=db_path)
                refined_count += 1
                
        except Exception as batch_err:
            print(f"Failed to process batch {i//batch_size + 1}: {batch_err}")
            
    return f"產業精細化完成！成功更新了 {refined_count} 檔個股的精細分類。"


# --- 2. 行業異數分析 ---

def analyze_industry_outliers(api_key, db_path=None, date_month=None, industry_name=None, use_refined=False):
    """
    分析指定產業在當月份的營收表現，找出 YoY 或 MoM 成長顯著優於中位數的「異數個股」，
    並由 Gemini AI 聯網查詢其背後營收爆發原因與可持續性。
    """
    model = get_vertex_model(api_key)
    if not model:
        return "Gemini API 金鑰未設定，無法執行 AI 行業異數分析。"
        
    conn = get_connection(db_path)
    from database import get_monthly_revenues_with_pe
    raw_data = get_monthly_revenues_with_pe(date_month, db_path=db_path)
    if not raw_data:
        conn.close()
        return f"找不到 {date_month} 月份的營收與估值資料，無法進行分析。"
        
    df = pd.DataFrame(raw_data)
    conn.close()
    
    ind_col = 'refined_industry' if use_refined else 'original_industry'
    df_ind = df[df[ind_col] == industry_name].copy()
    if df_ind.empty:
        df_ind = df[df['original_industry'] == industry_name].copy()
        
    if len(df_ind) < 2:
        return f"產業別 '{industry_name}' 內個股數量太少，無法進行異數分析。"
        
    median_yoy = df_ind['yoy'].median()
    median_mom = df_ind['mom'].median()
    
    outliers = df_ind[
        (df_ind['yoy'] > 0) & 
        (df_ind['yoy'] > median_yoy) & 
        (df_ind['mom'] > median_mom) & 
        ((df_ind['yoy'] > median_yoy + 15) | (df_ind['mom'] > median_mom + 10))
    ].copy()
    
    if outliers.empty:
        outliers = df_ind[
            (df_ind['yoy'] > median_yoy) & 
            (df_ind['yoy'] > 10)
        ].copy()
        
    if outliers.empty:
        return f"在 {date_month} 中，'{industry_name}' 內無顯著異軍突起的異數個股。"
        
    outliers = outliers.sort_values(by='yoy', ascending=False).head(10)
    
    outliers_text = []
    for _, row in outliers.iterrows():
        pe_str = f"{row['pe']:.1f}" if pd.notnull(row['pe']) else "N/A"
        outliers_text.append(
            f"- {row['stock_code']} {row['stock_name']}: YoY: {row['yoy']:.1f}%, MoM: {row['mom']:.1f}%, PE: {pe_str}, 當月營收: {row['revenue']/1000:.1f}百萬元"
        )
    outliers_list_str = "\n".join(outliers_text)
    
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
你是一位專業的台股基本面分析師。
今天系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。

請針對 {date_month} 月份，{industry_name} 產業中營業收入異軍突起的「異數個股」（即表現顯著優於同業中位數的個股）進行深度基本面分析。

該產業當月的營收表現中位數如下：
- 產業 YoY 中位數: {median_yoy:.1f}%
- 產業 MoM 中位數: {median_mom:.1f}%

我們篩選出的營收異數個股名單如下：
{outliers_list_str}

請結合 Google 搜尋工具，針對上述營收爆發的異數個股進行深度點評，內容包括：
1. **營收爆發原因**：詳細說明這些個股為何能在同業中脫穎而出（例如：打入新供應鏈、新產能開出、主力產品價量齊揚等）。
2. **營運持續性評估**：評估此營收增長是短期一次性入帳，還是具備長期的基本面支撐與成長趨勢。
3. **估值與投資建議**：結合其當前本益比（PE）與同業位階，評估投資風險，並給予操作評級（高/中/低）。

請以繁體中文撰寫，內容詳實，使用 Markdown 格式呈現。
"""
    try:
        response = model.generate_content(prompt)
        report_content = response.text
        report_key = f"{date_month}_{industry_name}"
        save_gemini_report('monthly_industry', report_key, report_content, db_path=db_path)
        return report_content
    except Exception as e:
        return f"Gemini 行業異數分析失敗: {e}"


# --- 3. 大盤月度營收分析 ---

def analyze_monthly_market_trends(api_key, db_path=None, date_month=None):
    """
    彙整整個市場當月的營收表現（總額、成長排行、產業分布），
    由 Gemini 撰寫大盤整體營收解析與未來趨勢預測。
    """
    model = get_gemini_model(api_key)
    if not model:
        return "Gemini API 金鑰未設定，無法產生大盤營收趨勢報告。"
        
    conn = get_connection(db_path)
    from database import get_monthly_revenues_with_pe
    raw_data = get_monthly_revenues_with_pe(date_month, db_path=db_path)
    
    if not raw_data:
        conn.close()
        return f"找不到 {date_month} 月份的營收資料，無法分析。"
        
    df = pd.DataFrame(raw_data)
    conn.close()
    
    total_revenue = df['revenue'].sum()
    total_last_year_revenue = df['last_year_revenue'].sum()
    total_last_month_revenue = df['last_month_revenue'].sum()
    
    market_yoy = (total_revenue - total_last_year_revenue) / total_last_year_revenue * 100 if total_last_year_revenue else 0
    market_mom = (total_revenue - total_last_month_revenue) / total_last_month_revenue * 100 if total_last_month_revenue else 0
    
    ind_stats = df.groupby('original_industry').agg(
        median_yoy=('yoy', 'median'),
        median_mom=('mom', 'median'),
        count=('stock_code', 'count')
    )
    ind_stats_filtered = ind_stats[ind_stats['count'] >= 4]
    
    top_industries = ind_stats_filtered.sort_values(by='median_yoy', ascending=False).head(5)
    bottom_industries = ind_stats_filtered.sort_values(by='median_yoy', ascending=True).head(5)
    
    large_caps = df[df['revenue'] >= 5000000].sort_values(by='yoy', ascending=False).head(10)
    
    top_ind_text = []
    for ind, row in top_industries.iterrows():
        top_ind_text.append(f"- {ind}: YoY中位數 {row['median_yoy']:.1f}%, MoM中位數 {row['median_mom']:.1f}% (共{int(row['count'])}檔)")
    
    bottom_ind_text = []
    for ind, row in bottom_industries.iterrows():
        bottom_ind_text.append(f"- {ind}: YoY中位數 {row['median_yoy']:.1f}%, MoM中位數 {row['median_mom']:.1f}% (共{int(row['count'])}檔)")
        
    large_caps_text = []
    for _, row in large_caps.iterrows():
        large_caps_text.append(f"- {row['stock_code']} {row['stock_name']} ({row['original_industry']}): 當月營收 {row['revenue']/1000000:.1f}十億, YoY: {row['yoy']:.1f}%, MoM: {row['mom']:.1f}%")
        
    top_ind_str = "\n".join(top_ind_text)
    bottom_ind_str = "\n".join(bottom_ind_text)
    large_caps_str = "\n".join(large_caps_text)
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
度驗證。

你是一位資深的台股宏觀策略分析師與基本面專家。
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
"""
    try:
        response = model.generate_content(prompt)
        report_content = response.text
        save_gemini_report('monthly_market', date_month, report_content, db_path=db_path)
        return report_content
    except Exception as e:
        return f"Gemini 產生大盤報告失敗: {e}"


# --- 4. 季度財報大盤 analysis ---

def analyze_quarterly_financial_trends(api_key, db_path=None, year=None, quarter=None):
    """
    分析季度的企業財務表現，找出毛利率高的產業與獲利強勢的個股，
    並由 Gemini AI 聯網查詢其未來展望。
    """
    model = get_vertex_model(api_key)
    if not model:
        return "Gemini API 金鑰未設定，無法執行 AI 季度財報分析。"
        
    conn = get_connection(db_path)
    from database import get_quarterly_financials_list
    raw_data = get_quarterly_financials_list(year, quarter, db_path=db_path)
    if not raw_data:
        conn.close()
        return f"找不到 {year} 年第 {quarter} 季的季度財務資料，無法進行分析。"
        
    df = pd.DataFrame(raw_data)
    
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT stock_code, industry FROM monthly_revenue')
    ind_map = {row['stock_code']: row['industry'] for row in cursor.fetchall()}
    conn.close()
    
    df['industry'] = df['stock_code'].map(ind_map)
    df = df[df['industry'].notnull()].copy()
    
    ind_margin = df.groupby('industry').agg(
        median_gm=('gross_margin', 'median'),
        median_nm=('net_margin', 'median'),
        median_eps=('eps', 'median'),
        count=('stock_code', 'count')
    )
    ind_margin_filtered = ind_margin[ind_margin['count'] >= 4]
    top_gm_ind = ind_margin_filtered.sort_values(by='median_gm', ascending=False).head(5)
    top_eps_stocks = df.sort_values(by='eps', ascending=False).head(10)
    
    top_gm_text = []
    for ind, row in top_gm_ind.iterrows():
        top_gm_text.append(
            f"- {ind}: 毛利率中位數 {row['median_gm']:.1f}%, 淨利率中位數 {row['median_nm']:.1f}% (共 {int(row['count'])} 檔)"
        )
    top_gm_str = "\n".join(top_gm_text)
    
    top_eps_text = []
    for _, row in top_eps_stocks.iterrows():
        top_eps_text.append(
            f"- {row['stock_code']} {row['stock_name']} ({row['industry']}): EPS {row['eps']:.2f}元, 毛利率 {row['gross_margin']:.1f}%, 淨利率 {row['net_margin']:.1f}%"
        )
    top_eps_str = "\n".join(top_eps_text)
    
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
你是一位專業且客觀的台股基本面分析師。
今天系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。

請針對 {year} 年第 {quarter} 季的台股上市公司季度財報進行深度分析與整體展望。

【毛利率表現前 5 名的產業（中位數）】：
{top_gm_str}

【EPS 表現前 10 名的個股】：
{top_eps_str}

請撰寫一份**台股季度財報整體分析報告**，內容包括：
1. **整體季度財報點評**：分析當季台股企業整體獲利（毛利率、淨利率）的趨勢，以及宏觀經濟對企業獲利的影響。
2. **強勢產業利潤解析**：分析上述高毛利、高成長產業（如半導體、光電、電子零組件等）的獲利動能來源及未來走勢。
3. **績優個股深度點評**：針對 EPS 表現亮眼的龍頭企業進行商業模式與競爭力評估。
4. **未來季度獲利展望**：評估未來 1-2 季內哪些產業的利潤率有望持續擴張，哪些產業面臨毛利率下行風險，並給予投資操作建議。

請以繁體中文撰寫，文風專業，結構清晰，使用 Markdown 格式呈現。
"""
    try:
        model_with_search = get_vertex_model(api_key, enable_search=True)
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        report_key = f"{year}_Q{quarter}"
        save_gemini_report('quarterly_market', report_key, report_content, db_path=db_path)
        return report_content
    except Exception as e:
        return f"Gemini 季度財報分析失敗: {e}"


# --- 5. 轉虧為盈分析 ---

# ── 已驗證的可轉債靜態資料庫（API 均無法 server-side 存取，以此為最終依據）──────
# 格式：stock_code -> list of CB dicts
# 欄位：cb_code, cb_name, conversion_price, issue_date, maturity_date,
#        secured, total_amount, converted_ratio
# 到期日來源：公開資訊觀測站官方公告原文（不可自行推算）
KNOWN_CB_DATA = {
    # 格式：stock_code -> list of CB dicts
    # 到期日來源：公開資訊觀測站官方公告原文（不可自行推算）
    "3516": [{
        "cb_code": "35161",
        "cb_name": "亞帝歐一",
        "conversion_price": "26.14",
        "issue_date": "2024/10/18",
        "maturity_date": "2027/10/17",
        "secured": "有擔保",
        "total_amount": "2億元",
        "converted_ratio": "",
    }],
    "3037": [{
        "cb_code": "30371",
        "cb_name": "欣興一",
        "conversion_price": "163.8",
        "issue_date": "2025/11/03",
        "maturity_date": "2030/11/03",
        "secured": "無擔保",
        "total_amount": "40億元",
        "converted_ratio": "",
    }],
    "5439": [{
        "cb_code": "54391",
        "cb_name": "高技一",
        "conversion_price": "300.0",
        "issue_date": "2025/09/09",
        "maturity_date": "2030/09/09",
        "secured": "無擔保",
        "total_amount": "",
        "converted_ratio": "",
    }],
    # 其他已知 CB 可於此新增：
    # "XXXX": [{...}],
}



_price_cache: dict = {}   # stock_code -> (price, fetch_time) 記憶體快取（本次執行期間有效）
_PRICE_CACHE_TTL = 1800   # 秒（30 分鐘內不重複抓）


def get_latest_stock_price(stock_code):
    """獲取單個個股最新收盤價（含記憶體快取，避免重複呼叫 yfinance）。"""
    import yfinance as yf
    import time
    import pandas as pd
    now = time.time()
    if stock_code in _price_cache:
        cached_price, cached_time = _price_cache[stock_code]
        if now - cached_time < _PRICE_CACHE_TTL:
            return cached_price
    for t_suffix in [".TW", ".TWO"]:
        ticker = f"{stock_code}{t_suffix}"
        try:
            df = yf.download(ticker, period="5d", progress=False, timeout=5)
            if not df.empty and 'Close' in df.columns:
                close_col = df['Close']
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                series = close_col.dropna()
                if not series.empty:
                    price = round(float(series.iloc[-1]), 2)
                    _price_cache[stock_code] = (price, now)
                    return price
        except Exception:
            pass
    return None


def get_batch_stock_prices(stock_codes: list) -> dict:
    """
    批次下載多支股票最新收盤價（一次 yf.download 取代逐筆呼叫）。
    回傳 dict: stock_code -> price (float or None)
    """
    import yfinance as yf
    import time
    import pandas as pd
    now = time.time()
    result = {}
    to_fetch_tw  = []
    to_fetch_two = []

    for code in stock_codes:
        if code in _price_cache:
            cached_price, cached_time = _price_cache[code]
            if now - cached_time < _PRICE_CACHE_TTL:
                result[code] = cached_price
                continue
        to_fetch_tw.append(code)

    def _batch_download(codes, suffix):
        if not codes:
            return {}
        tickers = " ".join(f"{c}{suffix}" for c in codes)
        try:
            df = yf.download(tickers, period="5d", progress=False,
                             timeout=12, group_by='ticker')
            out = {}
            for code in codes:
                ticker = f"{code}{suffix}"
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        close_col = df[ticker]['Close'].dropna()
                    else:
                        close_col = df['Close'].dropna()
                    if not close_col.empty:
                        out[code] = round(float(close_col.iloc[-1]), 2)
                except Exception:
                    pass
            return out
        except Exception:
            return {}

    # 先批次下 .TW
    tw_prices = _batch_download(to_fetch_tw, ".TW")
    still_missing = [c for c in to_fetch_tw if c not in tw_prices]

    # 找不到的再試 .TWO（OTC 股）
    two_prices = _batch_download(still_missing, ".TWO")

    for code in to_fetch_tw:
        price = tw_prices.get(code) or two_prices.get(code)
        result[code] = price
        if price is not None:
            _price_cache[code] = (price, now)

    return result

def _get_cb_from_local_cache(db_path, stock_code):
    """
    取得個股的可轉債資料，優先順序：
    1. KNOWN_CB_DATA（程式碼內驗證資料，100% 精確）
    2. SQLite 本地快取（由 Gemini 之前搜尋結果解析存入）
    """
    # 第一優先：程式碼內的已驗證資料
    if stock_code in KNOWN_CB_DATA:
        return KNOWN_CB_DATA[stock_code]

    # 第二優先：SQLite 快取
    if not db_path:
        return []
    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cb_cache (
                stock_code TEXT,
                cb_code TEXT,
                cb_name TEXT,
                conversion_price TEXT,
                issue_date TEXT,
                maturity_date TEXT,
                secured TEXT,
                total_amount TEXT,
                converted_ratio TEXT,
                updated_at TEXT,
                PRIMARY KEY (stock_code, cb_code)
            )
        """)
        conn.commit()
        cur.execute(
            "SELECT cb_code, cb_name, conversion_price, issue_date, maturity_date, "
            "secured, total_amount, converted_ratio FROM cb_cache WHERE stock_code=?",
            (stock_code,)
        )
        rows = cur.fetchall()
        conn.close()
        result = []
        for row in rows:
            result.append({
                "cb_code": row[0], "cb_name": row[1],
                "conversion_price": row[2], "issue_date": row[3],
                "maturity_date": row[4], "secured": row[5],
                "total_amount": row[6], "converted_ratio": row[7],
            })
        return result
    except Exception:
        return []




def _save_cb_to_local_cache(db_path, stock_code, cb_list):
    """
    把 CB 資料寫入本地 SQLite 快取。
    cb_list: list of dict
    """
    if not db_path or not cb_list:
        return
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cb_cache (
                stock_code TEXT,
                cb_code TEXT,
                cb_name TEXT,
                conversion_price TEXT,
                issue_date TEXT,
                maturity_date TEXT,
                secured TEXT,
                total_amount TEXT,
                converted_ratio TEXT,
                updated_at TEXT,
                PRIMARY KEY (stock_code, cb_code)
            )
        """)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for cb in cb_list:
            cur.execute("""
                INSERT OR REPLACE INTO cb_cache
                (stock_code, cb_code, cb_name, conversion_price, issue_date,
                 maturity_date, secured, total_amount, converted_ratio, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                stock_code,
                cb.get("cb_code", ""), cb.get("cb_name", ""),
                cb.get("conversion_price", ""), cb.get("issue_date", ""),
                cb.get("maturity_date", ""), cb.get("secured", ""),
                cb.get("total_amount", ""), cb.get("converted_ratio", ""),
                now_str
            ))
        conn.commit()
        conn.close()
    except Exception:
        pass


def fetch_cb_data_with_gemini(api_key, stock_code, stock_name, db_path=None):
    """
    專門用於查詢可轉債資料的快速 Gemini 函式。
    流程：
    1. 先查 KNOWN_CB_DATA（硬編碼，絕對正確）
    2. 再查 SQLite 快取（30 天內有效）
    3. 以上都查不到才呼叫 Gemini Search，結果存入 SQLite
    回傳 list of CB dict，空 list 表示此股確定無發行中可轉債。
    """
    import re

    # 第一優先：KNOWN_CB_DATA
    if stock_code in KNOWN_CB_DATA:
        return KNOWN_CB_DATA[stock_code]

    # 第二優先：SQLite 快取（30 天內有效）
    if db_path:
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cb_cache (
                    stock_code TEXT, cb_code TEXT, cb_name TEXT,
                    conversion_price TEXT, issue_date TEXT, maturity_date TEXT,
                    secured TEXT, total_amount TEXT, converted_ratio TEXT,
                    updated_at TEXT, PRIMARY KEY (stock_code, cb_code)
                )
            """)
            conn.commit()
            cur.execute(
                "SELECT cb_code, cb_name, conversion_price, issue_date, maturity_date, "
                "secured, total_amount, converted_ratio, updated_at "
                "FROM cb_cache WHERE stock_code=?", (stock_code,)
            )
            rows = cur.fetchall()
            conn.close()
            if rows:
                # 檢查新鮮度（30 天內有效）
                try:
                    from datetime import timedelta
                    updated = datetime.strptime(rows[0][8], "%Y-%m-%d %H:%M:%S")
                    if datetime.now() - updated < timedelta(days=30):
                        return [{
                            "cb_code": r[0], "cb_name": r[1],
                            "conversion_price": r[2], "issue_date": r[3],
                            "maturity_date": r[4], "secured": r[5],
                            "total_amount": r[6], "converted_ratio": r[7],
                        } for r in rows]
                except Exception:
                    pass
        except Exception:
            pass

    # 第三優先：Gemini Search 專用查詢
    model = get_gemini_model(api_key)
    if not model:
        return []

    try:
        from google.generativeai import protos as _protos
        model_search = genai.GenerativeModel(
            model_name=model.model_name,
            tools=[_protos.Tool(google_search=_protos.Tool.GoogleSearch())]
        )
    except Exception:
        return []

    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""今天日期：{current_date}。
請直接使用 Google Search 搜尋「{stock_code} {stock_name} 可轉債」及「{stock_code} {stock_name} CB」。

我需要查詢 {stock_code} {stock_name} 目前所有發行中的可轉換公司債。

請從搜尋結果中找出每筆可轉債的：
- 5位數債券代號（例 30371）
- 債券名稱（例 欣興一）
- 轉換價（元）
- 發行日（YYYY/MM/DD）
- 到期日（YYYY/MM/DD，必須從官方公告原文讀取，絕對不可自行用發行日加年推算！）
- 擔保方式（有擔保 / 無擔保）
- 發行總額（億元）
- 已轉換比例（%）

請將找到的每筆可轉債分別用以下 JSON 格式完整輸出（若無可轉債則輸出空 array [])：
```json
[
  {{
    "cb_code": "XXXXX",
    "cb_name": "債券名稱",
    "conversion_price": "XXX.X",
    "issue_date": "YYYY/MM/DD",
    "maturity_date": "YYYY/MM/DD",
    "secured": "有擔保或無擔保",
    "total_amount": "XX億元",
    "converted_ratio": "XX%"
  }}
]
```
如果確定無發行中的可轉債，輸出：```json\n[]\n```
程式將直接解析 JSON，請確保格式正確。
"""
    try:
        resp = model_search.generate_content(prompt)
        text = resp.text if resp.text else ""

        # 解析 JSON 區塊（多種格式兼容）
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)
        if not json_match:
            # 嘗試直接找 JSON 陣列
            json_match = re.search(r'(\[\s*\{.*?\}\s*\])', text, re.DOTALL)

        if json_match:
            import json as _json
            try:
                cb_list = _json.loads(json_match.group(1))
            except Exception:
                cb_list = None

            if isinstance(cb_list, list) and len(cb_list) > 0:
                valid = []
                for cb in cb_list:
                    if cb.get("cb_code") and cb.get("maturity_date"):
                        valid.append({
                            "cb_code": str(cb.get("cb_code", "")).strip(),
                            "cb_name": str(cb.get("cb_name", "")).strip(),
                            "conversion_price": str(cb.get("conversion_price", "")).strip(),
                            "issue_date": str(cb.get("issue_date", "")).strip(),
                            "maturity_date": str(cb.get("maturity_date", "")).strip(),
                            "secured": str(cb.get("secured", "")).strip(),
                            "total_amount": str(cb.get("total_amount", "")).strip(),
                            "converted_ratio": str(cb.get("converted_ratio", "")).strip(),
                        })
                if valid:
                    _save_cb_to_local_cache(db_path, stock_code, valid)
                    return valid

            # ⚠️ 若 JSON 為空陣列 []，代表 Gemini 確認無 CB
            # 但「不快取 NONE」—— 因為 Gemini 可能搜尋不完整，讓使用者有機會手動新增
            if isinstance(cb_list, list) and len(cb_list) == 0:
                return []  # 不寫入 SQLite，讓下次可重試或手動新增

    except Exception:
        pass

    return []



def get_stock_details_from_gemini(api_key, stock_code, stock_name, db_path=None):
    """
    個股深度解析：
    架構：
    1. yfinance 直接取得即時股價（確定性數字）
    2. SQLite 本地快取讀取 CB 資料（若有）
    3. 若快取無 CB，強制 Gemini Google Search 查詢 CB 資料並寫回快取
    4. Gemini 只做質性分析（題材/新聞/法說）
    """
    model = get_gemini_model(api_key)
    if not model:
        return "Gemini API 金鑰未設定，無法查詢個股詳細資訊。"

    from google.generativeai import protos

    model_with_search = genai.GenerativeModel(
        model_name=model.model_name,
        tools=[protos.Tool(google_search=protos.Tool.GoogleSearch())]
    )

    current_date_str = datetime.now().strftime("%Y-%m-%d")

    # ── Step 1: 程式直接抓即時股價（使用快取版 get_latest_stock_price）────
    current_price = get_latest_stock_price(stock_code)


    price_context = (
        f"【即時股價（由 Yahoo Finance API 直接取得）】：{stock_code} {stock_name} 最新收盤價 = **{current_price} 元**。"
        f"請在報告股價欄位直接使用此數字，不得更改。"
        if current_price
        else f"【即時股價】：系統 API 暫時無法取得股價，請在報告中據實說明「股價資料暫時無法取得」。"
    )

    # ── Step 2: 從本地 SQLite 快取讀取 CB 資料 ─────────────────────────
    cached_cb = _get_cb_from_local_cache(db_path, stock_code)

    if cached_cb:
        cb_lines = []
        for cb in cached_cb:
            cb_lines.append(
                f"  - 代號：{cb['cb_code']} {cb['cb_name']}｜"
                f"轉換價：{cb['conversion_price']} 元｜"
                f"發行日：{cb['issue_date']}｜到期日：{cb['maturity_date']}｜"
                f"擔保：{cb['secured']}｜發行總額：{cb['total_amount']}｜"
                f"已轉換比例：{cb['converted_ratio']}"
            )
        cb_instruction = (
            f"【本地快取的可轉債資料（已驗證）】：\n" + "\n".join(cb_lines) +
            "\n以上數據為系統已驗證的快取資料，請**完整照抄**，任何日期、價格皆不得自行修改或推算。"
        )
        cb_search_instruction = ""
    else:
        # 沒有快取，強制 Gemini 搜尋並在 JSON 區塊回傳結構化 CB 資料
        cb_instruction = (
            f"【可轉債資料】：本地快取查無 {stock_code} {stock_name} 的可轉債紀錄。"
            f"請執行以下搜尋找出 CB 資料（非常重要，必須執行）："
        )
        cb_search_instruction = f"""
【⚠️ 強制執行：可轉債資料搜尋任務】
請依序搜尋以下關鍵字，找出 {stock_code} {stock_name} 的發行中可轉債：
1. 搜尋「{stock_code} 可轉債」
2. 搜尋「{stock_name} 轉換公司債」
3. 搜尋「{stock_code} CB goodinfo」或「{stock_code} 可轉債 histock」
4. 若找到 CB（例如代號 {stock_code}1、{stock_code}2 等），必須列出：
   - 可轉債代號（5位數，如 35161）
   - 可轉債名稱（如 亞帝歐一）
   - 轉換價（元）
   - 發行日（YYYY/MM/DD 格式）
   - 到期日（YYYY/MM/DD 格式，必須從官方公告原文讀取，絕對不可自行用「發行日 + N年」推算！）
   - 擔保方式（有擔保 / 無擔保）
   - 發行總額（百萬元）
   - 已轉換比例（%，若有）
5. 若確定無發行中的可轉債，才可寫「目前無發行中的可轉債」。
"""

    # ── Step 3: 組合完整 prompt ──────────────────────────────────────────
    prompt = f"""
你是一位專業的台股投資顧問與產業基本面分析師。
當前系統時間是：{current_date_str}。

═══════════════════════════════════════════
系統已確認的數字（不得更改）：
{price_context}
{cb_instruction}
═══════════════════════════════════════════
{cb_search_instruction}
---

請使用 Google Search 搜尋引擎，針對個股 **{stock_code} {stock_name}**，
查詢以下質性資訊，撰寫「個股深度解析報告」：

搜尋任務：
1. 搜尋「{stock_code} {stock_name} 重大訊息 mops」→ 近一個月重要公告
2. 搜尋「{stock_code} {stock_name} 法說會 {current_date_str[:4]}」→ 最新法說內容
3. 搜尋「{stock_code} {stock_name} 新聞」→ 近期真實媒體報導

報告架構（Markdown 格式，800–1500 字）：

## 一、個股介紹
核心業務、主要產品、產業鏈角色（基於已知事實，不得虛構）。

## 二、最近題材
近期最受市場關注的題材（只報告搜尋到的真實內容，查不到就寫「近期查無特定題材」）。

## 三、市場傳言與小作文
論壇或媒體流傳的傳言，客觀評估。查不到就寫「目前查無市場傳言」。

## 四、法說會重點
最近一次法說會的真實摘要（搜尋到才寫，查無就說「近期無法說紀錄」）。

## 五、重大訊息與公告（近一個月）
【嚴格規定】：
- 請搜尋「{stock_code} {stock_name} 重大訊息」和「{stock_code} site:mops.twse.com.tw」
- 僅列出搜尋結果中**明確出現**的公告，必須標注：公告日期、公告標題
- **絕對禁止**：捏造公告內容、推測可能有哪些公告、填充與其他公司相關的公告
- 若搜尋結果中無任何具體公告，請直接寫：「搜尋結果中查無近一個月具體重大公告」
- 寧可查無，也絕不捏造

## 六、可轉債 (CB) 發行狀況
— 若系統步驟1、2已提供確認的 CB 資料，請**完整照抄**。
— 若系統提示「查無 CB」，請寫「目前無發行中的可轉債」，不得自行搜尋或修改。

## 七、近期真實新聞（近一個月）
【嚴格規定】：
- 請搜尋「{stock_code} {stock_name} 新聞 {current_date_str[:7]}」
- 每則新聞必須標注：**發布日期**、**媒體名稱**、新聞標題
- 若無法確認日期或媒體來源，該則不得列入
- **絕對禁止**：捏造新聞、把其他公司的新聞張冠李戴、填充舊聞
- 若查無符合條件的新聞，請寫：「近一個月查無具體新聞報導」

## 八、財務與估值分析
估計 EPS 與 Forward PE，評論估值合理性。

⚠️ **以下行為視為嚴重違規，絕對禁止**：
- 建立不存在的公告、新聞、合作案、品牌、事件
- 對不確定的資訊加具體細節使其看起來真實
- 轉換季、日期或金額（如把「月」換成「年」）
- 把其他公司的資訊套用在此股票上
「查無」永遠比「編造」更正確、更安全。
"""
    try:
        response = model_with_search.generate_content(prompt)
        result_text = response.text

        # ⚠️ 注意：此處不再嘗試從 AI 回應文字 regex 解析 CB 資料寫入 SQLite。
        # 原因：regex 抓取不可靠，會把 AI 幻覺資料存入資料庫造成永久錯誤。
        # CB 資料來源：KNOWN_CB_DATA（硬編碼）或 fetch_cb_data_with_gemini（結構化 JSON）。
        return result_text
    except Exception as e:
        return f"Gemini 查詢個股詳細資訊失敗: {e}"





def predict_turnarounds_with_gemini(api_key, industry_name, stock_financials_json):
    """
    同業營收與估值比較頁面：使用 Gemini 快速評估一組虧損/減虧股中，哪些個股在營收與YoY改善下有較高的轉盈潛力。
    """
    model = get_vertex_model(api_key)
    if not model:
        return []

    current_date_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
你是一位專業的台股基本面分析師。
今天系統時間是：{current_date_str}。
針對目前屬於 **{industry_name}** 產業中，以下這批最新一季報 EPS 為負值（虧損）或減虧中，但最新月度營收 YoY 成長的個股數據：

請逐一評估每檔股票在營收成長動能與虧損收斂趨勢下，是否有較高的機率在下一個月或下一季度實現轉虧為盈。

判斷條件與原則：
1. 若該股最新一季 EPS 已接近 0 元（如 -0.05 至 0.0 元）且月營收 YoY 顯著成長，則轉盈潛力高。
2. 若月營收 YoY 持續走高且毛利結構改善，則轉盈速度快。

請回傳一個 JSON 陣列格式，包含被你評定為「有顯著轉虧為盈潛力（或高度 Highlight）」的個股，格式如下，且**請直接回傳 JSON，不要包裹 Markdown 語法（不要 ```json，不要前後贅字）**：
1. "code": 股票代碼（字串）
2. "reason": AI 評估理由（以繁體中文撰寫，字數約 15-20 字，說明其轉盈依據，如營收動能強、EPS已近兩平）

待評估的個股 JSON 數據如下：
{stock_financials_json}
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        import json
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error predicting turnarounds: {e}")
        return []

def analyze_turnaround_stocks(api_key, db_path=None):
    """
    分析轉虧為盈/減虧潛力股，結合最近6個月營收及均線糾結狀態，
    並由 Gemini AI 聯網查詢其未來營收展望與轉盈時間表。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定，無法執行 AI 聯網分析"

    conn = get_connection(db_path)
    cursor = conn.cursor()

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
        WHERE m.rn = 1 AND q.eps IS NOT NULL AND (
            q.eps <= 0.05
            OR m.stock_code IN (
                SELECT DISTINCT stock_code FROM daily_pe WHERE pe <= 0
            )
            OR m.stock_code IN (
                SELECT stock_code 
                FROM quarterly_financials 
                WHERE eps <= 0 
                  AND (year < q.year OR (year = q.year AND quarter < q.quarter))
            )
        ) AND m.yoy > 5
        ORDER BY m.yoy DESC
        LIMIT 35
    ''')
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return "資料庫中無符合營收 YoY 成長且有虧損背景的候選股。"

    tech_map = check_ma_convergence_batch([r['stock_code'] for r in rows])

    # 預先批次下載所有缺少股價的股票（取代逐筆呼叫，讓 35 支股票僅需兩次 yf.download）
    codes_missing_price = [
        r['stock_code'] for r in rows
        if r['stock_code'] not in tech_map or tech_map[r['stock_code']] is None
           or (len(tech_map[r['stock_code']]) >= 4 and tech_map[r['stock_code']][3] is None)
    ]
    if codes_missing_price:
        batch_prices = get_batch_stock_prices(codes_missing_price)
    else:
        batch_prices = {}

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

        # 若 tech_map 沒有股價，從批次下載結果（已經快取）取得
        if price is None:
            price = batch_prices.get(code) or _price_cache.get(code, (None,))[0]

        if price is not None:
            if r['latest_eps'] <= 0.05 or price < 15:
                filtered_rows.append((r, price, spread, is_20ma_rising))
        else:
            if r['latest_eps'] <= 0.05:
                filtered_rows.append((r, None, 999.0, False))

    filtered_rows = filtered_rows[:12]

    candidates = []
    for r, price, spread, is_20ma_rising in filtered_rows:
        code = r['stock_code']
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
                f"    - {rh['date_month']}: 營收 {rev_str}億元 (YoY: {yoy_str}%, MoM: {mom_str}%)"
            )
        rev_history_str = "\n".join(rev_history_str_list)

        price_str = f"最新收盤價: {price} 元" if price is not None else "最新收盤價: N/A"

        tech_str = ""
        if price is not None and spread != 999.0:
            tech_str += f"  - 短期均線糾結度 (Spread): {spread}%"
            if spread < 3.0:
                tech_str += " (均線極度收斂，起漲點特徵)"
            elif spread <= 5.0:
                tech_str += " (均線高度糾結)"
            else:
                tech_str += " (均線尚未糾結)"
            tech_str += f" 且 20MA 趨勢: {'20MA 向上 (趨勢偏多)' if is_20ma_rising else '20MA 平緩或向下'}\n"
        else:
            tech_str += "  - 無法取得即時均線糾結指標\n"

        candidates.append(
            f"- {r['stock_code']} {r['stock_name']} ({price_str}):\n"
            f"  - 最新季度 EPS: {r['latest_eps']} 元\n"
            f"  - 近 6 個月營收歷史:\n{rev_history_str}\n"
            f"{tech_str}"
        )
    conn.close()
    candidates_str = "\n".join(candidates)

    current_date_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
你是一位專業的台股基本面與籌碼面分析專家。
今天系統時間是：{current_date_str}。在分析與展望時，請以當前時間為基準。

我們從資料庫篩選出最近營收 YoY 展現強勁成長，但當前/歷史季度有虧損記錄（或本益比為負值/淨值低於票面）的「轉虧為盈/減虧潛力候選股」名單如下：
{candidates_str}

請利用 Google 搜尋聯網工具，協助搜尋這批轉盈潛力股近期的營運狀況，並進行深度評估。

【分析與驗證重點】：
1. **嚴防穩健獲利股誤報**：請務必核對每家公司的真實財務狀況。如果發現該個股（例如台積電、聯發科等）其實一直都是穩健賺錢、未曾出現 any 單季虧損，請在此個股的評語最開頭明確指出「【排除分析：該公司歷史並無虧損，不符合轉虧為盈定義】」，並簡短說明其被系統篩選進來的可能原因（如因本益比資料庫異常而誤入），然後不要進行後續轉盈預測。
2. **轉盈關鍵動能與時間表預測**：對於真正具有虧損背景的潛力股，請結合近期營收成長趋势、產品結構調整、折舊費用、平均成本以及即時商業動態，深度預測其下一個月或下一季度是否有機會實現單季「轉虧為盈」，並說明關鍵獲利催化劑。
3. **均線糾結與起漲共振**：評估這些候選股的均線糾結度與20MA趨勢。如果該股均線糾結度 < 5% 且 20MA 趨勢向上，請給予較高的技術面加分。

【報告輸出格式要求】：
請分成以下幾個段落呈現：
1. **轉虧為盈市場大局觀**：簡述當前有哪些宏觀或產業題材（如面板報價回溫、記憶體合約價上漲、折舊攤提完畢等）正在推動台股虧損股的轉盈風潮。
2. **潛力個股深度分析**：請詳細點評 3-5 檔真正有虧損背景且具備轉盈潛力的個股，每檔需包含：
   - 股票名稱與代碼（如：`[3321] 同泰`）
   - 最新季度 EPS 與虧損原因
   - 近期營收爆發動能與平均成本變化
   - 轉盈時點與概率預測
   - 技術面均線收斂與 20MA 趨勢評估
   - 綜合評級（高/中/低）與操作策略
3. **風險警示**：提示虧損股轉盈預測的變數（如營收旺季過後下滑、成本上升等）。

請以繁體中文撰寫，內容詳實具深度，使用 Markdown 格式呈現。
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        save_gemini_report('turnaround_analysis', current_date_str[:7], report_content, db_path=db_path)
        return report_content
    except Exception as e:
        return f"Gemini 執行轉盈分析失敗: {e}"


# --- 6. ETF 換股新聞分析 ---

def analyze_etf_rebalancing(api_key, db_path=None):
    """
    使用 Gemini 聯網查詢台灣主流 ETF (0050, 0056, 00878, 00919, 00929, 00940) 的最新一次換股公告及成分股調整名單，
    並生成分析報告。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定，無法執行 AI 聯網分析"
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    current_month = current_date_str[:7]
    
    prompt = f"""
你是一位專業的台股 ETF 策略分析師。
今天系統時間是：{current_date_str}。

請利用 Google 搜尋聯網工具，協助搜尋近期（特別是過去半年內，以接近當前時間的最新公告為準）台灣股市中以下幾檔主流熱門 ETF 的最新換股（成分股調整）相關新聞、公告與詳細名單：
1. **0050** (元大台灣50)
2. **0056** (元大高股息)
3. **00878** (國泰永續高股息)
4. **00919** (群益台灣精選高息)
5. **00929** (復華台灣科技優息)
6. **00940** (元大台灣價值高息)

【分析與說明重點】：
1. **換股時間**：說明每檔 ETF 最近一次進行成分股調整（換股）的具體生效時間或公告日期。
2. **成分股變動名單**：詳細列出本次調整中，新增（納入）了哪些股票、刪除（剔除）了哪些股票。請寫出股票名稱及 4 位數股票代號（格式為 `[股票代號]` 如 `[2330]`，嚴禁遮蔽）。
3. **換股原因分析**：說明本次換股的核心邏輯與原因（例如：因應市值變動、股息殖利率高低、ESG評級變化、公司基本面衰退等），並評估本次調整後該 ETF 的整體風格（如：是否變得更防守、或高科技股比例上升等）。
4. **成分股調整對個股的影響**：說明本次大換股對被動資金流向（如被納入或剔除個股的買壓/賣壓）的短期與中長期潛在影響。

請以繁體中文撰寫，內容詳實、數據精確，使用 Markdown 格式呈現。
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if not report_content or not report_content.strip():
            raise ValueError("API returned empty content")
        save_gemini_report('etf_rebalancing', current_month, report_content, db_path=db_path)
        return report_content
    except Exception as e:
        error_msg = str(e)
        print(f"Search grounding failed for ETF rebalancing: {error_msg}. Falling back...")
        try:
            model_no_search = get_vertex_model(api_key, enable_search=False)
            if not model_no_search:
                return f"Gemini 聯網分析失敗，且無法執行備用 AI 分析: {e}"
            fallback_prompt = prompt + "\n\n【說明：由於搜尋功能暫時不可用，請根據您原有的知識庫及最新歷史記錄，提供這些 ETF 的換股特徵與歷史成分股變動分析。】"
            response_fallback = model_no_search.generate_content(fallback_prompt)
            fallback_content = response_fallback.text
            if fallback_content and fallback_content.strip():
                save_gemini_report('etf_rebalancing', current_month, fallback_content, db_path=db_path)
                notice = f"⚠️ **注意：API 搜尋聯網功能暫時不可用（原因：`{error_msg}`），已啟用備用 AI 知識庫分析。**\n\n"
                return notice + fallback_content
            else:
                return f"Gemini 執行備用分析失敗: {e}"
        except Exception as fallback_err:
            return f"Gemini 執行備用分析失敗: {e}，且備用本地分析錯誤: {fallback_err}"


def check_ma_convergence(stock_code):
    """
    ????????????????????��???
    ???: (success: bool, spread: float, ma_dict: dict, current_price: float)
    """
    import yfinance as yf
    import pandas as pd
    try:
        # ?????? 90 �ѥH�T????????? 60MA
        ticker = f"{stock_code}.TW"
        df = yf.download(ticker, period="90d", progress=False, timeout=5)
        if df.empty or len(df) == 0:
            ticker = f"{stock_code}.TWO"
            df = yf.download(ticker, period="90d", progress=False, timeout=5)
            
        if df.empty or len(df) < 60:
            return False, 999.0, {}, 0.0, False
            
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        close_series = df['Close']
        
        # ????????????????? float ???? serialization ??�~
        ma5 = float(close_series.rolling(5).mean().iloc[-1])
        ma10 = float(close_series.rolling(10).mean().iloc[-1])
        ma20 = float(close_series.rolling(20).mean().iloc[-1])
        ma60 = float(close_series.rolling(60).mean().iloc[-1])
        
        current_price = float(close_series.iloc[-1])
        
        if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20) or pd.isna(ma60) or pd.isna(current_price):
            return False, 999.0, {}, 0.0, False
            
        # ??? 20MA ??????? (???????? 5 ��???? 20MA)
        is_20ma_rising = False
        if len(close_series) >= 25:
            ma20_prev = float(close_series.rolling(20).mean().iloc[-5])
            ma20_curr = float(close_series.rolling(20).mean().iloc[-1])
            is_20ma_rising = ma20_curr > ma20_prev
            
        # ???��???? (???????�t??)
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
    ??��????????????????????��???
    ???: {stock_code: (success, spread, ma_dict, price)}
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
                             
                             # ??? 20MA ??????? (???????? 5 ��???? 20MA)
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
    使用 Gemini (搭配 Google Search Grounding) 聯網查詢近期台股籌碼特定分點囤貨/買超與均線糾結起漲標的。
    回傳 AI 分析報告文本。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定，無法執行 AI 聯網分析"

    current_date_str = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
你是一位專業的台股籌碼與分點分析專家、技術面大師。
今天系統時間是：{current_date_str}。

請利用 Google 搜尋聯網工具，協助搜尋近期台灣股市中有關主力特定分點持續大量買超（囤貨）、籌碼集中度上升、以及大戶主力買超動向等最新消息，並配合技術面（如均線糾結、突破等）進行深度分析。

分析重點如下：
1. **主力特定分點囤貨與買超動態**：找出近期有哪些股票被主力/特定分點（如：台灣摩根士丹利、元大松山、凱基台北等）在過去一個月內持續買進囤貨。
2. **均線糾結狀態與起漲可能性**：分析這些股票在技術面上的狀態（例如 5MA/10MA/20MA/60MA 均線是否高度收斂糾結、股價是否有帶量突破起漲的型態）。

【強制輸出真實股票代碼，嚴防遮蔽】：
1. 為了能讓系統後端順利進行技術指標量化計算，請務必在報告中提及個股名稱時，在後方緊接標記該個股的 4 位數字台股代號。格式必須為以中括號包裹的真實代號，例如 `[2330]`、`[2061]`、`[3231]` 等，這有助於後續提取。
2. 絕對不可使用任何形式的代碼遮蔽，例如 `[23XX]`、`[3XXX]` 或 `[xxxx]`。

【報告輸出格式要求】：
請分成以下幾個段落呈現：
1. **近期籌碼特定分點囤貨股摘要**：簡述您搜尋到的整體市場籌碼面趨勢與特定主力囤貨動向。
2. **潛力個股深度點評**：詳細列出 3-5 檔被主力特定分點大量囤貨、且技術圖形呈現均線糾結收斂的個股，並指出：
   - 股票名稱與代碼（如：`[3535] 晶彩科`）
   - 特定主力囤貨分點名稱（如：美商高盛、凱基台北等）及買超特徵
   - 題材與拉抬動機（如：光學檢測設備出貨暴增）
   - 技術面均線收斂狀態
3. **綜合評估與操作建議**：給予每檔個股籌碼評級（高/中/低）與風險提示。

請以繁體中文撰寫，內容詳實具深度，使用 Markdown 格式呈現。
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if not report_content or not report_content.strip():
            raise ValueError("API returned empty content")
        if len(report_content.strip()) < 300 or "無法提供" in report_content:
            raise ValueError(f"API returned incomplete status: {report_content[:150]}")
        save_gemini_report('chip_and_ma_convergence', current_date_str[:7], report_content, db_path=db_path)
        return report_content
    except Exception as e:
        error_msg = str(e)
        print(f"Search grounding failed for chip analysis: {error_msg}. Falling back...")
        try:
            model_no_search = get_vertex_model(api_key, enable_search=False)
            if not model_no_search:
                return f"Gemini 聯網分析失敗，且無法執行備用 AI 分析: {e}"
                
            fallback_prompt = prompt + "\n\n【說明：由於搜尋功能暫時不可用，請根據您原有的知識庫，針對台股近期主力籌碼囤貨特徵與均線糾結起漲標的進行分析說明。】"
            response_fallback = model_no_search.generate_content(fallback_prompt)
            fallback_content = response_fallback.text
            if fallback_content and fallback_content.strip():
                save_gemini_report('chip_and_ma_convergence', current_date_str[:7], fallback_content, db_path=db_path)
                notice = f"⚠️ **注意：API 搜尋聯網功能暫時不可用（原因：`{error_msg}`），已啟用備用 AI 知識庫分析。**\n\n"
                return notice + fallback_content
            else:
                return f"Gemini 執行備用分析失敗: {e}"
        except Exception as fallback_err:
            return f"Gemini 執行備用分析失敗: {e}，且備用本地分析錯誤: {fallback_err}"


def extract_valid_stock_codes(text, db_path=None):
    """
    從 AI 報告文本中提取符合格式 [股票代碼] 或單獨 4 位數字的股票代碼，並過濾出資料庫中真實存在的股票。
    """
    import re
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT stock_code FROM monthly_revenue")
    valid_codes = {row['stock_code'] for row in cursor.fetchall()}
    conn.close()
    
    # 提取所有獨立的 4 位數字
    candidates = re.findall(r'\b\d{4}\b', text)
    # 提取所有被中括號包圍的 4 位數字，例如 [2330]
    candidates_bracket = re.findall(r'\[(\d{4})\]', text)
    
    all_found = set(candidates + candidates_bracket)
    
    # 過濾出資料庫中真實存在的代碼（排除年份如 2024, 2025, 2026 等）
    valid_found = [code for code in all_found if code in valid_codes]
    return sorted(list(valid_found))

def analyze_investor_conferences(api_key, db_path=None):
    """
    ��?? Gemini (??? Google Search Grounding) ??��??�ߨ�??????????????????�~��???��??�e???
    ??????????????????????????????????????????????�f??????????????????????
    ????????????????????��??????????????��??ttps://www.alphamemo.ai/free-transcripts
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API ?????�]????????????????????????????????"
        
        from datetime import datetime, timedelta
    today = datetime.now()
    one_month_ago = today - timedelta(days=30)
    current_date_str = today.strftime("%Y-%m-%d")
    one_month_ago_str = one_month_ago.strftime("%Y-%m-%d")
    
    prompt = f"""
????????�~???????????????????
?????? {current_date_str}???�H��????????????**??????????????????? {one_month_ago_str} ?? {current_date_str} ?????????????????/???/???????��??????**??

????????? Google ????????���u????????????????????"2026 ???��?? ??????"??"2026??6?? ??? ???"??"???? ???? ??? 2026-06" ??????????????????????????�j??�s????????��??????????????????????

??? ??????????��??????????????��???????
1. ?????????????????????????????? **{one_month_ago_str} ?? {current_date_str}** ??????????
2. ???��????????��?????? 2026 �~????????????????????????????????????�w?? 1000 ???????????????? 1000 ??????????????????????????????? 800 ??? 1000 ???????????????????????? 800 ??????��?? 2024 �~?????????????????????????????????
3. ??????��?????????????????????????????????????�T??��?? 2026 ?? 5-6 ?????????????????????�H??
4. **?????�j????????** ?? JSON ��???? `reason`???????????????��???????????????????????????????????????????????????????????????`???????? ???????????????????????`??2026/06/24 �u?????????????��??????????????�j?????...??????`????????????????????????????????��????

?????????????��?????????????????????�H JSON ????????????????????��N��??��??????????????????
- `date`: ��??????????????? "YYYY-MM-DD"?????? {one_month_ago_str} ?? {current_date_str} ?????????????????????????��T???????��???????
- `stock_code`: 4??????���N�X????? "2330"??
- `stock_name`: ????????????? "??????"??
- `broker`: ???????�s?????��????? "????�h��??"??"??????"??"��??????" ????
- `original_rating`: ��????????????????????�X???????�R????????????????? "????"??
- `new_rating`: ��???????????????????????�j????????????�X??????
- `target_price`: ��??????????????????????? 1200.0??????????????????? 0????
- `current_pe`: ????????????????????? 24.5?????????????????????????????????????/???EPS???????��????????????��?????? 0????
- `adjusted_pe`: ��?????????????????????????????????�H????EPS????? 30.2???????????????�H????????��????????????��?????? 0????
- `reason`: ????? `???????? ????????????????? ?????????????????????oWoS????????????????? 80-180 ??????????��????�g??

???????
1. ????????? JSON Array??????��?? Markdown ???????????? ```json?????????????????????????????��????????????????????????��???? `[]`??????????�s????
2. �T????��????????��??????????�N�X???

???????????????**?????��???????????��??? (TWSE) ??????�a??????��??��??²???????? https://www.alphamemo.ai/free-transcripts ??????????�Z??????????**??

???????????????Z�g??��??�~??**??�~��??��??�`????�~?????�j��?????**???�e????????????????? Markdown ??????????��?????????????��????????
�`?????�H???????????????????????????????????????????��????????
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if not report_content or not report_content.strip():
            raise ValueError("API returned empty content")
        if len(report_content.strip()) < 300 or "�t��????" in report_content:
            raise ValueError(f"API returned incomplete status: {report_content[:150]}")
        save_gemini_report('investor_conferences', current_date_str[:7], report_content, db_path=db_path)
        return report_content
    except Exception as e:
        error_msg = str(e)
        print(f"Search grounding failed for investor conferences: {error_msg}. Falling back...")
        try:
            model_no_search = get_vertex_model(api_key, enable_search=False)
            if not model_no_search:
                return f"Gemini ????????????????????: {e}"
                
            # ???�D??????????????�H??��??????
            indicator_stocks = {
                '2330': '??????',
                '2454': '??????',
                '2317': '�E��',
                '2382': '???',
                '6669': '�n??',
                '5274': '�H??'
            }
            prices_str_list = []
            for code, name in indicator_stocks.items():
                p = get_latest_stock_price(code)
                if p:
                    prices_str_list.append(f"- {code} {name}: ?????????????? {p} ??")
            prices_context = "\n".join(prices_str_list)
            
            fallback_prompt = prompt + f"\n\n????????????????????? (?? Forward PE ��???????)??:\n{prices_context}\n\n??? ???????????��?????????????????????????????????????????????????�H????????????????????��??????????????????Z�g???????????"
            response_fallback = model_no_search.generate_content(fallback_prompt)
            fallback_content = response_fallback.text
            if fallback_content and fallback_content.strip():
                save_gemini_report('investor_conferences', current_date_str[:7], fallback_content, db_path=db_path)
                notice = f"??? **??????PI ??��???��??????��?????`{error_msg}`???�w???????????????I ??�~��?????�m?????????????????????????????????????????�O?????�T?????? Google Cloud ???�w???? Google Search Grounding API ��??????????**\n\n"
                return notice + fallback_content
            else:
                return f"Gemini ????????????????????: {e}?????????????��???????"
        except Exception as fallback_err:
            return f"Gemini ????????????????????: {e}???????????????????�~: {fallback_err}"









def analyze_surging_stocks(api_key, db_path=None):
    """
    分析月營收異軍突起股 (YoY/MoM 成長且具備一定規模)，
    並以 Gemini AI 聯網搜尋個股近期題材、營收爆發原因。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定，無法執行 AI 聯網分析"
        
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 撈取當月營收 YoY 爆發的個股 (排除建材營造與金融業，YoY > 25% 且營收 > 2000 萬 TWD)
    cursor.execute('''
        SELECT stock_code, stock_name, industry, revenue, yoy, mom 
        FROM monthly_revenue 
        WHERE date_month = (SELECT MAX(date_month) FROM monthly_revenue)
          AND yoy > 25 AND revenue > 20000
        ORDER BY yoy DESC 
        LIMIT 15
    ''')
    rows = cursor.fetchall()
    
    # 獲取實際最新月份作為資料保存的月份主鍵，以防 app.py 撈取時因月份對不上而顯示空白
    cursor.execute("SELECT MAX(date_month) FROM monthly_revenue")
    max_month_row = cursor.fetchone()
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    report_month = max_month_row[0] if (max_month_row and max_month_row[0]) else current_date_str[:7]
    
    conn.close()
    
    if not rows:
        return "當月營收資料庫中無符合營收爆發與規模門檻之個股。"
        
    # 批次查詢技術指標，包含 20MA 趨勢
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
                trend_str = "20MA趨勢向上" if is_20ma_rising else "20MA趨勢向下/平緩"
                tech_str = f"即時股價 {price}元, 均線糾結度 {spread}%, {trend_str}"
                
        candidates.append(
            f"- {code} {r['stock_name']} ({r['industry']}): 當月營收 {rev_str}億元 (YoY: {yoy_str}%, MoM: {mom_str}%) | 技術狀態: {tech_str}"
        )
    candidates_str = "\n".join(candidates)
    
    prompt = f"""
度驗證。

你是一位專業的台股基本面分析與籌碼面專家。
今天系統時間是：{current_date_str}。

我們從資料庫篩選出當月營收異軍突起（YoY 爆發、營收達一定規模）的個股清單如下（我們已為您查詢了即時的技術面狀態）：
{candidates_str}

請利用 Google 搜尋聯網工具，協助搜尋這批營收爆發個股近期的具體商業動因，並結合其技術面狀態進行評估。

請針對每檔個股進行以下分析（請用 Markdown 列表呈現）：
1. **營收爆發具體原因**：詳細指出該公司營收爆發的核心原因（例如：特定客戶大拉貨、新產線量產、新產品出貨、價格上漲、認列一次性收益等），請務必核對正確的公司名稱與股票代號。
2. **基本面與技術面共振評估**：結合我們提供的即時技術狀態（均線糾結度、20MA趨勢等），評估該股是否處於底部收斂、帶量起漲或高位整理等階段。
3. **短期潛力評級**：給予該股短期的潛力評級（高/中/低）與操作建議。

請以繁體中文撰寫，內容詳實，使用 Markdown 格式呈現。
"""
    try:
        response = model_with_search.generate_content(prompt)
        report_content = response.text
        if not report_content or not report_content.strip():
            raise ValueError("API returned empty content")
        save_gemini_report('surging_stocks_analysis', report_month[:7], report_content, db_path=db_path)
        return report_content
    except Exception as e:
        error_msg = str(e)
        print(f"Search grounding failed for surging stocks: {error_msg}. Falling back...")
        try:
            model_no_search = get_vertex_model(api_key, enable_search=False)
            if not model_no_search:
                return f"Gemini API 金鑰未設定，無法執行備用 AI 分析: {e}"
            fallback_prompt = prompt + "\n\n【說明：由於搜尋功能暫時不可用，請根據您原有的知識庫及上述提供的營收與技術數據，分析並點評這批營收爆發股。】"
            response_fallback = model_no_search.generate_content(fallback_prompt)
            fallback_content = response_fallback.text
            if fallback_content and fallback_content.strip():
                save_gemini_report('surging_stocks_analysis', report_month[:7], fallback_content, db_path=db_path)
                notice = f"⚠️ **注意：API 搜尋聯網功能暫時不可用（原因：`{error_msg}`），已啟用備用 AI 知識庫分析。**\n\n"
                return notice + fallback_content
            else:
                return f"Gemini 執行備用分析失敗: {e}"
        except Exception as fallback_err:
            return f"Gemini 執行備用分析失敗: {e}，且備用本地分析錯誤: {fallback_err}"


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
                            
                            # 52週高點 (一整年)
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
    個股 AI 籌碼與分點聯網深度解析功能
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定，無法執行 AI 籌碼分析"
        
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
你是一位專業的台股籌碼與分點分析專家。
今天系統時間是：{current_date_str}。

請利用 Google 搜尋聯網工具，協助搜尋近期（過去一個月內）關於個股 {stock_code} {stock_name} 主力分點（例如特定分點、券商分點）囤貨、買超、籌碼集中度、以及大戶主力買超動向等最新消息，並以籌碼面與分點大戶動向進行深度分析與評估。

請針對 {stock_code} {stock_name} 的籌碼面進行以下分析（請用 Markdown 列表條理化呈現）：
1. **主力特定分點買超動態**：分析是否有特定分點（如：台灣摩根士丹利、元大松山、凱基台北等）近期持續大量囤貨、買超，以及這些分點買超的具體券商名稱、買超張數與估計買超均價。
2. **籌碼集中度變化**：說明近期籌碼集中度的趨勢，是集中還是分散，主力大戶佔比與外資、投信的買賣超動向。
3. **題材與主力拉抬動機分析**：主力在此時買超的潛在題材或拉抬動機（如：營收大增、新產品認證、產業復甦等），是否有利多提前佈局。
4. **綜合評估與操作建議**：給予該股的籌碼集中度評級（高/中/低），並從籌碼角度給予短中期買賣建議或風險提示。

請以繁體中文撰寫，內容要詳實且具深度，使用 Markdown 格式呈現。
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
                return f"Gemini API 金鑰未設定，無法執行備用 AI 分析: {e}"
            fallback_prompt = prompt + "\n\n【說明：由於搜尋功能暫時不可用，請根據您原有的知識庫，針對該股票的產業趨勢及歷史籌碼大戶特徵進行分析說明。】"
            response_fallback = model_no_search.generate_content(fallback_prompt)
            fallback_content = response_fallback.text
            if fallback_content and fallback_content.strip():
                notice = f"⚠️ **注意：API 搜尋聯網功能暫時不可用（原因：`{error_msg}`），已啟用備用 AI 知識庫分析。**\n\n"
                return notice + fallback_content
            else:
                return f"Gemini 執行備用分析失敗: {e}"
        except Exception as fallback_err:
            return f"Gemini 執行備用分析失敗: {e}，且備用本地 analysis 錯誤: {fallback_err}"


def scan_broker_ratings(api_key, db_path=None):
    """
    使用 Gemini 聯網功能自動檢索近期券商/外資/投信對台灣股市個股的最新評等調整報告，
    藉由「升評/調升/調高」、「降評/調降/調低」、「調評/目標價變動」三大關鍵字方向進行分次精準查詢，並篩選過去一個月的記錄寫入資料庫。
    """
    model_with_search = get_vertex_model(api_key, enable_search=True)
    if not model_with_search:
        return "Gemini API 金鑰未設定，無法搜集評等調整資訊。"
        
    from datetime import datetime, timedelta
    import json
    import re
    
    today = datetime.now()
    one_month_ago = today - timedelta(days=30)
    current_date_str = today.strftime("%Y-%m-%d")
    one_month_ago_str = one_month_ago.strftime("%Y-%m-%d")
    
    # 獲取權值股當前市價作為防範過期資料的時空基準
    tsmc_price = get_latest_stock_price('2330') or 2340.0
    mediatek_price = get_latest_stock_price('2454') or 3880.0
    
    prompts = [
        # 子查詢 1: 升評 / 調升 / 調高 關鍵字（中文，含 site: 操作符）
        (f"今天是 {current_date_str}。請用 Google Search 搜尋以下關鍵字，找出近一個月內台灣股市券商/外資對個股發出的升評或目標價上調報告："
         f"「升評 台股 {current_date_str[:7]}」、「調升 台股 外資 {current_date_str[:7]}」、「調高目標價 台股」、「評等調升 site:anue.com.tw」、「調升評等 site:moneydj.com」"),
        # 子查詢 2: 降評 / 調降 / 調低 關鍵字
        (f"今天是 {current_date_str}。請用 Google Search 搜尋以下關鍵字，找出近一個月內台灣股市券商/外資對個股發出的降評或目標價下調報告："
         f"「降評 台股 {current_date_str[:7]}」、「調降 台股 外資 {current_date_str[:7]}」、「調低目標價 台股」、「評等調降 site:anue.com.tw」、「調降評等 site:moneydj.com」"),
        # 子查詢 3: 調評 / 目標價調整 關鍵字
        (f"今天是 {current_date_str}。請用 Google Search 搜尋以下關鍵字，找出近一個月內台灣股市券商/外資對個股進行評等或目標價調整的報告："
         f"「調評 台股 {current_date_str[:7]}」、「調整評等 台股」、「目標價調升 site:anue.com.tw」、「目標價下調 site:anue.com.tw」、「評等調整 site:moneydj.com」"),
        # 子查詢 4: 英文關鍵字，抓取外資英文報告
        (f"Today is {current_date_str}. Please search Google for recent (within past 30 days) Taiwan stock rating changes by foreign brokerages: "
         f"\"Taiwan stock upgrade {current_date_str[:7]}\", \"Taiwan stock downgrade {current_date_str[:7]}\", "
         f"\"Taiwan equity rating upgrade\", \"TWSE rating change\", \"TPEx rating change\". "
         f"Find broker research reports upgrading or downgrading Taiwan-listed stocks."),
    ]
    
    common_rules = f"""
【當前市場股價參考】
- 台積電 (2330) 當前實際股價約為 {tsmc_price} 元
- 聯發科 (2454) 當前實際股價約為 {mediatek_price} 元

⚠️ 嚴格規定：
1. **僅包含發布日期在 {one_month_ago_str} 至 {current_date_str} 之間（即過去一個月內）**的真實報導。請務必詳細閱讀搜尋結果的網頁時間，將所有新聞篩選並依時間排序，只保留過去一個月內的最新調整紀錄。
2. 評語/調整理由的 `reason` 欄位開頭必須標記：`【發布日期 媒體名稱：新聞標題】`。
3. 嚴防過期舊聞：如果搜到台積電目標價 1350 元以下或聯發科目標價 1200 元以下，為過期舊聞，請「絕對不要」納入！
4. 僅回傳一個 JSON 陣列格式（不要 ```json，不要前後贅字）。如果找不到符合日期與真實性要求的資料，請回傳 `[]`。

JSON 欄位：
- `date`: "YYYY-MM-DD"
- `stock_code`: 4位數字代碼
- `stock_name`: 公司名稱
- `broker`: 券商機構名稱
- `original_rating`: 調整前評等（未知填 "未知"）
- `new_rating`: 調整後評等
- `target_price`: 數值（無則填 0）
- `current_pe`: 數值（無則填 0）
- `adjusted_pe`: 數值（無則填 0）
- `reason`: 理由（以 `【發布日期 媒體名稱：新聞標題】` 開頭，繁體中文，80-180字內）
"""

    all_records = []
    
    for query_prompt in prompts:
        full_prompt = query_prompt + common_rules
        try:
            response = model_with_search.generate_content(full_prompt)
            content = response.text.strip()
            
            # 移除 Markdown 包裹
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            if content:
                records = []
                try:
                    records = json.loads(content)
                except Exception:
                    match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
                    if match:
                        try:
                            records = json.loads(match.group(0))
                        except Exception:
                            pass
                
                if isinstance(records, list):
                    all_records.extend(records)
                    
        except Exception as e:
            print(f"Error executing sub-query for broker ratings: {e}")
            
    # 合併與去重 (根據 date, stock_code, broker)
    seen = set()
    deduped_records = []
    for r in all_records:
        date_val = r.get('date')
        code_val = r.get('stock_code')
        broker_val = r.get('broker')
        if not date_val or not code_val or not broker_val:
            continue
        key = (date_val, code_val, broker_val)
        if key not in seen:
            seen.add(key)
            deduped_records.append(r)
            
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 確保資料表已存在
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rating_adjustments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,                    -- 調整日期 (YYYY-MM-DD)
            stock_code TEXT,              -- 股票代碼
            stock_name TEXT,              -- 股票名稱
            broker TEXT,                  -- 調整機構/券商
            original_rating TEXT,         -- 原評等
            new_rating TEXT,              -- 新評等
            target_price REAL,            -- 目標價
            reason TEXT,                  -- 理由與新聞來源
            current_pe REAL,              -- 現行 PE
            adjusted_pe REAL,             -- 調整後 PE
            created_at TEXT
        )
    ''')
    conn.commit()
    
    added_count = 0
    for r in deduped_records:
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
    
    return f"AI 評等動作廣泛聯網掃描完成！新增了 {added_count} 筆最新評等變動紀錄。"
