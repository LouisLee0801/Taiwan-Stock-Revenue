import requests
import re
from datetime import datetime
import pandas as pd
from io import StringIO

# --- 輔助清洗與轉換函數 ---

def clean_float(val):
    """清理數值字串並轉換成 float，若為空或無法轉換則傳回 None"""
    if val is None:
        return None
    val_str = str(val).strip().replace(',', '')
    if val_str == '' or val_str == '-' or val_str.lower() == 'nan' or val_str.lower() == 'null':
        return None
    try:
        return float(val_str)
    except ValueError:
        return None

def clean_int(val):
    """清理整數字串並轉換成 int"""
    f = clean_float(val)
    return int(f) if f is not None else None

def convert_roc_year_month(roc_ym):
    """
    將民國年月 (例如 '11504' 或 '115/04') 轉換為西元年月字串 (格式: 'YYYY-MM')
    """
    if not roc_ym:
        return None
    roc_ym_str = str(roc_ym).strip().replace('/', '')
    
    # 匹配後面兩位是月份，前面是民國年
    if len(roc_ym_str) >= 4:
        try:
            month = int(roc_ym_str[-2:])
            year = int(roc_ym_str[:-2])
            ad_year = year + 1911
            return f"{ad_year:04d}-{month:02d}"
        except ValueError:
            pass
    return None

def convert_roc_date(roc_date):
    """
    將民國年月日 (例如 '1150604' 或 '115/06/04') 轉換為西元年月日字串 (格式: 'YYYY-MM-DD')
    """
    if not roc_date:
        return None
    roc_date_str = str(roc_date).strip().replace('/', '')
    
    # 匹配後四位是月日，前面是民國年
    if len(roc_date_str) >= 5:
        try:
            day = int(roc_date_str[-2:])
            month = int(roc_date_str[-4:-2])
            year = int(roc_date_str[:-4])
            ad_year = year + 1911
            return f"{ad_year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass
    return None

# --- API 爬取函數 ---

def fetch_monthly_revenue():
    """
    爬取上市公司與上櫃公司的當月營收資料並進行合併與清理。
    傳回值為字典列表，可直接存入 SQLite。
    """
    records = []
    
    # 1. 爬取上市公司月營收 (TWSE)
    twse_url = 'https://openapi.twse.com.tw/v1/opendata/t187ap05_L'
    print("Fetching TWSE monthly revenue...")
    try:
        r = requests.get(twse_url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            print(f"TWSE revenue: retrieved {len(data)} records.")
            for row in data:
                # 資料清洗
                date_month = convert_roc_year_month(row.get('資料年月'))
                if not date_month:
                    continue
                
                records.append({
                    'date_month': date_month,
                    'stock_code': str(row.get('公司代號', '')).strip(),
                    'stock_name': str(row.get('公司名稱', '')).strip(),
                    'industry': str(row.get('產業別', '')).strip(),
                    'revenue': clean_float(row.get('營業收入-當月營收')),
                    'last_month_revenue': clean_float(row.get('營業收入-上月營收')),
                    'last_year_revenue': clean_float(row.get('營業收入-去年當月營收')),
                    'mom': clean_float(row.get('營業收入-上月比較增減(%)')),
                    'yoy': clean_float(row.get('營業收入-去年同月增減(%)')),
                    'cum_revenue': clean_float(row.get('累計營業收入-當月累計營收')),
                    'cum_yoy': clean_float(row.get('累計營業收入-前期比較增減(%)')),
                    'notes': str(row.get('備註', '')).strip()
                })
        else:
            print(f"TWSE monthly revenue failed. Status code: {r.status_code}")
    except Exception as e:
        print(f"Error fetching TWSE monthly revenue: {e}")

    # 2. 爬取上櫃公司月營收 (TPEx)
    tpex_url = 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O'
    print("Fetching TPEx monthly revenue...")
    try:
        r = requests.get(tpex_url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            print(f"TPEx revenue: retrieved {len(data)} records.")
            for row in data:
                date_month = convert_roc_year_month(row.get('資料年月'))
                if not date_month:
                    continue
                
                records.append({
                    'date_month': date_month,
                    'stock_code': str(row.get('公司代號', '')).strip(),
                    'stock_name': str(row.get('公司名稱', '')).strip(),
                    'industry': str(row.get('產業別', '')).strip(),
                    'revenue': clean_float(row.get('營業收入-當月營收')),
                    'last_month_revenue': clean_float(row.get('營業收入-上月營收')),
                    'last_year_revenue': clean_float(row.get('營業收入-去年當月營收')),
                    'mom': clean_float(row.get('營業收入-上月比較增減(%)')),
                    'yoy': clean_float(row.get('營業收入-去年同月增減(%)')),
                    'cum_revenue': clean_float(row.get('累計營業收入-當月累計營收')),
                    'cum_yoy': clean_float(row.get('累計營業收入-前期比較增減(%)')),
                    'notes': str(row.get('備註', '')).strip()
                })
        else:
            print(f"TPEx monthly revenue failed. Status code: {r.status_code}")
    except Exception as e:
        print(f"Error fetching TPEx monthly revenue: {e}")
        
    return records

def fetch_daily_pe():
    """
    爬取上市公司與上櫃公司的每日本益比、殖利率、股價淨值比資料並進行合併與清理。
    傳回值為字典列表。
    """
    records = []
    
    # 1. 上市公司 PE/PB/DY (TWSE)
    twse_url = 'https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL'
    print("Fetching TWSE daily PE/PB/DY...")
    try:
        r = requests.get(twse_url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            print(f"TWSE PE: retrieved {len(data)} records.")
            for row in data:
                date = convert_roc_date(row.get('Date'))
                if not date:
                    continue
                
                records.append({
                    'date': date,
                    'stock_code': str(row.get('Code', '')).strip(),
                    'stock_name': str(row.get('Name', '')).strip(),
                    'pe': clean_float(row.get('PEratio')),
                    'dy': clean_float(row.get('DividendYield')),
                    'pb': clean_float(row.get('PBratio'))
                })
        else:
            print(f"TWSE PE failed. Status code: {r.status_code}")
    except Exception as e:
        print(f"Error fetching TWSE PE: {e}")

    # 2. 上櫃公司 PE/PB/DY (TPEx)
    tpex_url = 'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis'
    print("Fetching TPEx daily PE/PB/DY...")
    try:
        r = requests.get(tpex_url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            print(f"TPEx PE: retrieved {len(data)} records.")
            for row in data:
                date = convert_roc_date(row.get('Date'))
                if not date:
                    continue
                
                records.append({
                    'date': date,
                    'stock_code': str(row.get('SecuritiesCompanyCode', '')).strip(),
                    'stock_name': str(row.get('CompanyName', '')).strip(),
                    'pe': clean_float(row.get('PriceEarningRatio')),
                    'dy': clean_float(row.get('YieldRatio')),
                    'pb': clean_float(row.get('PriceBookRatio'))
                })
        else:
            print(f"TPEx PE failed. Status code: {r.status_code}")
    except Exception as e:
        print(f"Error fetching TPEx PE: {e}")
        
    return records

def fetch_quarterly_financials():
    """
    爬取上市公司與上櫃公司的季度綜合損益表（僅限一般業），計算利潤率並進行清理。
    傳回值為字典列表。
    """
    records = []
    
    # 1. 上市公司損益表 (TWSE - 一般業)
    twse_url = 'https://openapi.twse.com.tw/v1/opendata/t187ap06_L_ci'
    print("Fetching TWSE quarterly financials...")
    try:
        r = requests.get(twse_url, timeout=25)
        if r.status_code == 200:
            data = r.json()
            print(f"TWSE financials: retrieved {len(data)} records.")
            for row in data:
                # 年度轉換成西元
                roc_year = clean_int(row.get('年度'))
                if not roc_year:
                    continue
                year = roc_year + 1911
                quarter = clean_int(row.get('季別'))
                if not quarter:
                    continue
                
                revenue = clean_float(row.get('營業收入'))
                gross_profit = clean_float(row.get('營業毛利（毛損）'))
                net_profit = clean_float(row.get('本期淨利（淨損）'))
                eps = clean_float(row.get('基本每股盈餘（元）'))
                
                # 計算毛利率與淨利率
                gross_margin = (gross_profit / revenue * 100) if (revenue and gross_profit is not None) else None
                net_margin = (net_profit / revenue * 100) if (revenue and net_profit is not None) else None
                
                records.append({
                    'year': year,
                    'quarter': quarter,
                    'stock_code': str(row.get('公司代號', '')).strip(),
                    'stock_name': str(row.get('公司名稱', '')).strip(),
                    'revenue': revenue,
                    'gross_profit': gross_profit,
                    'net_profit': net_profit,
                    'eps': eps,
                    'gross_margin': gross_margin,
                    'net_margin': net_margin
                })
        else:
            print(f"TWSE financials failed. Status code: {r.status_code}")
    except Exception as e:
        print(f"Error fetching TWSE financials: {e}")

    # 2. 上櫃公司損益表 (TPEx - 一般業)
    tpex_url = 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap06_O_ci'
    print("Fetching TPEx quarterly financials...")
    try:
        r = requests.get(tpex_url, timeout=25)
        if r.status_code == 200:
            data = r.json()
            print(f"TPEx financials: retrieved {len(data)} records.")
            for row in data:
                roc_year = clean_int(row.get('Year'))
                if not roc_year:
                    continue
                year = roc_year + 1911
                quarter = clean_int(row.get('Season'))
                if not quarter:
                    continue
                
                revenue = clean_float(row.get('營業收入'))
                gross_profit = clean_float(row.get('營業毛利（毛損）'))
                net_profit = clean_float(row.get('本期淨利（淨損）'))
                eps = clean_float(row.get('基本每股盈餘（元）'))
                
                gross_margin = (gross_profit / revenue * 100) if (revenue and gross_profit is not None) else None
                net_margin = (net_profit / revenue * 100) if (revenue and net_profit is not None) else None
                
                records.append({
                    'year': year,
                    'quarter': quarter,
                    'stock_code': str(row.get('SecuritiesCompanyCode', '')).strip(),
                    'stock_name': str(row.get('CompanyName', '')).strip(),
                    'revenue': revenue,
                    'gross_profit': gross_profit,
                    'net_profit': net_profit,
                    'eps': eps,
                    'gross_margin': gross_margin,
                    'net_margin': net_margin
                })
        else:
            print(f"TPEx financials failed. Status code: {r.status_code}")
    except Exception as e:
        print(f"Error fetching TPEx financials: {e}")
        
    return records

# --- 歷史資料爬取函數 ---

def parse_mops_revenue_from_html(year, month, market='sii'):
    """
    從舊版公開資訊觀測站 (mopsov) 靜態 HTML 中解析月度營收表格。
    """
    roc_year = year - 1911
    url = f'https://mopsov.twse.com.tw/nas/t21/{market}/t21sc03_{roc_year}_{month}_0.html'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        print(f"MOPS HTML {url} returned status code: {r.status_code}")
        return []
        
    r.encoding = 'big5'
    
    # 處理 Pandas read_html 解析
    dfs = pd.read_html(StringIO(r.text))
    records = []
    current_industry = '未知'
    
    for df in dfs:
        found_industry = False
        
        # 1. 檢查欄位名稱 (headers) 是否包含產業別
        cols_to_check = []
        if isinstance(df.columns, pd.MultiIndex):
            for col in df.columns:
                cols_to_check.extend([str(c) for c in col])
        else:
            cols_to_check = [str(c) for c in df.columns]
            
        for col_str in cols_to_check:
            if '產業別：' in col_str:
                match = re.search(r'產業別：([^\s\xa0\n]+)', col_str)
                if match:
                    current_industry = match.group(1).strip()
                    if '單位' in current_industry:
                        current_industry = current_industry.split('單位')[0]
                    elif '單' in current_industry:
                        current_industry = current_industry.split('單')[0]
                    found_industry = True
                    break
        
        # 2. 檢查儲存格內容
        if not found_industry:
            for col in df.columns:
                for val in df[col]:
                    val_str = str(val)
                    if '產業別：' in val_str:
                        match = re.search(r'產業別：([^\s\xa0\n]+)', val_str)
                        if match:
                            current_industry = match.group(1).strip()
                            if '單位' in current_industry:
                                current_industry = current_industry.split('單位')[0]
                            elif '單' in current_industry:
                                current_industry = current_industry.split('單')[0]
                            found_industry = True
                            break
                if found_industry:
                    break
        
        if found_industry:
            continue
            
        # 判斷是否為數據表格
        col_str = str(df.columns.tolist())
        if '公司 代號' in col_str or '公司代號' in col_str:
            if isinstance(df.columns, pd.MultiIndex):
                flat_cols = []
                for col in df.columns:
                    flat_cols.append(col[1] if col[1] else col[0])
                df.columns = flat_cols
                
            df.columns = [str(c).replace(' ', '') for c in df.columns]
            
            for _, row in df.iterrows():
                code = str(row.get('公司代號', '')).strip()
                if not code or code == 'nan' or '合計' in code or '合計' in str(row.get('公司名稱', '')):
                    continue
                
                if not re.match(r'^\d+$', code):
                    continue
                    
                records.append({
                    'date_month': f"{year}-{month:02d}",
                    'stock_code': code,
                    'stock_name': str(row.get('公司名稱', '')).strip(),
                    'industry': current_industry,
                    'revenue': clean_float(row.get('當月營收')),
                    'last_month_revenue': clean_float(row.get('上月營收')),
                    'last_year_revenue': clean_float(row.get('去年當月營收')),
                    'mom': clean_float(row.get('上月比較增減(%)') or row.get('上月比較增減')),
                    'yoy': clean_float(row.get('去年同月增減(%)') or row.get('去年同月增減')),
                    'cum_revenue': clean_float(row.get('當月累計營收')),
                    'cum_yoy': clean_float(row.get('前期比較增減(%)') or row.get('前期比較增減')),
                    'notes': str(row.get('備註', '')).strip() if pd.notna(row.get('備註')) else ''
                })
                
    return records

def fetch_historical_monthly_revenue(year, month):
    """
    獲取特定西元年月 (如 2026, 1) 的歷史月度營收（合併上市與上櫃）。
    """
    records = []
    
    # 1. 爬取上市公司 (sii)
    try:
        sii_records = parse_mops_revenue_from_html(year, month, 'sii')
        records.extend(sii_records)
        print(f"Historical Scrape: retrieved {len(sii_records)} listed records for {year}-{month:02d}")
    except Exception as e:
        print(f"Error parsing listed historical revenue for {year}-{month}: {e}")
        
    # 2. 爬取上櫃公司 (otc)
    try:
        otc_records = parse_mops_revenue_from_html(year, month, 'otc')
        records.extend(otc_records)
        print(f"Historical Scrape: retrieved {len(otc_records)} OTC records for {year}-{month:02d}")
    except Exception as e:
        print(f"Error parsing OTC historical revenue for {year}-{month}: {e}")
        
    return records
