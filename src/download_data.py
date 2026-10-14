"""
模組名稱：download_data.py
說明：負責從主要與備用來源下載 50_Startups.csv，執行嚴格的資料品質與完整性檢查，並保存到 data/ 資料夾。
"""

import os
import io
import requests
import pandas as pd

PRIMARY_URL = "https://raw.githubusercontent.com/harimittapalli/Mulitple-Linear-Reggression/master/50_Startups.csv"
BACKUP_URL = "https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "50_Startups.csv")

EXPECTED_COLUMNS = ["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"]
EXPECTED_ROWS = 50

def download_dataset() -> pd.DataFrame:
    """下載資料集並進行驗證，失敗時拋出例外。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    urls = [("主要來源 (Primary)", PRIMARY_URL), ("備用來源 (Backup)", BACKUP_URL)]
    downloaded_content = None
    success_source = None

    for name, url in urls:
        print(f"[下載] 正在嘗試從 {name} 下載：{url}")
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.text.strip()) > 0:
                downloaded_content = resp.text
                success_source = (name, url)
                print(f"[下載成功] 成功自 {name} 獲取資料。")
                break
            else:
                print(f"[警告] {name} 回應非 200 或內容為空 (Status: {resp.status_code})")
        except Exception as e:
            print(f"[警告] 連線至 {name} 發生異常：{e}")

    if downloaded_content is None:
        raise RuntimeError("無法從主要與備用來源獲取真實資料，下載終止。依規範不得生成假資料。")

    # 驗證 CSV 格式
    try:
        df = pd.read_csv(io.StringIO(downloaded_content))
    except Exception as e:
        raise ValueError(f"CSV 解析失敗：{e}")

    # 檢查欄位
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"資料欄位不符合預期，缺少欄位：{missing_cols}。實際欄位：{list(df.columns)}")

    # 檢查列數
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"資料筆數不符合預期（預期 {EXPECTED_ROWS} 筆，實際 {len(df)} 筆）。")

    # 儲存本機副本
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8") as f:
        f.write(downloaded_content)
    print(f"[檔案保存] 資料已成功寫入：{OUTPUT_CSV_PATH}")

    # 品質檢查
    null_counts = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated().sum())
    state_counts = df["State"].value_counts().to_dict()

    print("\n--- 資料品質檢查報告 ---")
    print(f"資料來源：{success_source[0]} ({success_source[1]})")
    print(f"總筆數：{len(df)}")
    print(f"欄位列表與型別：\n{df.dtypes}")
    print(f"各欄缺失值 (Null Counts)：{null_counts}")
    print(f"重複列數 (Duplicate Rows)：{duplicates}")
    print(f"State 類別分佈：{state_counts}")

    return df

if __name__ == "__main__":
    download_dataset()
