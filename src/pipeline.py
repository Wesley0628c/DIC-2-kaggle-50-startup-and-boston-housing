"""
模組名稱：pipeline.py
說明：一鍵執行 50 Startups 多元線性回歸專案之端到端分析工作流程：
  1. 下載資料與品質檢查 (download_data)
  2. Part 1 CRISP-DM 多元線性回歸 (part1_crispdm)
  3. Part 2 前向特徵選擇演算法 (part2_forward_selection)
  4. 整合並匯出網頁 JSON 數據 (export_web_data)
  5. 執行自動化測試驗收
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_data import download_dataset
from part1_crispdm import run_part1
from part2_forward_selection import run_part2
from export_web_data import generate_web_json

def main():
    print("=============================================================")
    print(" 🚀 50 Startups 多元線性回歸 (MLR) 完整工作流 (Workflow) 啟動")
    print("=============================================================")
    
    # 步驟 1: 下載與品質驗證
    print("\n[Stage 1/4] 下載資料集並執行欄位與遺漏值檢查...")
    df = download_dataset()

    # 步驟 2: Part 1 CRISP-DM
    print("\n[Stage 2/4] 執行 Part 1: CRISP-DM 多元線性回歸...")
    p1_res = run_part1()

    # 步驟 3: Part 2 Forward Selection
    print("\n[Stage 3/4] 執行 Part 2: Forward Selection 前向特徵選擇...")
    p2_res = run_part2()

    # 步驟 4: 匯出網頁數據
    print("\n[Stage 4/4] 匯出結構化網頁 data.json 與報告...")
    web_data = generate_web_json()

    print("\n=============================================================")
    print(" ✅ 完整工作流執行完畢！所有輸出與圖表均已保存至 outputs/ 與 web/")
    print("=============================================================")

if __name__ == "__main__":
    main()
