"""
腳本名稱：capture_bonus_screenshots.py
用途：自動針對老師規定的「加總分 1分」與「加總分 2分」功能進行自動化操作與高清截圖，
並直接儲存至使用者的桌面資料夾：/Users/esleyw/Desktop/多元線性回歸_加分項目截圖/
包含：
  1. 加總分 1分：有下拉視窗呈現兩種 problem（主題一 50 Startups vs 主題二 Boston Housing）
  2. 加總分 2分：可任意輸入 Dataset（如 https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv），完成自動解析、變數選取、純 JS 線性回歸求解與成果呈現
"""

import os
import time
from playwright.sync_api import sync_playwright

DESKTOP_DIR = "/Users/esleyw/Desktop/多元線性回歸_加分項目截圖"
MAIN_DESKTOP_DIR = "/Users/esleyw/Desktop/多元線性回歸_作業截圖"
os.makedirs(DESKTOP_DIR, exist_ok=True)
os.makedirs(MAIN_DESKTOP_DIR, exist_ok=True)
WEB_URL = "http://localhost:8000"

SAMPLE_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"


def capture_bonus():
    print(f"🚀 開始加分功能截圖程序，儲存至：{DESKTOP_DIR}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 950},
            device_scale_factor=2
        )
        page = context.new_page()

        # 1. 載入網頁
        page.goto(WEB_URL, wait_until="networkidle")
        page.wait_for_timeout(1500)

        # =============================================================
        # 【加總分 1分】下拉視窗呈現兩種 Problem
        # =============================================================
        print("📸 [1/4] 正在截取【加總分 1分】：頂部下拉視窗呈現兩種 Problem 特寫...")
        header_el = page.locator(".site-header")
        header_el.screenshot(
            path=os.path.join(DESKTOP_DIR, "加分1分_01_頂部下拉視窗呈現兩種Problem.png")
        )

        print("📸 [2/4] 正在截取【加總分 1分】：下拉切換至主題二 Boston Housing 視角...")
        page.select_option("#problem-dropdown", "boston")
        page.wait_for_timeout(800)
        boston_sec = page.locator("#boston-section")
        boston_sec.scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.screenshot(
            path=os.path.join(DESKTOP_DIR, "加分1分_02_下拉切換至主題二_Boston_Housing成果.png"),
            clip={"x": 100, "y": 80, "width": 1240, "height": 850}
        )

        # =============================================================
        # 【加總分 2分】可任意輸入 Dataset 網址並分析
        # =============================================================
        print("📸 [3/4] 正在截取【加總分 2分】：輸入 Dataset 網址並讀取解析預覽...")
        # 切換至自訂資料集視角
        page.select_option("#problem-dropdown", "custom")
        page.wait_for_timeout(800)

        custom_sec = page.locator("#custom-dataset-section")
        custom_sec.scroll_into_view_if_needed()
        page.wait_for_timeout(500)

        # 填入指定的資料集網址
        url_input = page.locator("#custom-dataset-url")
        url_input.fill(SAMPLE_URL)
        page.wait_for_timeout(300)

        # 點擊「讀取網址 CSV」按鈕
        fetch_btn = page.locator("#btn-fetch-custom-url")
        fetch_btn.click()
        # 等待遠端 GitHub 檔案下載與解析完成
        page.wait_for_timeout(3000)

        # 截取已成功讀取並呈現前 5 筆資料預覽、目標 y 下拉與特徵 X 清單
        custom_sec.scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.screenshot(
            path=os.path.join(DESKTOP_DIR, "加分2分_01_任意輸入Dataset網址成功解析與變數設定.png"),
            clip={"x": 100, "y": 80, "width": 1240, "height": 880}
        )

        print("📸 [4/4] 正在截取【加總分 2分】：執行純 JS 線性回歸求解與成果呈現...")
        # 點擊「執行純 JavaScript 線性回歸求解」按鈕
        run_btn = page.locator("#btn-run-custom-training")
        run_btn.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        run_btn.click()
        page.wait_for_timeout(1500)

        # 滾動至回歸結果面板（包含 R²、RMSE、MAE、回歸方程式與 Actual vs Pred 散佈圖）
        results_panel = page.locator("#custom-dataset-results")
        results_panel.scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        results_panel.screenshot(
            path=os.path.join(DESKTOP_DIR, "加分2分_02_自訂資料集即時回歸求解指標與圖表成果.png")
        )

        browser.close()

    # 同步複製一份至作業截圖總資料夾中，方便查閱
    import shutil
    for f in os.listdir(DESKTOP_DIR):
        src = os.path.join(DESKTOP_DIR, f)
        dst = os.path.join(MAIN_DESKTOP_DIR, f)
        shutil.copy(src, dst)

    print(f"\n🎉 加分項目截圖已全數成功產出！")
    print(f"  專屬資料夾：{DESKTOP_DIR}")
    print(f"  總截圖資料夾：{MAIN_DESKTOP_DIR}")


if __name__ == "__main__":
    capture_bonus()
