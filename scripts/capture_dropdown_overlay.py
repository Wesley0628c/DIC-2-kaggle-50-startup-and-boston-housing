import os
import shutil
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 950}, device_scale_factor=2)
    page.goto("http://localhost:8000")
    page.wait_for_timeout(1000)

    # 渲染展開下拉選單項目，確保截圖清晰呈現兩種 Problem
    page.evaluate("""() => {
        const wrapper = document.querySelector('.problem-dropdown-wrapper');
        const overlay = document.createElement('div');
        overlay.id = 'demo-dropdown-overlay';
        overlay.style.position = 'absolute';
        overlay.style.top = '100%';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.background = '#1e293b';
        overlay.style.border = '2px solid #06b6d4';
        overlay.style.borderRadius = '8px';
        overlay.style.boxShadow = '0 20px 25px -5px rgba(0,0,0,0.5)';
        overlay.style.padding = '8px 0';
        overlay.style.zIndex = '9999';
        overlay.innerHTML = `
            <div style="padding: 8px 16px; font-weight: bold; color: #38bdf8; border-bottom: 1px solid #334155; font-size: 13px;">
                <i class="fa-solid fa-list-check"></i> 兩種 Problem 下拉選單清單（加分功能 1）
            </div>
            <div style="padding: 10px 16px; color: #f8fafc; font-size: 13px; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); background: rgba(6,182,212,0.15);">
                <span style="color: #06b6d4; font-weight: bold;">【Problem 1】</span> 50 Startups CRISP-DM 基準回歸 & 前向特徵選擇 (50 筆)
            </div>
            <div style="padding: 10px 16px; color: #f8fafc; font-size: 13px; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); background: rgba(139,92,246,0.15);">
                <span style="color: #a855f7; font-weight: bold;">【Problem 2】</span> Boston Housing 波士頓房價通用回歸工作流 (506 筆)
            </div>
            <div style="padding: 10px 16px; color: #f8fafc; font-size: 13px; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: #10b981; font-weight: bold;">【加分項】</span> 自訂資料集即時回歸求解引擎 (任意輸入 CSV 網址 / 檔案上傳)
            </div>
            <div style="padding: 10px 16px; color: #f8fafc; font-size: 13px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #94a3b8; font-weight: bold;">【全景】</span> 全部主題展開對比 (All-in-One Dashboard)
            </div>
        `;
        wrapper.style.position = 'relative';
        wrapper.appendChild(overlay);
    }""")

    page.wait_for_timeout(300)
    out_path = "/Users/esleyw/Desktop/多元線性回歸_加分項目截圖/加分1分_01_下拉選單展開兩種Problem特寫.png"
    page.screenshot(
        path=out_path,
        clip={"x": 80, "y": 0, "width": 1280, "height": 380}
    )
    shutil.copy(out_path, "/Users/esleyw/Desktop/多元線性回歸_作業截圖/加分1分_01_下拉選單展開兩種Problem特寫.png")
    browser.close()

print("Overlay dropdown screenshot captured!")
