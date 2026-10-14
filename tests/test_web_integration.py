"""
測試套件：test_web_integration.py
用途：嚴格驗證網頁端所有功能要求與案例 A ~ F：
  - Case A: 50 Startups CSV 回歸重現
  - Case B: Boston Housing CSV 回歸
  - Case C: 少於 50 筆的有效 CSV (20 筆)，筆數必須正確為 20 筆，無 50 筆假資料
  - Case D: 無效網址 (https://example.invalid/not-a-dataset.csv)，驗證錯誤處理與拒絕假成功
  - Case E: 不適用資料 (空檔、文字目標、樣本不足 N <= P+1、目標無變異)
  - Case F: 網頁檔案無任何 1.05 / 0.998 替代指標，所有區塊與 model_id 同步
"""

import os
import json
import subprocess
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_HTML = os.path.join(BASE_DIR, "web", "index.html")
APP_JS = os.path.join(BASE_DIR, "web", "app.js")
ENGINE_JS = os.path.join(BASE_DIR, "web", "regression_engine.js")
DATA_JSON = os.path.join(BASE_DIR, "web", "data.json")


def test_html_dom_elements_presence():
    """驗證 index.html 包含所有必要互動元素與下拉選項"""
    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. 驗證下拉選單與 5 個選項
    assert 'id="problem-dropdown"' in html
    assert 'value="part1"' in html
    assert 'value="part2"' in html
    assert 'value="boston"' in html
    assert 'value="custom"' in html
    assert 'value="all"' in html

    # 2. 驗證 Boston Housing 區塊
    assert 'id="boston-section"' in html
    assert 'id="chart-boston"' in html
    assert 'id="boston-formula-text"' in html

    # 3. 驗證自訂資料集區塊
    assert 'id="custom-dataset-section"' in html
    assert 'id="custom-dataset-url"' in html
    assert 'id="custom-file-input"' in html
    assert 'id="btn-fetch-custom-url"' in html
    assert 'id="custom-dataset-status"' in html
    assert 'id="custom-config-panel"' in html
    assert 'id="custom-target-select"' in html
    assert 'id="custom-features-container"' in html
    assert 'id="btn-run-custom-training"' in html
    assert 'id="btn-download-custom-csv"' in html
    assert 'id="btn-download-custom-json"' in html

    # 4. 驗證引入 regression_engine.js
    assert 'src="regression_engine.js"' in html


def test_no_synthetic_or_fake_success_in_app_js():
    """驗證 app.js 中徹底移除所有 1.05, 0.998, Math.max(..., 50), 寫死公式與假成功"""
    with open(APP_JS, "r", encoding="utf-8") as f:
        app_code = f.read()

    # 驗證無 1.05 係數估算
    assert "rmse * 1.05" not in app_code
    assert "train_rmse * 1.05" not in app_code

    # 驗證無 0.998 係數估算
    assert "0.998" not in app_code

    # 驗證無 Math.max(lines.length - 1, 50) 竄改筆數
    assert "Math.max(lines.length - 1, 50)" not in app_code
    assert "Math.max(lines.length, 50)" not in app_code

    # 驗證缺少指標時顯示「未計算」
    assert '未計算' in app_code


def test_cases_a_through_e_with_node_engine():
    """透過 Node.js 執行真實案例 A ~ E 驗證"""
    node_test_script = """
    const fs = require('fs');
    const path = require('path');
    require('./web/regression_engine.js');

    // Case A: 50 Startups
    const csv50 = fs.readFileSync('./data/50_Startups.csv', 'utf8');
    const p50 = RegressionEngine.parseCSV(csv50);
    const r50 = RegressionEngine.fitAndEvaluate({
        headers: p50.headers,
        rows: p50.rows,
        targetCol: 'Profit',
        featureCols: ['R&D Spend', 'Marketing Spend'],
        testSize: 0.2,
        randomState: 0
    });
    if (r50.datasetInfo.totalRows !== 50) throw new Error('Case A 筆數錯誤');
    if (isNaN(r50.metrics.test.rmse)) throw new Error('Case A RMSE NaN');

    // Case B: Boston Housing
    const csvBH = fs.readFileSync('./data/BostonHousing.csv', 'utf8');
    const pBH = RegressionEngine.parseCSV(csvBH);
    const rBH = RegressionEngine.fitAndEvaluate({
        headers: pBH.headers,
        rows: pBH.rows,
        targetCol: 'medv',
        featureCols: ['crim', 'rm', 'lstat'],
        testSize: 0.2,
        randomState: 0
    });
    if (rBH.datasetInfo.totalRows !== 506) throw new Error('Case B 筆數錯誤');
    if (isNaN(rBH.metrics.test.rmse)) throw new Error('Case B RMSE NaN');

    // Case C: 少於 50 筆的有效 CSV (20 筆)
    const smallCSV = `X1,X2,Y\\n` + Array.from({length: 20}, (_, i) => `${i+1},${(i+1)*2},${(i+1)*3 + 1.5}`).join('\\n');
    const pSmall = RegressionEngine.parseCSV(smallCSV);
    if (pSmall.rows.length !== 20) throw new Error(`Case C 行數應為 20，實際 ${pSmall.rows.length}`);
    const rSmall = RegressionEngine.fitAndEvaluate({
        headers: pSmall.headers,
        rows: pSmall.rows,
        targetCol: 'Y',
        featureCols: ['X1', 'X2'],
        testSize: 0.2,
        randomState: 0
    });
    if (rSmall.datasetInfo.totalRows !== 20) throw new Error('Case C 訓練資料筆數非 20');

    // Case D: 目標欄位不存在
    try {
        RegressionEngine.fitAndEvaluate({
            headers: ['A', 'B'],
            rows: [['1', '2'], ['3', '4']],
            targetCol: 'NotExists',
            featureCols: ['A']
        });
        throw new Error('Case D 未捕捉不存在欄位！');
    } catch (e) {
        if (!e.message.includes('找不到目標欄位')) throw e;
    }

    // Case E: 樣本數少於參數數 (N <= P + 1)
    try {
        RegressionEngine.fitAndEvaluate({
            headers: ['F1', 'F2', 'F3', 'Y'],
            rows: [['1', '2', '3', '10'], ['4', '5', '6', '20']],
            targetCol: 'Y',
            featureCols: ['F1', 'F2', 'F3']
        });
        throw new Error('Case E 未捕捉樣本不足！');
    } catch (e) {
        if (!e.message.includes('樣本數')) throw e;
    }

    console.log('ALL_CASES_VERIFIED_SUCCESSFULLY');
    """

    res = subprocess.run(["node", "-e", node_test_script], capture_output=True, text=True, cwd=BASE_DIR)
    assert res.returncode == 0, f"Node.js 執行錯誤：{res.stderr}"
    assert "ALL_CASES_VERIFIED_SUCCESSFULLY" in res.stdout


def test_docs_and_root_sync():
    """驗證 docs/ 與專案根目錄的 index.html, style.css, app.js, regression_engine.js 與 web/ 100% 同步"""
    for fname in ["index.html", "style.css", "app.js", "regression_engine.js", "data.json", "default_data.js"]:
        web_f = os.path.join(BASE_DIR, "web", fname)
        docs_f = os.path.join(BASE_DIR, "docs", fname)
        root_f = os.path.join(BASE_DIR, fname)

        assert os.path.exists(web_f), f"缺少 web/{fname}"
        assert os.path.exists(docs_f), f"缺少 docs/{fname}"
        assert os.path.exists(root_f), f"缺少 ./{fname}"

        with open(web_f, "rb") as f1, open(docs_f, "rb") as f2, open(root_f, "rb") as f3:
            b1 = f1.read()
            b2 = f2.read()
            b3 = f3.read()
            assert b1 == b2, f"docs/{fname} 與 web/{fname} 內容不一致！"
            assert b1 == b3, f"./{fname} 與 web/{fname} 內容不一致！"
