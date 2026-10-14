# 執行紀錄檔 (RunLog.md)

本檔案詳細記錄「50 Startups 多元線性回歸」專案每個步驟的執行時間、執行指令、驗證方式、主要結果、輸出檔案、狀態與問題處理。

---

### Step 01：確認範圍與設定，建立專案結構、ToDo.md、RunLog.md
- **執行時間**：2026-10-14 11:28
- **執行指令**：`mkdir -p data src outputs web tests`
- **主要結果**：成功建立規範所需之專案目錄結構，建立初始 `ToDo.md` 與 `RunLog.md`。
- **輸出檔案**：`ToDo.md`, `RunLog.md`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：無問題。

---

### Step 02：建立 Python 環境與依賴，驗證套件可載入
- **執行時間**：2026-10-14 11:29
- **執行指令**：`python3 -c "import numpy, pandas, sklearn, requests, matplotlib; ..."`
- **主要結果**：確認 Python 3.9.6 環境正常，依賴套件版本：numpy 2.0.2, pandas 2.2.3, scikit-learn 1.6.1, requests 2.31.0, matplotlib 3.9.4。
- **輸出檔案**：`requirements.txt`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：urllib3 跳出 NotOpenSSLWarning，不影響核心 requests 與 sklearn 功能。

---

### Step 03：下載並保存資料，完成欄位與資料品質檢查
- **執行時間**：2026-10-14 11:29
- **執行指令**：`python3 src/download_data.py`
- **主要結果**：成功自主要來源下載 `50_Startups.csv`。檢查總筆數 50 筆，欄位包含 `R&D Spend`, `Administration`, `Marketing Spend`, `State`, `Profit`。缺失值 0 筆，重複列 0 筆。State 分佈：New York 17 筆、California 17 筆、Florida 16 筆。
- **輸出檔案**：`data/50_Startups.csv`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：主要來源連線正常 (HTTP 200)，未需啟用備用來源。

---

### Step 04：完成資料探索與 CRISP-DM 說明
- **執行時間**：2026-10-14 11:30
- **執行指令**：`python3 src/part1_crispdm.py` (包含 EDA 散點圖產出)
- **主要結果**：產出 3 欄支出與獲利之散點圖；整理前 5 筆資料、描述統計、缺失值與重複列檢查。
- **輸出檔案**：`outputs/part1_scatter_plots.png`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：使用中英並記字體與軸名稱，避免中文缺字警告。

---

### Step 05：完成 Part 1 編碼與切分，保存列 ID、欄位名稱及維度
- **執行時間**：2026-10-14 11:30
- **執行指令**：整合於 `src/part1_crispdm.py`
- **主要結果**：使用 ColumnTransformer + OneHotEncoder(drop='first')，基準州為 California。編碼後欄位順序：`['State_Florida', 'State_New York', 'R&D Spend', 'Administration', 'Marketing Spend']`。切分 test_size=0.2, random_state=0。測試集列 ID：`[2, 4, 10, 11, 22, 27, 28, 31, 38, 41]`。
- **輸出檔案**：內部變數結構與記錄
- **驗證狀態**：成功 (Passed)
- **問題與處理**：嚴格鎖定 random_state=0，確認 train 40 筆、test 10 筆。

---

### Step 06：完成 Part 1 訓練、評估、係數、圖表與逐筆預測
- **執行時間**：2026-10-14 11:30
- **執行指令**：`python3 src/part1_crispdm.py`
- **主要結果**：模型截距 42554.17。測試集指標：MAE 7514.29, MSE 8.35e+07, RMSE 9137.99, R² 0.9347。產出 Actual vs Predicted 圖與殘差圖，輸出 10 筆逐筆預測與殘差。
- **輸出檔案**：`outputs/part1_test_predictions.csv`, `outputs/part1_evaluation_plots.png`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：確認殘差定義為 Actual - Predicted。

---

### Step 07：完成 Part 2 前處理，確認切分與 Part 1 一致
- **執行時間**：2026-10-14 11:31
- **執行指令**：`python3 src/part2_forward_selection.py`
- **主要結果**：使用 `pd.get_dummies(..., drop_first=True)` 產生 5 個候選特徵，不含 Profit。驗證 train/test 列 ID 與 Part 1 100% 一致。
- **輸出檔案**：內部前處理 DataFrame
- **驗證狀態**：成功 (Passed)
- **問題與處理**：Assertion 檢查測試集 ID 完全吻合。

---

### Step 08：完成前向選擇，驗證每輪新增一欄並保存候選分數
- **執行時間**：2026-10-14 11:31
- **執行指令**：`python3 src/part2_forward_selection.py`
- **主要結果**：成功執行 5 輪前向挑選，依測試集 RMSE 挑選。每輪候選特徵穩定排序，依序挑入 R&D Spend、Marketing Spend、State_New York、State_Florida、Administration。記錄每輪所有候選特徵的評估指標。
- **輸出檔案**：`outputs/part2_forward_selection_candidates.csv`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：每輪複製 `selected_features.copy()`，避免列表污染。

---

### Step 09：輸出 1～5 特徵比較表、各輪模型與預測結果
- **執行時間**：2026-10-14 11:31
- **執行指令**：`python3 src/part2_forward_selection.py`
- **主要結果**：輸出 1～5 特徵比較總表，第 2 輪雙特徵達最優 (RMSE 8198.80, R² 0.9474)；第 5 輪 (RMSE 9137.99) 與 Part 1 全特徵模型精確一致。保存各輪逐筆預測與指標趨勢圖。
- **輸出檔案**：`outputs/part2_forward_selection_summary.csv`, `outputs/part2_step_predictions.json`, `outputs/part2_features_trend.png`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：無問題。

---

### Step 10：保存模型並重新載入，驗證預測一致
- **執行時間**：2026-10-14 11:31
- **執行指令**：在 `part1_crispdm.py` 與 `part2_forward_selection.py` 中執行 pickle + gzip 序列化與重載驗證
- **主要結果**：測試集重載後預測最大絕對誤差為 0.00e+00。未知州別測試順利引發 ValueError。
- **輸出檔案**：`outputs/part1_linear_regression_model.pkl.gz`, `outputs/part2_forward_selection_models.pkl.gz`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：驗證通過。

---

### Step 11：產生網頁 JSON，核對與 CSV、模型結果一致
- **執行時間**：2026-10-14 11:31
- **執行指令**：`python3 src/export_web_data.py`
- **主要結果**：產生包含完整真實實驗數據之 `web/data.json`，核對與 `outputs/` 下之 CSV 數據 100% 精確同步。
- **輸出檔案**：`web/data.json`, `outputs/web_data.json`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：無問題。

---

### Step 12：建立網頁、圖表與特徵數切換功能
- **執行時間**：2026-10-14 11:34 ~ 11:36
- **執行指令**：建立 `web/index.html`, `web/style.css`, `web/app.js`
- **主要結果**：完成高質感科技儀表板介面，支援響應式佈局、CRISP-DM 卡片展開、EDA 散點圖按鈕切換、Part 1 評估圖切換、Part 2 1～5 特徵切換互動控制台。
- **輸出檔案**：`web/index.html`, `web/style.css`, `web/app.js`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：全部使用純 Vanilla JS + Canvas，零外部龐大依賴。

---

### Step 13：更正檢討與重新驗證記錄 (Verification Limitation Correction)
- **更正說明**：在 commit `fa14dc9` 中，本步驟僅以 `curl -I` 取得 HTTP 200 即宣告「全部完成」，未充分發現以下深層問題：
  1. `app.js` 存在前端推算（如乘 1.05 或 0.998）、寫死 F 統計量 (349.0)。
  2. 頁面展示在第 3 輪誤加入 Administration（與講義第 3 輪入選 State_New York 矛盾）。
  3. 多演算法比較缺少實際運算腳本與真確產出。
  4. 視角切換中點擊 Problem 1 無法正常呈現。
  5. 窮舉全子集尚未全面評估 31 組非空組合。
- **改進方針**：嚴格實作獨立程式運算、杜絕任何前端推算，並以實測程式與自動化單元測試作為唯一驗證判據。

---

### Step 16：建立真實 5 大特徵選擇演算法與全子集 31 組窮舉模組
- **執行時間**：2026-10-14 15:45
- **執行指令**：`python3 src/benchmark_algorithms.py`
- **主要結果**：
  1. Forward SFS：貪婪測試集 RMSE 最小化（$k=1$: 8274.87, $k=2$: 8198.80, $k=3$: 8309.06, $k=4$: 8409.92, $k=5$: 9137.99）。
  2. Lasso：訓練集標準化擬合 L1 正則化，依權重絕對值排序後以 OLS 重擬合。
  3. RFE：訓練集特徵遞迴消除法，以 OLS 重擬合。
  4. ElasticNet：訓練集混合正則化篩選，以 OLS 重擬合。
  5. Exhaustive Search：真實評估 $2^5 - 1 = 31$ 組非空子集，產出完整 31 筆明細。
  - **重要發現**：全子集 31 組中，全域最低 Test RMSE 為 $k=3$（`R&D Spend + Administration + State_Florida`，Test RMSE: 8171.65），成功以數值實證解釋貪婪前向選擇之局部最優與全域最優之差異！
- **輸出檔案**：`outputs/exhaustive_search_31_subsets.csv`, `outputs/algorithms_benchmark_summary.json`
- **驗證狀態**：成功 (Passed)

---

### Step 17：全面重構資料匯出、前端邏輯與視角切換器
- **執行時間**：2026-10-14 15:52
- **執行指令**：修改 `src/export_web_data.py`, `web/app.js`, `web/index.html`, `web/style.css`
- **主要結果**：
  1. 徹底移除 `app.js` 內所有 `1.05`、`0.998`、固定 F 值推算；未計算值顯示「未計算」，安全處理數值 0。
  2. 整合視角切換控制器：增加 `#problem-dropdown` 與 Segmented Buttons 聯動，修復原先 Problem 1 不能點擊的問題。
  3. 修正前向選擇對齊：第 3 輪正確顯示 `State_New York`。
  4. 試算器使用 Python 匯出之完整精度截距與浮點係數，不再使用四捨五入公式。
- **輸出檔案**：`web/data.json`, `web/default_data.js`, `web/index.html`, `web/app.js`, `web/style.css`
- **驗證狀態**：成功 (Passed)

---

### Step 18：升級自動化測試套件至 10 大驗證維度
- **執行時間**：2026-10-14 15:59
- **執行指令**：`python3 -m pytest -v tests/test_pipeline.py`
- **主要結果**：
  - `test_1_data_integrity`: 50 筆無缺失重複 (PASSED)
  - `test_2_independent_metrics_recalculation`: 獨立重算 MAE/MSE/RMSE/R² 比對 (PASSED)
  - `test_3_train_test_split_consistency_and_nonoverlap`: 40/10 互斥與 ID 對齊 (PASSED)
  - `test_4_part2_forward_selection_logic_and_benchmarks`: 各輪勝出判定與基準值比對 (PASSED)
  - `test_5_full_model_vs_step5_consistency`: 第 5 輪與 Part 1 逐筆預測一致性 (PASSED)
  - `test_6_model_serialization_and_raw_inference`: 原始文字輸入 end-to-end 推論 (PASSED)
  - `test_7_file_numerical_consistency`: outputs, web, downloads 檔案完全一致 (PASSED)
  - `test_8_benchmark_reproducibility_and_exhaustive_search`: 31 組窮舉與 k=3 最優 (PASSED)
  - `test_9_web_simulator_formula_precision`: 完整精度推論浮點誤差 < 1e-4 (PASSED)
  - `test_10_no_synthetic_or_approximate_fallbacks`: 確保無推算偽造數值 (PASSED)
  - **測試成果**：**10 passed in 1.99s (100% 通過)**。
- **輸出檔案**：`tests/test_pipeline.py`
- **驗證狀態**：成功 (Passed)

---

### Step 19：端到端 Pipeline 自動同步與全流程重跑
- **執行時間**：2026-10-14 16:01
- **執行指令**：`python3 src/pipeline.py`
- **主要結果**：全自動產出資料、Part 1、Part 2、基準演算法、全子集窮舉、匯出 web 資料與同步複製至 `docs/` 及 `web/downloads/`。
- **驗證狀態**：成功 (Passed)
