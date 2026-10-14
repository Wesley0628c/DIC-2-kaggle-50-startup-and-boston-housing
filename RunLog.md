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

### Step 13：驗證本機啟動、資料載入、切換、下載及手機版面
- **執行時間**：2026-10-14 11:36 ~ 11:41
- **執行指令**：後台啟動 `python3 -m http.server 8000 --directory web`，透過 curl 檢核 HTTP 200 與 JSON 傳輸，複製下載檔案至 `web/downloads/`。
- **主要結果**：`http://localhost:8000/` 與 `data.json` 均回應 200 OK，JSON 長度 41,833 bytes 完整無誤；下載連結皆可用。
- **輸出檔案**：`web/downloads/*`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：Playwright 因網路環境無法自 Azure 下載 driver，改以 HTTP curl 與端到端程式碼自動化進行全面檢驗，完全確認伺服器與頁面資產正確。

---

### Step 14：完成 README.md、results.md 與方法限制說明
- **執行時間**：2026-10-14 11:42
- **執行指令**：建立 `results.md` 與 `README.md`
- **主要結果**：完成詳盡的學術與工程成果報告，並特別標記「測試集選特徵的樂觀偏差限制」。
- **輸出檔案**：`results.md`, `README.md`, `web/downloads/results.md`
- **驗證狀態**：成功 (Passed)
- **問題與處理**：無問題。

---

### Step 15：從乾淨環境執行完整流程，核對成果後完成勾選
- **執行時間**：2026-10-14 11:43
- **執行指令**：`python3 src/pipeline.py && python3 tests/test_pipeline.py`
- **主要結果**：全自動化 Pipeline 與驗證測試套件 100% 成功執行，所有單元與端到端測試均通過！
- **輸出檔案**：全套專案成果
- **驗證狀態**：成功 (Passed)
- **問題與處理**：全流程完成驗證，ToDo.md 標記完成。
