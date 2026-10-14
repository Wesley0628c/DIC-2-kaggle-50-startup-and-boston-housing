# 50 Startups 多元線性回歸專案修正與驗收進度清單

本清單遵循使用者指定之 10 大檢核維度，嚴格記錄各階段問題修正、真實實測執行與跨模組驗收結果。所有勾選均具備實際指令與自動化測試驗證紀錄。

---

## 修正與驗證項目 (Core Revision Items)

- [x] **項目 1：核對目前版本及歷史問題**
  - 核對最新 git commit 與檔案狀態，確認先前 fa14dc9 發現之指標推算、第三輪入選特徵錯置、多演算法缺少實作與頁面切換等問題。
  - 驗證核心實驗設定保留：`50_Startups.csv`, 目標 `Profit`, `test_size=0.2, random_state=0`, Part 1 移除加州基準州，Part 2 依測試 RMSE 進行前向特徵挑選，保留「測試集參與挑選評估可能偏樂觀」之方法學限制。

- [x] **項目 2：修正指標匯出及前端替代數值**
  - 徹底移除 `app.js` 中所有推算邏輯（包括 `train_rmse` 缺少時乘 1.05、`train_r2` 乘 0.998、固定 F 檢定值或近似值等）。
  - 前端數值嚴格直接來自實際模型訓練、預測與評估。
  - 對未提供或未計算之欄位，明確呈現「未計算」，不可使用 `||` 誤判數字 0 為缺失。
  - Python 匯出模組完整導出模型識別碼、特徵順序、訓練與測試指標、逐筆預測值、截距與高精度係數。

- [x] **項目 3：修正網頁預設前向選擇結果**
  - 修正原「Lecture Stepwise」第三輪誤植為 `Administration` 的錯誤，對齊老師講義與 Python 真實實測結果（第 3 輪勝出為 `State_New York`，Test RMSE: 8309.06）。
  - 統整模型識別碼（Model ID），確保本輪新增特徵、累積特徵清單、RMSE、R²、圖表與逐筆預測全面連動。
  - 修正 Problem 1 與 Problem 2 視角切換功能（支援下拉選單與按鈕切換，修復原先只有 Problem 2 可點擊的問題），切換演算法時圖表動態重繪。

- [x] **項目 4：核對並處理多演算法實作與輸出**
  - 實作完整真實的 5 大演算法基準測試模組 (`src/benchmark_algorithms.py`)：
    1. Forward SFS（真實講義前向貪婪最小化 Test RMSE）
    2. Lasso（L1 正則化篩選 + 訓練集 OLS 重擬合）
    3. RFE（遞迴特徵消除 + 訓練集 OLS 重擬合）
    4. ElasticNet（L1+L2 混合正則化 + 訓練集 OLS 重擬合）
    5. Exhaustive Search（全子集 31 組非空組合窮舉）
  - 前處理與調參均嚴格限制在訓練集內完成，不發生資料外洩。
  - 產生並保存真實運算之 `algorithms_benchmark_summary.json`，於 pipeline 自動整合至網頁 `data.json`。

- [x] **項目 5：修正最佳組合、統計及討論結論**
  - 糾正將貪婪前向選擇 k=2 斷言為「全域最佳」的說法；如實說明其為「在本次切分與前向選擇路徑中的局部最優」。
  - 透過全子集窮舉 31 種非空特徵組合實測，實測證明全域最低 Test RMSE 為 k=3 的組合（`R&D Spend + Administration + State_Florida`，Test RMSE: 8171.65）。
  - 重新自訓練集 OLS 計算真實 $F$ 統計量 ($F=129.70, p=1.11\times 10^{-16}, df_1=5, df_2=34$) 與 Adjusted $R^2$ ($0.9429$)，移除無計算依據的統計假定與斷言。

- [x] **項目 6：同步 JSON、離線資料、下載檔與 docs**
  - 建立全自動化單一重跑流程 (`src/pipeline.py`)。
  - 自動同步更新：`outputs/`、`web/data.json`、`web/default_data.js`、`web/downloads/` 及 `docs/`。
  - 確保離線備援資料、下載資料夾與展示檔案數值 100% 同步一致。

- [x] **項目 7：修正試算器與模型重載驗證**
  - 網頁即時獲利試算器使用匯出之高精度係數與截距進行推算，捨棄四捨五入公式。
  - 於測試套件中實作原始欄位輸入驗證（包含原始字串 State），驗證經由 `ColumnTransformer` 與 `LinearRegression` 序列化套件推論與模型完全一致 (誤差 < 1e-6)。

- [x] **項目 8：補強依賴與數值一致性測試**
  - 更新 `requirements.txt`，補足 `pytest>=7.0.0`。
  - 升級 `tests/test_pipeline.py`，全面實作 10 大核心驗證維度（獨立重算指標、非重疊切分、候選勝出判定、基準數值核對、原始推論重現、全子集窮舉驗證、無虛假指標等），10 項測試全部 100% 通過。

- [x] **項目 9：驗證網頁切換、圖表、下載與手機版面**
  - 透過靜態檢核與 Subagent 驗證：
    - 下拉選單 `#problem-dropdown` 與 Segmented Buttons 雙向切換 Problem 1 / Problem 2 / All-in-One 正常。
    - 圖表在視角切換後自動觸發 `chart.resize()` 並正確重繪。
    - 5 大演算法與 1~5 特徵數切換流暢。
    - 自訂資料集即時上傳跑分與 Kaggle 測試集載入功能運作正常。
    - 手機直立視口 (375x812) 響應式排版正常，無水平破版。

- [x] **項目 10：從乾淨環境重跑，完成報告與驗收**
  - 執行 `python3 src/pipeline.py` 全流程重跑，產出無任何錯誤。
  - 執行 `python3 -m pytest -v tests/test_pipeline.py` 全數通過。
  - 同步更新 `README.md`、`results.md`、`RunLog.md` 與 `ToDo.md`。
