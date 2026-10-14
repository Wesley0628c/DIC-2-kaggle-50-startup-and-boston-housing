# 多元線性回歸課程要求重構與防洩漏完整驗收清單 (ToDo.md)

本清單依據多元線性回歸課程嚴格規範與最新需求建立，專注於徹底解決「測試集資料洩漏」、「特徵選擇誤用測試集」、「缺失值處理盲區（空字串誤判為 0）」以及「網站數據與說明一致性」等核心問題。每一步驟嚴格遵循「只有在執行並驗證成功後才能打勾」之原則。

---

## 執行分支宣告
- **當前工作分支**：`feature/ml-regression-refactor`
- **原則**：在獨立分支完成所有修改與測試；未另行授權前，不合併 main 亦不觸發正式網站部署。

---

## 核心任務清單 (Step 1 ~ Step 9)

- [x] **Step 1：修正資料切分與前處理**
  - [x] 確定嚴格處理順序：原始資料 $\rightarrow$ 處理無效目標值（移除缺失/非數值目標列並記錄筆數，絕不填補 $y$） $\rightarrow$ 切分訓練集 (80%)／測試集 (20%)（固定 random_state） $\rightarrow$ 僅從訓練資料 fit 前處理轉換器 $\rightarrow$ 訓練模型 $\rightarrow$ 評估。
  - [x] Python 端採用 `Pipeline` 與 `ColumnTransformer`，結合 `SimpleImputer(strategy='median')` 與 `OneHotEncoder(drop='first', handle_unknown='ignore')`。
  - [x] 規避虛擬變數陷阱 (Dummy Variable Trap，保留 $n-1$ 欄)，同時對測試集/未知類別具備容錯能力不崩潰。
  - [x] 在交叉驗證時，前處理必須在每個 training fold 內重新 fit，杜絕跨 fold 資料洩漏。
  - [x] 驗收：測試資料的數值或類別改變，不得改變已學得的訓練集前處理參數。

- [x] **Step 2：修正特徵選擇使用測試集的問題**
  - [x] 檢查並重構 `src/part2_forward_selection.py`、`src/benchmark_algorithms.py` 及 Notebook。
  - [x] **嚴格禁止使用最終測試集 RMSE 選擇特徵、特徵數 $k$、演算法或超參數**。
  - [x] 於訓練集內採用固定且一致的 5-fold cross-validation，每輪前向選擇以平均 validation RMSE 作為挑選依據，並由 CV 決定最佳特徵數 $k$（實測 $k=2$，CV RMSE $9474.32$）。
  - [x] Lasso、RFE、ElasticNet、全子集窮舉均遵守 CV fold 內特徵選擇原則。
  - [x] 明確區分「Lasso 直接預測」與「Lasso 選特徵後以 OLS 重擬合」。
  - [x] 特徵選擇與模型確定後，才在完整訓練集上重新 fit，並產生最終獨立測試集評估結果 ($8198.80$)。
  - [x] 原「測試集 RMSE 選特徵」版本移至獨立之歷史教學示範區，明確標示樂觀偏差 (Optimistic Bias) 與資料洩漏限制。

- [x] **Step 3：修正自訂 CSV 引擎 (regression_engine.js)**
  - [x] 嚴格區分空白字串 `""`、空白字元、`null`、`undefined`、`NaN` 與合法數值 `0`，禁止 `Number("")` 將空值誤判為 0。
  - [x] 數值缺失值使用訓練集有效數值之中位數填補，偶數筆資料之中位數正確計算中間兩值之平均數。
  - [x] 針對全缺失欄位、重複欄名、無有效特徵、樣本不足 ($n \le p+1$)、無效 `test_size` 提供精準錯誤訊息。
  - [x] $R^2$ 或 Adjusted $R^2$ 不適用（如目標無變異）時顯示「不適用」，不得偽造為 0 或 1。
  - [x] 匯出模型參數需包含特徵順序、類別對應、填補中位數、係數與截距。
  - [x] 撰寫包含缺失值、異常值的單元測試 (`test_js_engine.js` 與 `test_web_integration.py` 案例 A~E 全數通過)。

- [x] **Step 4：補上 Optuna 優化超參數**
  - [x] 保留 LinearRegression 作為基準模型，加入 Optuna 超參數調優（Ridge 回歸）。
  - [x] Optuna objective 採用訓練集內 5-fold CV RMSE，前處理與回歸模型包在同一個 Pipeline 內。
  - [x] 以對數尺度 (log scale, $10^{-4} \sim 10^4$) 搜尋正則化強度 $\alpha$。
  - [x] 固定 TPESampler seed (42)，執行 30 trials，記錄各 trial 軌跡、最佳參數 ($\alpha=0.1878$)、CV RMSE ($10034.58$) 及標準差。
  - [x] 誠實呈現基準模型與調優模型之比對（L2 微縮在全特徵微幅改善 CV +0.07%，但獨立測試集退步 -1.71%，遠不如稀疏特徵選擇 $k=2$）。

- [x] **Step 5：補齊圖表與課程說明**
  - [x] 產出固定種子隨機合成資料散佈圖（標註為教學示範：`part1_synthetic_demo_scatter.png`）。
  - [x] 產出真實特徵與目標散佈圖 (`real_data_scatters.png`)、Actual vs. Predicted 散佈圖 (`actual_vs_predicted.png`)、殘差圖 (`residuals_plot.png`)。
  - [x] 繪製特徵數對 Training RMSE / Cross-validation RMSE 變化曲線 (`part2_features_trend.png`)。
  - [x] 繪製學習曲線 (Learning Curve)：訓練資料筆數對 Training / Validation RMSE (`learning_curve.png`)。
  - [x] 補正課程統計概念白話說明：MAE/MSE/RMSE/R² 意義與單位、Train/Test 切分必要性、過擬合/欠擬合判讀、誤差/殘差常態性與等變異性假設、p-value 正確統計意義。

- [x] **Step 6：修正網站數據與文字一致性**
  - [x] Boston Housing 之回歸公式、係數表、解釋文字讀取同一份模型輸出（更新 RM +3.7087, NOX -16.1236, CHAS +2.3408, LSTAT -0.4928）。
  - [x] 特徵選擇的按鈕、表格、圖表與標籤依實際 CV 選取順序動態生成 (`renderKStepperButtons`)。
  - [x] 切換演算法時同步更新特徵標籤與最佳 $k$，無資料時顯示未計算。
  - [x] 清楚標示指標類型：Training, Cross-Validation, Test。
  - [x] 檢查並確認所有模型、CSV、JSON、Notebook 下載連結有效。

- [x] **Step 7：完成可重用 workflow 與部署**
  - [x] 將資料準備、訓練、評估、匯出抽成通用模組 `src/regression_workflow.py`，供 50 Startups 與 Boston Housing 共用。
  - [x] 使用 gzip + pickle 儲存可直接接受原始 DataFrame 輸入的完整 Pipeline 套件，驗證重載前後預測 100% 一致 (Max diff = 0.00e+00)。
  - [x] 建立合理的模型監控與維護計畫（缺失率、未知類別、真實標籤回饋機制），誠實標示現況與未來計畫。

- [x] **Step 8：同步 Notebook 與報告**
  - [x] 更新 `50_Startup_Multiple_Linear_Regression_CRISP_DM.ipynb`，包含 CRISP-DM 六階段、5-fold CV 特徵選擇、Lasso、Optuna、圖表與重載。
  - [x] 於乾淨真實 Kernel 完整執行 Notebook，保留真實 outputs 與圖表。
  - [x] 更新 `README.md`、`results.md`、`RunLog.md`，作業完成度表依實測填寫。

- [x] **Step 9：測試與最終交付**
  - [x] 執行並更新所有單元測試（更新原有斷言為驗證無洩漏 CV 方法與數值一致性，不寫死舊 k=2）。
  - [x] 全套 19 項測試 100% 通過 (`python3 -m pytest -v tests/`)。
  - [x] 完成完整檢驗與繁體中文最終成果交付報告。
