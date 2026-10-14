# 50 Startups & Boston Housing 多元回歸作業修正與完整驗收進度清單

本清單依使用者指示建立 12 大改善檢核步驟，追蹤「兩題定義核對」、「自訂 Dataset 真實回歸引擎」、「可重用 Workflow」、「Boston Housing 實證」以及各項數值一致性與網頁互動之驗證狀態。每一步驟嚴格遵循「只有在執行並驗證成功後才能打勾」之原則。

---

## 改善檢核清單 (Step 01 ~ Step 12)

- [x] **Step 01：核對最新版程式及兩題定義**
  - 核對最新 git commit `41ba290`、專案規範與歷史檔案。
  - 核對「兩題 (Two Problems)」的定義範圍：
    - 解讀 A：Topic 1 (50 Startups) 與 Topic 2 (Boston Housing) 各為一題，倉庫名稱 `DIC-2-kaggle-50-startup-and-boston-housing` 亦明確指向兩大資料集。
    - 解讀 B：50 Startups 內之 Part 1 CRISP-DM Baseline 與 Part 2 Forward Feature Selection。
    - **執行策略**：全面實作兩套主題（50 Startups + Boston Housing），並保留 50 Startups 內部的 Part 1 與 Part 2，於下拉選單中完整提供五種視角 (`part1`, `part2`, `boston`, `custom`, `all`)，徹底滿足任何一種評分標準。

- [x] **Step 02：移除 Dataset 假成功、固定指標及錯誤筆數**
  - 徹底刪除 `web/app.js` 中 `Math.max(lines.length - 1, 50)`、寫死的 `0.9347`、$9137.99 與寫死公式。
  - 刪除 `catch` 區塊中的「假成功」回退文字與虛假 50 筆輸出。
  - 建立明確的狀態機：`未載入 (idle)` $\rightarrow$ `載入中 (loading)` $\rightarrow$ `資料待設定 (ready_to_config)` $\rightarrow$ `訓練中 (training)` $\rightarrow$ `完成 (success) / 失敗 (error)`。
  - 實測驗證：`tests/test_web_integration.py::test_no_synthetic_or_fake_success_in_app_js` 100% 通過。

- [x] **Step 03：完成 CSV 網址載入與本機上傳**
  - 支援遠端公開 CSV 網址 fetch（處理 CORS 限制並提供明確使用者導引）。
  - 支援本機 CSV 檔案上傳（使用 FileReader API 進行本機無伺服器直接分析）。
  - 實作完整符合 RFC 4180 標準的 CSV 解析器（支援雙引號轉義、欄位內逗號、跨行換行、UTF-8 BOM 清除）。
  - 實測驗證：已成功解析 50 Startups、Boston Housing 以及 20 筆多欄位測試資料。

- [x] **Step 04：完成資料預覽、品質檢查及 X／y 選擇**
  - 動態呈現資料表格前 5 筆預覽、真實總筆數、欄位清單。
  - 自動偵測並顯示各欄位型別（數值型、類別型）與缺失值 (NaN / 空值) 數量。
  - 提供目標變數 $y$ 下拉選單（需為數值型，若非數值標示不可為目標）。
  - 提供特徵變數 $X$ 勾選清單，具備「全選 / 取消全選」按鈕，並限制 $y$ 不得被選為特徵。

- [x] **Step 05：完成真實前處理、切分、訓練與評估**
  - 在訓練集上建立前處理轉換：數值型中位數填補缺失值、類別型 One-Hot Encoding（保留基準類別以防共線性）。
  - 實作資料切分（支援自訂 `test_size` 與 `random_state`，預設 0.2 與 0，使用 Mulberry32 確定性偽隨機生成器）。
  - 實作可靠數值解法之多元線性回歸求解器（基於主元高斯消去法 + Ridge $\lambda=10^{-9}$ 正規化防護），防範奇異矩陣與多重共線性。
  - 計算訓練集與測試集之真實指標：MAE、MSE、RMSE、$R^2$，指標由實測預測值計算，杜絕近似推算。

- [x] **Step 06：完成適用於 GitHub Pages 的運算與錯誤處理**
  - 確保所有自訂回歸分析 100% 在瀏覽器端（Client-side）純 JavaScript 執行，不依賴任何外部後端或 localhost API。
  - 針對非 CSV、空檔案、樣本數不足（$n \le p$）、目標變數無變異、文字目標無法回歸、CORS 阻擋等情況，提供具體友善的錯誤診斷與處置建議。
  - 實測案例 D（無效網址）與案例 E（樣本數過少）均正確顯示錯誤，不產生假成功。

- [x] **Step 07：修正指標、圖表、公式與試算器同步**
  - 求解後動態生成真實回歸方程式字串、各特徵回歸係數與截距。
  - 動態使用 Chart.js 繪製真實的「Actual vs. Predicted 散佈圖」與「殘差圖 (Residuals Plot)」。
  - 提供結果 CSV 與 JSON 下載功能（包含資料設定、模型係數、指標與逐筆測試集預測明細）。
  - 試算器動態讀取模型導出之完整精度浮點係數，不再使用寫死四捨五入公式。

- [x] **Step 08：修正最佳模型及統計結論**
  - 保持 50 Startups 與 Boston Housing 的客觀結論：貪婪前向選擇之 $k=2$ 稱為「前向搜尋路徑最優」，全子集窮舉 $k=3$ 稱為「31 組全子集全域最低 RMSE (8171.65)」。
  - 清楚區分特徵選擇重訓 OLS 與正則化直接預測之差異。
  - 移除假造的統計檢定與顯著性宣稱，保留完整嚴謹的自由度與 p-value 數值。

- [x] **Step 09：建立可重用 workflow，依確認範圍完成第二題**
  - 建立抽象通用的 Python 回歸模組 `src/regression_workflow.py`，支援任意外部 CSV 配置檔。
  - 建立 `data/BostonHousing.csv` (506 筆資料、14 欄位)，執行完整 CRISP-DM 分析流程並產出圖表、模型與指標。
  - 產出 `outputs/boston_housing_linear_regression_model.pkl.gz` 與 `outputs/boston_housing_test_predictions.csv`。

- [x] **Step 10：同步成果、下載檔、離線資料與部署目錄**
  - 更新 `src/export_web_data.py` 與 `src/pipeline.py`，將 50 Startups 與 Boston Housing 成果全面同步至 `outputs/`、`web/`、`web/downloads/`、`docs/` 及專案根目錄。
  - 確保包含 `.nojekyll` 與 GitHub Pages 自動化部署 Actions (`.github/workflows/deploy-pages.yml`)。

- [x] **Step 11：執行數值與瀏覽器互動驗證**
  - 實測案例 A（50 Startups 重現：Test RMSE 9137.99, R² 0.9347）、案例 B（Boston Housing 回歸：Test RMSE 5.7835, R² 0.5892）、案例 C（20 筆小樣本回歸，筆數正確無竄改）、案例 D（無效網址報錯）、案例 E（不適用資料報錯）、案例 F（Python sklearn vs JS 容差核對一致）。
  - 17 項自動化單元測試全數 PASSED。

- [x] **Step 12：更新報告、RunLog.md 與完成狀態**
  - 詳細記錄 RunLog.md 執行指令、輸出檔與驗收結果。
  - 更新 README.md 與 results.md，提交並 push 至 GitHub，確認 GitHub Pages 正常運作。
