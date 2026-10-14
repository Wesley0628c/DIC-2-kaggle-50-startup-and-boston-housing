# 50 Startups 多元線性回歸分析與特徵選擇實測報告

## 專案執行摘要 (Executive Summary)
本專案針對「50 Startups」資料集，完整執行 **Part 1：CRISP-DM 跨行業資料探勘標準流程之多元線性回歸** 與 **Part 2：依課堂講義規範自建的前向特徵選擇演算法 (Forward Feature Selection)**。全流程以純 Python 實作，資料切分固定為 80% 訓練集 (40 筆) 與 20% 測試集 (10 筆)，隨機種子鎖定 `random_state=0`，確保全實驗 100% 可重現。

### 核心實驗成果摘要
1. **Part 1 基準模型 (Baseline Model，全 5 個編碼特徵)**：
   - 截距 $b_0 = 42554.17$
   - 預測公式：
     $$\text{Profit} = 42554.17 - 959.28 \times \text{State\_Florida} + 699.37 \times \text{State\_New York} + 0.7735 \times \text{R\&D Spend} + 0.0329 \times \text{Administration} + 0.0366 \times \text{Marketing Spend}$$
   - 測試集指標：**$\text{RMSE} = 9137.99$**，**$R^2 = 0.9347$**，**$\text{MAE} = 7514.29$**。
2. **Part 2 前向特徵選擇 (Forward Selection)**：
   - 第 1 輪入選：`R&D Spend`（測試 RMSE: 8274.87, $R^2$: 0.9465）
   - 第 2 輪入選：`Marketing Spend`（測試 **RMSE: 8198.80**，**$R^2$: 0.9474**）—— **前向選擇路徑中最優模型！**
   - 第 3 輪入選：`State_New York`（測試 RMSE: 8309.06, $R^2$: 0.9460）
   - 第 4 輪入選：`State_Florida`（測試 RMSE: 8409.92, $R^2$: 0.9447）
   - 第 5 輪入選：`Administration`（測試 RMSE: 9137.99, $R^2$: 0.9347）—— **完全契合 Part 1 全特徵基準模型**。
3. **全子集 31 組非空組合窮舉 (Exhaustive Search)**：
   - 實測評估全部 $2^5 - 1 = 31$ 組非空特徵組合。
   - **全域最低測試 RMSE**：$k=3$ 之組合【`R&D Spend + Administration + State_Florida`】（測試集 RMSE 為 **$8171.65**）。
   - **重要方法學結論**：前向特徵選擇得到的 $k=2$ 最優（RMSE: 8198.80），是「在本次切分與前向選擇路徑中」的局部最優，而非所有可能特徵子集之全域最優。貪婪演算法因在第 2 輪選入 Marketing Spend，後續路徑無法再回頭檢驗包含 Administration 與 State_Florida 的最優 3 特徵組合。

---

## 一、CRISP-DM 六階段完整流程剖析 (Part 1)

### A. Business Understanding (商業理解)
- **研究目標**：建立多元線性回歸模型，預測企業獲利能力 (Profit)，協助創投基金 (Venture Capital) 與初創企業管理階層評估資金配置之投資報酬率。
- **變數定義**：
  - 自變數 $X$（特徵）：研發支出 (`R&D Spend`)、行政支出 (`Administration`)、行銷支出 (`Marketing Spend`)、所在州 (`State`)。
  - 應變數 $y$（目標）：企業獲利 (`Profit`)。
- **統計學界線**：線性回歸呈現的是特徵與利潤間之**統計相依性 (Statistical Association)**，不能直接等同因果關係 (Causality)。

### B. Data Understanding (資料理解與品質檢驗)
- **資料規模**：50 筆觀察值 (Rows)、5 個原始欄位 (Columns)。
- **品質檢核**：
  - 缺失值 (Missing Values)：0 筆，資料完整度 100%。
  - 重複列 (Duplicates)：0 筆。
  - 地理分佈：紐約州 (New York) 17 筆、加州 (California) 17 筆、佛羅里達州 (Florida) 16 筆。
  - 零支出情況：包含 2 筆零研發支出與 3 筆零行銷支出。依商業邏輯，早期初創公司處於純概念驗證或無行銷預算期為正常商業現象，依規範不擅自刪除。

### C. Data Preparation (資料準備與虛擬變數陷阱)
- **虛擬變數編碼 (One-Hot Encoding)**：
  - 原始類別變數 `State` 共有 3 個類別：`['California', 'Florida', 'New York']`。
  - 依照講義設定，採用 `drop_first=True`（或 `OneHotEncoder(drop='first')`），以字母排序首位的 **California** 作為基準州 (Reference State)。
  - 編碼後產生 2 個虛擬欄位：`State_Florida` 與 `State_New York`。
  - **特徵維度**：原始 4 個特徵轉換為 **5 個模型輸入特徵**。
- **統計學核心觀念：虛擬變數陷阱 (Dummy Variable Trap)**：
  - 若保留所有 3 個州別虛擬欄位，則 $\text{State\_CA} + \text{State\_FL} + \text{State\_NY} = 1$。
  - 此等式與模型截距項常數向量 1 構成**完全多重共線性 (Perfect Multicollinearity)**，會導致矩陣 $(X^T X)$ 不可逆而無法估計回歸係數。
  - **重要提醒**：剃除一欄是為了解決虛擬變數與截距間的完全共線性，**並非消除所有特徵之間的多重共線性**（例如研發與行銷支出之間的相關性仍然存在）。
- **資料切分**：`test_size=0.2, random_state=0`。
  - 訓練集：40 筆；測試集：10 筆（原始列索引：`[2, 4, 10, 11, 22, 27, 28, 31, 38, 41]`）。

### D. Modeling (多元線性回歸建模與訓練集統計推論)
- **估計方法**：常規最小平方法 (Ordinary Least Squares, OLS)，包含截距項。
- **訓練集 OLS 參數統計檢定表 ($n=40, p=5, df_{\text{residuals}}=34$)**：
  | 變數 (Variable) | 係數估計值 ($\hat{\beta}$) | 標準誤 (Std. Error) | t 統計量 ($t$-stat) | p 值 ($p$-value) | 統計顯著性 ($\alpha=0.05$) |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **截距項 (Intercept)** | 42554.1676 | 8358.5382 | 5.0911 | $1.31 \times 10^{-5}$ | 極顯著 (***) |
  | **R&D Spend** | +0.7735 | 0.0551 | 14.0298 | $1.11 \times 10^{-15}$ | 極顯著 (***) |
  | **Administration** | +0.0329 | 0.0665 | 0.4945 | 0.6241 | 不顯著 ($p > 0.05$) |
  | **Marketing Spend** | +0.0366 | 0.0194 | 1.8841 | 0.0681 | 邊際顯著 ($p < 0.10$) |
  | **State_Florida** | -959.2842 | 4038.1077 | -0.2376 | 0.8136 | 不顯著 ($p > 0.05$) |
  | **State_New York** | +699.3691 | 3661.5627 | 0.1910 | 0.8497 | 不顯著 ($p > 0.05$) |

- **訓練集模型整體統計量**：
  - 樣本數：$n = 40$，參數個數：$k = 5$（不含截距），殘差自由度：$df = 34$。
  - **決定係數 $R^2$**：$0.9502$
  - **校正決定係數 Adjusted $R^2$**：$1 - \frac{(1 - R^2)(n - 1)}{n - p - 1} = 0.9429$
  - **整體 F 檢定統計量**：$F = 129.70$，對應 $p\text{-value} = 1.11 \times 10^{-16}$（拒絕虛無假設，回歸方程式整體極顯著）。

### E. Evaluation (模型評估指標)
- **評估指標比較**：
  | 指標 (Metric) | 訓練集 (Train, 40 筆) | 測試集 (Test, 10 筆) | 公式與白話意義 |
  | :--- | :--- | :--- | :--- |
  | **MAE** | 6341.5358 | 7514.2937 | 平均絕對誤差：預測值與實際值差距之絕對值平均。 |
  | **MSE** | 81571001.8008 | 83502864.0326 | 均方誤差：殘差平方平均，對大誤差懲罰重。 |
  | **RMSE** | 9031.6666 | 9137.9902 | 均方根誤差：$\sqrt{\text{MSE}}$，單位與利潤相同（美元）。 |
  | **$R^2$** | 0.9502 | 0.9347 | 判定係數：模型解釋獲利變異量的比例（**不得稱為準確率**）。 |

- **逐筆測試預測明細 (Test Predictions Breakdown)**：
  | 原始 ID | 所在州 | 研發支出 | 行政支出 | 行銷支出 | 實際利潤 ($) | 預測利潤 ($) | 殘差 ($) |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | #2 | Florida | 153441.51 | 101145.55 | 407934.54 | 191050.39 | 178537.48 | +12512.91 |
  | #4 | Florida | 142107.34 | 91391.77 | 366168.42 | 166187.94 | 171695.53 | -5507.59 |
  | #10 | Florida | 101913.08 | 110594.11 | 229160.95 | 146121.95 | 133959.08 | +12162.87 |
  | #11 | California | 100671.96 | 91790.61 | 249744.55 | 144259.40 | 132646.68 | +11612.72 |
  | #22 | Florida | 73994.56 | 122782.75 | 303319.26 | 110352.25 | 113943.43 | -3591.18 |
  | #27 | New York | 72107.60 | 127864.55 | 353183.81 | 105008.31 | 115682.02 | -10673.71 |
  | #28 | Florida | 66051.52 | 182645.56 | 118148.20 | 103282.38 | 100984.77 | +2297.61 |
  | #31 | New York | 61136.38 | 152701.92 | 88218.23 | 97483.56 | 97411.39 | +72.17 |
  | #38 | New York | 20229.59 | 65947.93 | 185265.10 | 81229.06 | 67104.99 | +14124.07 |
  | #41 | Florida | 27892.92 | 84710.77 | 164470.71 | 77798.83 | 74883.35 | +2915.48 |

### F. Deployment (模型保存與重載相容性)
- 使用 `pickle` + `gzip` 壓縮序列化保存至 `outputs/part1_linear_regression_model.pkl.gz`。
- 儲存內容包含：`ColumnTransformer`、特徵順序、基準州、係數、訓練組態與合法州別名單。
- **端到端原始欄位推論驗證**：以包含原始字串 State 的原始 DataFrame 輸入，經管線推論之預測值與儲存值最大誤差 $< 10^{-6}$，完全一致。

---

## 二、自建 Forward Selection (前向特徵選擇) 實測剖析 (Part 2)

### 演算法選擇邏輯
- 依講義規定，從空特徵集合開始，每次選入一個候選特徵使得**測試集 RMSE 最小**。
- 候選特徵共 5 個：`['Administration', 'Marketing Spend', 'R&D Spend', 'State_Florida', 'State_New York']`。

### 1～5 個特徵漸進比較總表 (Progression Comparison Table)
| 輪數 (特徵數) | 本輪新增入選特徵 | 累積特徵子集 | 測試集 RMSE ($) | 測試集 $R^2$ | 訓練集 RMSE ($) | 訓練集 $R^2$ | 訓練集 F 統計量 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | `R&D Spend` | R&D | 8274.87 | 0.9465 | 9493.58 | 0.9450 | 652.39 |
| **2** | `Marketing Spend` | R&D + Marketing | **8198.80** | **0.9474** | 9078.62 | 0.9497 | 349.04 |
| **3** | `State_New York` | R&D + Marketing + NY | 8309.06 | 0.9460 | 9072.39 | 0.9498 | 227.35 |
| **4** | `State_Florida` | R&D + Marketing + NY + FL | 8409.92 | 0.9447 | 9069.94 | 0.9498 | 166.08 |
| **5** | `Administration` | 全部 5 特徵 | 9137.99 | 0.9347 | 9031.67 | 0.9502 | 129.70 |

---

## 三、五大特徵選擇演算法真實基準測試與 31 組全子集窮舉 (Section IV & VI)

為了消除過去報告中僅有一種前向演算法卻列出多個重複名稱的問題，本專案完整實作並評估了 5 種特徵選擇演算法：

### 1. 五大演算法之橫向比較 (All 5 Algorithms Benchmark)
| 演算法 (Method) | 前處理與挑選依據 | 最優特徵數 $k$ | 最優特徵子集 | 測試集 RMSE ($) | 測試集 $R^2$ |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Forward Selection (SFS)** | 貪婪依測試集 RMSE 最小挑選 | $k=2$ | R&D Spend + Marketing Spend | 8198.80 | 0.9474 |
| **Lasso (L1 正則化)** | 僅訓練集標準化擬合 L1 權重絕對值排序，再以 OLS 重擬合 | $k=2$ | R&D Spend + Marketing Spend | 8198.80 | 0.9474 |
| **RFE (遞迴消除)** | 僅訓練集標準化特徵以 OLS 逐輪反向剔除，再以 OLS 重擬合 | $k=2$ | R&D Spend + Marketing Spend | 8198.80 | 0.9474 |
| **ElasticNet (L1+L2)** | 僅訓練集混合正則化權重排序，再以 OLS 重擬合 | $k=2$ | R&D Spend + Marketing Spend | 8198.80 | 0.9474 |
| **Exhaustive Search (窮舉)** | 遍歷全部 $2^5 - 1 = 31$ 組非空特徵子集，於相同切分評估 | **$k=3$ (全域最優)** | **R&D + Administration + State_Florida** | **8171.65** | **0.9477** |

### 2. 全子集 31 組窮舉之關鍵發現
- **各特徵數最低測試 RMSE 組合**：
  - $k=1$：`['R&D Spend']`，RMSE = 8274.87
  - $k=2$：`['R&D Spend', 'Marketing Spend']`，RMSE = 8198.80
  - **$k=3$ (全域最低)**：`['R&D Spend', 'Administration', 'State_Florida']`，**RMSE = 8171.65**
  - $k=4$：`['R&D Spend', 'Administration', 'State_Florida', 'State_New York']`，RMSE = 8178.37
  - $k=5$：`['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']`，RMSE = 9137.99
- **統計與方法學警示**：
  - 前向選擇在 $k=2$ 取得最低 RMSE 8198.80，僅能宣稱**「在本次切分與前向選擇貪婪路徑中表現最佳」**。
  - 不能直接斷言 $k=2$ 是所有特徵組合的全域最佳，因為貪婪演算法選定 Marketing 後即永久鎖定，無法探測到包含 Administration 與 State_Florida 的 $k=3$ 組合。

---

## 四、方法限制與教學嚴謹性聲明 (Important Methodology Limitation)

> [!WARNING]
> **本實作忠實再現課堂講義流程，以『測試集 (Test Set)』的表現作為前向特徵挑選的判定依據。**
> 在機器學習的嚴格方法論中：
> 1. 當測試集參與了特徵選拔 (Model Selection)，測試集已不再具備「完全獨立性」，其產生的評估指標會偏向樂觀 (Optimistic Bias)。
> 2. 嚴謹的工程規範應採用 **訓練集內部之 K-Fold 交叉驗證 (Cross-Validation)** 或額外切分 **驗證集 (Validation Set)** 進行特徵篩選，最後才由完全未參與選擇的測試集進行最終評估。
> 3. 單次切分測試集樣本數較小 ($n=10$)，誤差隨特徵數增加可能受少數樣本波動影響，不可草率斷定特徵純屬雜訊或絕對過擬合。

---

## 五、主題二：Boston Housing 波士頓房價多元線性回歸實證 (Topic 2)

依據作業題目「do two problems but choose one of the topic to upload」之規範與倉庫名稱 `DIC-2-kaggle-50-startup-and-boston-housing`，本專案以通用 CRISP-DM 工作流模組 (`src/regression_workflow.py`) 完整實作 Boston Housing 資料集之多元線性回歸：

### 1. 資料特性與設定
- **資料來源**：`data/BostonHousing.csv`
- **資料規模**：506 筆觀察值 (Rows)、14 個欄位 (13 個特徵 + 1 個目標)
- **目標變數 $y$**：`medv`（自住房屋價格中位數，單位：千美元）
- **切分設定**：`test_size=0.2, random_state=0`（訓練集 404 筆、測試集 102 筆）

### 2. 實測模型評估指標
| 評估維度 | 測試集 (Test Set) | 訓練集 (Train Set) |
| :--- | :---: | :---: |
| **樣本數量** | 102 筆 | 404 筆 |
| **RMSE (均方根誤差)** | **$5.7835** ($5,783.50) | $4.3962 |
| **MAE (平均絕對誤差)** | **$3.8429** ($3,842.90) | $3.1032 |
| **MSE (均方誤差)** | **33.4490** | 19.3265 |
| **$R^2$ (判定係數)** | **0.5892** (58.9%) | 0.7730 (77.3%) |
| **Adjusted $R^2$** | 0.5284 | **0.7291** |

### 3. 特徵經濟與環境學詮釋
- **房間數 (RM)**：係數約 **+3.8099**，代表每增加一間房，房價平均上升約 $3,810 美元（顯著正向影響）。
- **氮氧化物濃度 (NOX)**：係數約 **-17.7666**，空氣污染顯著折抑房價。
- **查爾斯河畔 (CHAS)**：係數約 **+2.6867**，臨河景觀享有顯著房價溢價。
- **低收入人口比例 (LSTAT)**：係數約 **-0.5249**，周邊弱勢人口比率與房價呈顯著負相關。

---

## 六、純瀏覽器端自訂資料集回歸實驗室 (Client-side ML Engine)

針對 GitHub Pages 靜態站點環境，本專案自主研發純 JavaScript 多元線性回歸引擎 (`web/regression_engine.js`)，徹底根除偽造成功與寫死指標：
1. **RFC 4180 標準 CSV 解析**：完整處理引號包裹、引號轉義、跨行儲存格、欄位內逗號與 UTF-8 BOM。
2. **自動型別與缺失值品質分析**：自動識別數值欄位與類別欄位，統計空值並以中位數自動填補。
3. **安全變數配置**：
   - 目標變數 $y$ 動態下拉（非數值欄位禁用）。
   - 特徵變數 $X$ 網格勾選，嚴格限制 $y$ 不可進入 $X$。
   - 支援自訂 `test_size` 與 `random_state`（Mulberry32 確定性演算法）。
4. **數值穩定 OLS 求解器**：採用主元高斯消去法，並內建 Ridge $\lambda=10^{-9}$ 正則化防護，徹底杜絕奇異矩陣與崩潰。
5. **動態視覺化與成果匯出**：即時生成 Actual vs. Predicted 散佈圖與殘差圖，支援下載測試集逐筆預測 CSV 與模型設定 JSON。

---

## 七、全專案自動化測試驗證彙整 (17 項測試全數通過)

執行指令：`python3 -m pytest -v tests/`
- `tests/test_pipeline.py` (13 項測試通過)：
  - `test_1_data_integrity`: 50 筆無缺失無重複。
  - `test_2_independent_metrics_recalculation`: 獨立重算 Part 1 MAE/MSE/RMSE/R² 完全一致。
  - `test_3_train_test_split_consistency_and_nonoverlap`: 40/10 互斥與原始列 ID 對齊。
  - `test_4_part2_forward_selection_logic_and_benchmarks`: 前向選擇每輪勝出邏輯與標準數值。
  - `test_5_full_model_vs_step5_consistency`: Part 2 輪 5 與 Part 1 全特徵預測一致。
  - `test_6_model_serialization_and_raw_inference`: 原始文字 State 輸入端到端重載推論差異 0。
  - `test_7_file_numerical_consistency`: outputs, web, downloads 檔案完全同步。
  - `test_8_benchmark_reproducibility_and_exhaustive_search`: 31 組窮舉全域最優為 k=3 (8171.65)。
  - `test_9_web_simulator_formula_precision`: 完整精度係數試算誤差 < 1e-4。
  - `test_10_no_synthetic_or_approximate_fallbacks`: 確保無推算偽造數值。
  - `test_11_boston_housing_workflow`: Boston Housing 506 筆資料、模型指標與存檔重載。
  - `test_12_python_vs_js_regression_engine_parity`: Python 與 JS 回歸引擎數值比對一致。
  - `test_13_small_dataset_row_count_correctness`: 20 筆小樣本回歸無 Math.max(n, 50) 竄改。
- `tests/test_web_integration.py` (4 項測試通過)：
  - `test_html_dom_elements_presence`: 驗證 HTML DOM 節點、五大視角下拉切換完整性。
  - `test_no_synthetic_or_fake_success_in_app_js`: 驗證 app.js 杜絕 1.05/0.998/Math.max。
  - `test_cases_a_through_e_with_node_engine`: 案例 A~E Node.js 執行驗證（含無效網址與樣本不足）。
  - `test_docs_and_root_sync`: web/docs/root 三處靜態檔案 100% 同步。

**全數測試結果：17 passed in 2.01s (100% PASSED)**。

