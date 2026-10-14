# 50 Startups 多元線性回歸分析與前向特徵選擇 (Forward Selection) 實測報告

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
   - 第 2 輪入選：`Marketing Spend`（測試 **RMSE: 8198.80**，**$R^2$: 0.9474**）—— **全場表現最優模型！**
   - 第 3 輪入選：`State_New York`（測試 RMSE: 8309.06, $R^2$: 0.9460）
   - 第 4 輪入選：`State_Florida`（測試 RMSE: 8409.92, $R^2$: 0.9447）
   - 第 5 輪入選：`Administration`（測試 RMSE: 9137.99, $R^2$: 0.9347）—— **完全契合 Part 1 全特徵基準模型**。

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

### D. Modeling (多元線性回歸建模)
- **估計方法**：常規最小平方法 (Ordinary Least Squares, OLS)，包含截距項。
- **迴歸係數表**：
  | 特徵欄位 (Feature) | 係數 (Coefficient) | 經濟意涵解讀 |
  | :--- | :--- | :--- |
  | 截距 ($b_0$) | 42554.1676 | 基準狀態（加州、三項支出均為 0）時之基礎獲利預期 |
  | `State_Florida` | -959.2842 | 相對加州，位於佛州預期獲利微幅減少約 $959 美元 |
  | `State_New York` | +699.3691 | 相對加州，位於紐約預期獲利微幅增加約 $699 美元 |
  | `R&D Spend` | +0.7735 | 研發支出每增加 1 美元，獲利預期顯著增加約 0.77 美元 |
  | `Administration` | +0.0329 | 行政支出每增加 1 美元，獲利微幅增加 0.03 美元（邊際效益極低） |
  | `Marketing Spend` | +0.0366 | 行銷支出每增加 1 美元，獲利預期增加 0.04 美元 |

### E. Evaluation (模型評估指標)
- **評估指標比較**：
  | 指標 (Metric) | 訓練集 (Train, 40 筆) | 測試集 (Test, 10 筆) | 公式與白話意義 |
  | :--- | :--- | :--- | :--- |
  | **MAE** | 6341.5358 | 7514.2937 | 平均絕對誤差：預測值與實際值差距之絕對值平均。 |
  | **MSE** | 81571001.8008 | 83502864.0326 | 均方誤差：殘差平方平均，對大誤差懲罰重。 |
  | **RMSE** | 9031.6666 | 9137.9902 | 均方根誤差：$\sqrt{\text{MSE}}$，單位與利潤相同（美元）。 |
  | **$R^2$** | 0.9502 | 0.9347 | 判定係數：模型解釋獲利變異量的比例（**不得稱為準確率**）。 |

- **過度擬合 (Overfitting) 討論**：
  - 訓練集 $R^2 = 0.9502$，測試集 $R^2 = 0.9347$，兩者差距僅約 0.0155；訓練集 RMSE 為 9031.67，測試集 RMSE 為 9137.99。
  - 模型在測試集表現平穩，但因單次切分測試樣本僅 10 筆，統計代表性有限，不得單憑一次切分即斷言模型絕對無過擬合。

- **逐筆測試預測明細 (Test Predictions Breakdown)**：
  | 原始 ID | 所在州 | 研發支出 | 行政支出 | 行銷支出 | 實際利潤 ($) | 預測利潤 ($) | 殘差 ($) |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | #2 | Florida | 153441.51 | 101145.55 | 407934.54 | 191050.39 | 182151.06 | +8899.33 |
  | #4 | Florida | 142107.34 | 91391.77 | 366168.42 | 166187.94 | 171540.57 | -5352.63 |
  | #10 | Florida | 101913.08 | 110594.11 | 229160.95 | 146121.95 | 134747.88 | +11374.07 |
  | #11 | California | 100671.96 | 91790.61 | 249744.55 | 144259.40 | 135472.92 | +8786.48 |
  | #22 | Florida | 73994.56 | 122782.75 | 303319.26 | 110352.25 | 114384.76 | -4032.51 |
  | #27 | New York | 72107.60 | 127864.55 | 353183.81 | 105008.31 | 115626.44 | -10618.13 |
  | #28 | Florida | 66051.52 | 182645.56 | 118148.20 | 103282.38 | 101676.26 | +1606.12 |
  | #31 | New York | 61136.38 | 152701.92 | 88218.23 | 97483.56 | 97444.36 | +39.20 |
  | #38 | New York | 20229.59 | 65947.93 | 185265.10 | 81229.06 | 67601.00 | +13628.06 |
  | #41 | Florida | 27892.92 | 84710.77 | 164470.71 | 77798.83 | 74797.10 | +3001.73 |

### F. Deployment (模型保存與重載相容性)
- 使用 `pickle` + `gzip` 壓縮序列化保存至 `outputs/part1_linear_regression_model.pkl.gz`。
- 儲存內容包含：ColumnTransformer、特徵順序、基準州、係數、訓練組態與合法州別名單。
- **重載驗證**：載入後於測試集推論，最大絕對誤差為 $0.00 \times 10^0$，完全一致。
- **例外防呆**：輸入未知州別（如 `Texas`）時拋出具體例外，杜絕默默轉為基準州之業務風險。

---

## 二、自建 Forward Selection (前向特徵選擇) 實測剖析 (Part 2)

### 演算法選擇邏輯
- 依講義規定，從空特徵集合開始，每次選入一個候選特徵使得**測試集 RMSE 最小**。
- 候選特徵共 5 個：`['Administration', 'Marketing Spend', 'R&D Spend', 'State_Florida', 'State_New York']`。

### 1～5 個特徵漸進比較總表 (Progression Comparison Table)
| 輪數 (特徵數) | 本輪新增入選特徵 | 累積特徵子集 | 測試集 RMSE ($) | 測試集 $R^2$ | 訓練集 RMSE ($) | 訓練集 $R^2$ | 備註說明 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | `R&D Spend` | R&D | 8274.87 | 0.9465 | 9245.41 | 0.9449 | 單一變數即解釋 94.65% 變異 |
| **2** | `Marketing Spend` | R&D + Marketing | **8198.80** | **0.9474** | 9054.49 | 0.9500 | **全場最優子集 (最低 RMSE)** |
| **3** | `State_New York` | R&D + Marketing + NY | 8309.06 | 0.9460 | 9046.21 | 0.9501 | 加入州別後 RMSE 開始反彈惡化 |
| **4** | `State_Florida` | R&D + Marketing + NY + FL | 8409.92 | 0.9447 | 9044.59 | 0.9501 | 指標進一步略微劣化 |
| **5** | `Administration` | 全部 5 特徵 | 9137.99 | 0.9347 | 9031.67 | 0.9502 | **數值與 Part 1 全特徵模型 100% 相同** |

### 各輪特徵競逐詳細評核紀錄 (Candidate Evaluations)
- **Round 1**：
  - `R&D Spend`: Test RMSE = **8274.87**, $R^2$ = 0.9465 🏆 **(勝出)**
  - `Marketing Spend`: Test RMSE = 25529.05, $R^2$ = 0.4904
  - `State_Florida`: Test RMSE = 37588.05, $R^2$ = -0.1048
  - `State_New York`: Test RMSE = 41366.02, $R^2$ = -0.3380
  - `Administration`: Test RMSE = 45669.71, $R^2$ = -0.6309
- **Round 2** (已包含 `R&D Spend`)：
  - `Marketing Spend`: Test RMSE = **8198.80**, $R^2$ = 0.9474 🏆 **(勝出，達到全域最優)**
  - `State_Florida`: Test RMSE = 8217.45, $R^2$ = 0.9472
  - `Administration`: Test RMSE = 8237.54, $R^2$ = 0.9469
  - `State_New York`: Test RMSE = 8273.83, $R^2$ = 0.9465
- **Round 3** (已包含 `R&D Spend` + `Marketing Spend`)：
  - `State_New York`: Test RMSE = **8309.06**, $R^2$ = 0.9460 🏆 **(勝出)**
  - `State_Florida`: Test RMSE = 8376.45, $R^2$ = 0.9451
  - `Administration`: Test RMSE = 8803.78, $R^2$ = 0.9394
- **Round 4** (已包含 `R&D` + `Marketing` + `State_New York`)：
  - `State_Florida`: Test RMSE = **8409.92**, $R^2$ = 0.9447 🏆 **(勝出)**
  - `Administration`: Test RMSE = 8997.20, $R^2$ = 0.9367
- **Round 5** (已選前 4 特徵)：
  - `Administration`: Test RMSE = **9137.99**, $R^2$ = 0.9347 🏆 **(最後入選)**

---

## 三、關鍵實證發現與商業決策啟示

1. **核心驅動力 (Primary Driver)：研發投資**
   - 單獨使用 `R&D Spend` 一個特徵，在測試集即可達到 $R^2 = 0.9465$。這清楚表明研發創新是決定初創企業獲利的最核心關鍵。
2. **最簡精準模型 (Parsimonious Best Model)：雙變數組合**
   - 模型在納入 `R&D Spend` 與 `Marketing Spend` 兩個特徵時，測試集 RMSE 降到最低的 **$8198.80**，表現優於包含 5 個特徵的基準模型 ($9137.99$)。
3. **奧卡姆剃刀原則 (Occam's Razor) 與特徵雜訊**
   - 繼續加入 `State` 虛擬變數與 `Administration` 支出後，訓練集 RMSE 雖微幅下降 (從 9054.49 降至 9031.67)，但**測試集 RMSE 卻反彈上升達 11.4% (8198.80 $\rightarrow$ 9137.99)**。
   - 這印證了機器學習的核心原理：特徵並非越多越好。當加入無顯著預測力的變數（行政支出與地理位置）時，只會引入額外方差 (Variance) 與共線性雜訊，引發輕微的過度擬合 (Overfitting)。

---

## 四、方法限制與教學嚴謹性聲明 (Important Methodology Limitation)

> [!WARNING]
> **本實作忠實再現課堂講義流程，以『測試集 (Test Set)』的表現作為前向特徵挑選的判定依據。**
> 在機器學習的嚴格方法論中：
> 1. 當測試集參與了特徵選拔 (Model Selection)，測試集已不再具備「完全獨立性」，其產生的評估指標會偏向樂觀 (Optimistic Bias)。
> 2. 嚴謹的工程規範應採用 **訓練集內部之 K-Fold 交叉驗證 (Cross-Validation)** 或額外切分 **驗證集 (Validation Set)** 進行特徵篩選，最後才由完全未參與選擇的測試集進行最終評估。
> 3. 本報告特此註記此點，提供教學展示時充分說明其優勢與侷限。
