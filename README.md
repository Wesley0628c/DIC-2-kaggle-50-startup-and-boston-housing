# 📈 50 Startups 多元線性回歸實證研究室 (Aether Quant Lab)

> **課程作業實證成果展示專案**  
> 涵蓋 **CRISP-DM 六大階段規範**、**Ordinary Least Squares (OLS) 多元線性回歸基準**、**自建前向特徵選擇 (Forward Feature Selection)** 與 **奧卡姆剃刀谷地 (Valley at k=2) 實證**。

---

## 🌐 專案線上展示與 GitHub Pages 公開網址

- **GitHub Pages 預計公開網址**：  
  👉 **[https://wesley0628c.github.io/DIC-2-kaggle-50-startup-and-boston-housing/](https://wesley0628c.github.io/DIC-2-kaggle-50-startup-and-boston-housing/)**  
  *(備用直接連結：[https://wesley0628c.github.io/DIC-2-kaggle-50-startup-and-boston-housing/web/index.html](https://wesley0628c.github.io/DIC-2-kaggle-50-startup-and-boston-housing/web/index.html))*
- **GitHub 專案原始碼倉庫**：  
  🔗 [https://github.com/Wesley0628c/DIC-2-kaggle-50-startup-and-boston-housing.git](https://github.com/Wesley0628c/DIC-2-kaggle-50-startup-and-boston-housing.git)

---

## 🎯 專案核心成果摘要

| 評估維度 | Part 1：全特徵基準模型 (k = 5) | Part 2：前向選擇路徑最優子集 (k = 2) | 全子集窮舉全域最優 (k = 3 👑) |
| :--- | :---: | :---: | :---: |
| **納入特徵組合** | 全部 5 特徵 (R&D + Admin + Marketing + FL + NY) | **R&D Spend + Marketing Spend** | **R&D + Admin + State_Florida** |
| **測試集 RMSE (均方根誤差)** | **$9,137.99** | **$8,198.80** (前向選擇路徑最優) | **$8,171.65** (全域 31 組最低) |
| **測試集 R² (決定係數)** | 0.9347 (93.47%) | **0.9474 (94.74%)** | **0.9477 (94.77%)** |
| **測試集 MAE (平均絕對誤差)** | $7,514.29 | **$6,886.59** | **$6,801.37** |
| **訓練集校正決定係數 (Adj. R²)** | 0.9429 ($df=34$) | 0.9469 ($df=37$) | 0.9463 ($df=36$) |
| **訓練集整體 F-Statistic** | 129.70 ($p = 1.11 \times 10^{-16}$) | 349.04 ($p = 1.11 \times 10^{-16}$) | 230.12 ($p = 1.11 \times 10^{-16}$) |

---

## 📐 Problem 1：CRISP-DM 多元線性回歸基準模型

### 1. CRISP-DM 六大標準階段實踐
1. **Business Understanding (商業理解)**：預測 50 家初創企業之獲利 (Profit)，識別最關鍵支出驅動因數，為創投資金分配提供量化依據。
2. **Data Understanding (資料理解與 EDA)**：探索 50 筆無缺失真實資料，繪製支出 vs. 獲利之散佈圖，發現研發支出與獲利呈現極強正相關 ($r = 0.973$)。
3. **Data Preparation (資料準備)**：對 `State` 類別進行 One-Hot 編碼，**刪除基準州 California (加州)** 以避免虛擬變數陷阱 (Dummy Variable Trap)；固定 `test_size=0.2, random_state=0` 切分 40 筆訓練集與 10 筆測試集。
4. **Modeling (模型建置)**：採用 Ordinary Least Squares (OLS) 求解全域最佳回歸超平面。
5. **Evaluation (模型評估)**：嚴格驗證 $RMSE = \sqrt{MSE}$ 數學恆等式，產出殘差常態分佈圖與 45 度理想擬合線 ($y = x$)。
6. **Deployment (模型部署)**：以 `pickle + gzip` 導出壓縮模型，並匯出結構化 JSON 與視覺化 Web 儀表板。

### 2. 實測多元回歸經驗方程式
$$\text{Profit} = 42554.17 - 959.2842 \times \text{State\_Florida} + 699.3691 \times \text{State\_New York} + 0.7735 \times \text{R\&D Spend} + 0.0329 \times \text{Administration} + 0.0366 \times \text{Marketing Spend}$$

- **基準州 (Reference State)**：California (當地位於 California 時，$\text{State\_Florida}=0, \text{State\_New York}=0$)。
- **訓練集 OLS 統計推論 ($n=40, p=5, df=34$)**：
  - **R&D Spend ($\beta = +0.7735, t = 14.03, p = 1.11 \times 10^{-15}$)**：達到極顯著水準 ($p < 0.001$)，為企業獲利之核心支柱。
  - **Marketing Spend ($\beta = +0.0366, t = 1.88, p = 0.0681$)**：邊際顯著 ($p < 0.10$)。
  - **Administration ($\beta = +0.0329, t = 0.49, p = 0.6241$)** 與 **州別虛擬變數**：係數微小且未達顯著水準 ($p > 0.05$)。

---

## 📉 Problem 2：前向特徵選擇與全子集窮舉實測

### 1. 前向特徵選擇路徑之谷地 (Valley at k=2)
- **k = 1**：僅使用 `R&D Spend` 時，測試集 RMSE 為 **$8,274.87**。
- **k = 2 (路徑最優谷地)**：加入 `Marketing Spend` 後，測試集 RMSE 降至 **$8,198.80**。
- **k ≥ 3 (前向路徑誤差攀升)**：陸續加入州別虛擬變數（第 3 輪 `State_New York`、第 4 輪 `State_Florida`）與行政支出後，測試集 RMSE 反彈攀升至 **$9,137.99**。

### 2. 五大特徵選擇演算法真實基準測試矩陣 (All 5 Methods Benchmark)

| 演算法 (Method) | 篩選依據與運算方式 | 最優特徵數 $k$ | 最優特徵子集 | 最低測試集 RMSE | 測試集 R² |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Forward Selection (SFS)** | 貪婪依測試集 RMSE 最小逐輪選入 | $k=2$ | R&D Spend + Marketing Spend | **$8,198.80** | **0.9474** |
| **Lasso (L1 正則化)** | 僅訓練集標準化特徵擬合 L1 權重排序，再以 OLS 重擬合 | $k=2$ | R&D Spend + Marketing Spend | **$8,198.80** | **0.9474** |
| **RFE (遞迴特徵消除法)** | 僅訓練集標準化特徵逐輪剔除權重最小者，再以 OLS 重擬合 | $k=2$ | R&D Spend + Marketing Spend | **$8,198.80** | **0.9474** |
| **ElasticNet (L1 + L2)** | 僅訓練集混合正則化權重排序，再以 OLS 重擬合 | $k=2$ | R&D Spend + Marketing Spend | **$8,198.80** | **0.9474** |
| **Exhaustive (31 組窮舉)** | 完整遍歷 $2^5 - 1 = 31$ 組非空子集實測 | **$k=3$ (全域最優)** | **R&D + Administration + State_Florida** | **$8,171.65$** | **0.9477** |

---

## 🏛️ Topic 2：Boston Housing 波士頓房價多元回歸實證

針對題目要求「do two problems but choose one of the topic to upload」與專案名稱 `50-startup-and-boston-housing`，專案重用通用的 `src/regression_workflow.py` 完整實作第二大資料集：
- **資料特性**：506 筆樣本、13 個自變數、目標 `medv`（房價中位數，千美元）。
- **實測指標**：Test RMSE = **$5.7835** ($5,783.50)，Test R² = **0.5892** (58.9%)，Train Adj R² = **0.7291**。
- **特徵經濟學意義**：房間數 RM ($\beta = +3.81$) 與查爾斯河畔 CHAS ($\beta = +2.69$) 顯著溢價；氮氧化物濃度 NOX ($\beta = -17.77$) 與弱勢人口比 LSTAT ($\beta = -0.52$) 顯著折抑房價。

---

## 🧪 互動自訂資料集回歸實驗室 (Client-side ML Engine)

針對 GitHub Pages 靜態網站環境，實作純前端 JavaScript 線性回歸求解器 (`web/regression_engine.js`)：
1. **資料輸入多軌化**：支援遠端公開 CSV 網址（Fetch，含 CORS 診斷引導）與本機 CSV 上傳（FileReader API）。
2. **RFC 4180 標準解析**：支援雙引號、轉義雙引號、欄位內逗號、跨行文字與 UTF-8 BOM。
3. **資料品質預覽**：前 5 筆預覽、真實總筆數、欄位資料型別判斷（數值型 vs 類別型）與缺失值檢測。
4. **安全特徵與目標配置**：目標變數 y 下拉（非數值禁用）、特徵變數 X 網格勾選（限制 y 不得重複作為特徵）、支援自訂 test_size 與 random_state。
5. **數值穩定 OLS 求解**：主元高斯消去法並加入 Ridge $\lambda=10^{-9}$ 正則化防護，避免共線性導致奇異矩陣。
6. **動態呈現**：真實計算 MAE, MSE, RMSE, R²，即時繪製 Actual vs Pred 散佈圖與殘差圖，並提供逐筆預測 CSV 與模型 JSON 下載。

---

## 🖥️ 網頁設計特色 (Aether Quant Lab)

- **下拉選單五大視角切換器**：支援 `part1` (50 Startups Baseline)、`part2` (前向選擇)、`boston` (波士頓房價)、`custom` (自訂資料集 ML 引擎)、`all` (全景一覽)。
- **杜絕假成功與寫死指標**：所有指標與方程式均由實際模型計算導出；缺少數值時顯示「未計算」，安全處理數值 0。
- **完整精度試算器**：直接讀取導出之高精度係數浮點數，不再採用手動四捨五入公式。

---

## 📂 專案目錄結構

```text
DIC Multiple Linear Regression/
├── index.html               # 專案首頁 (GitHub Pages 部署根目錄)
├── docs/                    # GitHub Pages /docs 部署目錄
├── web/                     # 實證視覺化前端
│   ├── index.html           # Aether Quant Lab 互動儀表板
│   ├── style.css            # 獨立現代幾何科技感樣式系統
│   ├── app.js               # 前端互動控制器 (五視角切換、圖表渲染、試算器)
│   ├── regression_engine.js # 純前端瀏覽器端線性回歸求解器 (RFC 4180 CSV + OLS)
│   ├── chart.umd.min.js     # 本地 Chart.js 核心庫
│   ├── data.json            # 完整真實計算結構化 JSON
│   ├── default_data.js      # 離線回退內嵌資料集
│   └── downloads/           # PKL.GZ 模型檔、CSV 預測明細與 Jupyter Notebook
├── src/                     # Python 核心實作原始碼
│   ├── download_data.py     # 資料獲取與品質完整性檢核
│   ├── part1_crispdm.py     # Part 1 CRISP-DM 六階段回歸基準
│   ├── part2_forward_selection.py # Part 2 前向選擇與奧卡姆剃刀實測
│   ├── benchmark_algorithms.py   # 五大特徵選擇演算法與 31 組窮舉實測
│   ├── regression_workflow.py    # 通用可重用 CRISP-DM 回歸工作流模組
│   ├── export_web_data.py   # 真實數據匯出模組
│   └── pipeline.py          # 端到端一鍵執行全流程腳本
├── data/                    # 原始數據集 (50_Startups.csv, BostonHousing.csv)
├── outputs/                 # 序列化模型與評估報告產出
├── tests/                   # 自動化測試套件 (17 項單元測試全數通過)
│   ├── test_pipeline.py     # 後端數值、模型推論與前向選擇測試 (13 項)
│   ├── test_web_integration.py # 前端 DOM、自訂回歸與無假指標測試 (4 項)
│   └── test_js_engine.js    # Node.js 端 RegressionEngine 測試
├── 50_Startup_Multiple_Linear_Regression_CRISP_DM.ipynb # 真實 Kernel 執行的 Jupyter Notebook
├── results.md               # 完整成果報告與統計檢定紀錄
├── RunLog.md                # 實作執行歷程記錄
├── ToDo.md                  # 12 大改善檢核追蹤清單
└── requirements.txt         # Python 依賴清單
```

---

## 🚀 本機重現與執行指南

### 1. 安裝環境依賴
```bash
pip install -r requirements.txt
```

### 2. 端到端一鍵執行全流程
```bash
python3 src/pipeline.py
```
*此指令將自動執行：50 Startups 資料檢核 $\rightarrow$ CRISP-DM 基準回歸 $\rightarrow$ 前向特徵選擇 $\rightarrow$ 五大演算法與 31 組窮舉 $\rightarrow$ Boston Housing 通用回歸 $\rightarrow$ 匯出 Web 結構化數據 $\rightarrow$ 產出圖表並 100% 同步至 docs/ 與 web/downloads/。*

### 3. 執行自動化測試套件 (17 項測試全數通過)
```bash
python3 -m pytest -v tests/
```

### 4. 本機啟動互動網頁展示平台
```bash
python3 -m http.server 8000 --directory web
```
開啟瀏覽器訪問：`http://localhost:8000` 即可體驗完整互動儀表板。

> **方法學核心洞察**：
> 1. 前向特徵選擇與三種收縮／消除法一致判定雙特徵【R&D Spend + Marketing Spend】為貪婪路徑中最優組合。
> 2. 透過窮舉 31 組非空特徵子集，實證發現 $k=3$ 的組合【`R&D Spend + Administration + State_Florida`】具備更低的測試 RMSE ($8,171.65$)。這充分證明**貪婪演算法因每一步鎖定局部最佳特徵，無法保證抵達全子集全域最佳解**。

---

## 🖥️ 網頁設計特色 (Aether Quant Lab)

- **原創科技感設計**：不同於一般模板，採用專屬「深空曜石 (Deep Obsidian)」與「學術純淨 (Paper Light)」雙主題無縫切換。
- **特徵谷地焦點視覺化**：動態校準 Y 軸範圍與高亮標籤，清晰凸顯 k=2 處之奧卡姆剃刀最低點。
- **五法同台超圖表 (Compare All 5)**：一鍵切換將五種特徵選擇方法之 RMSE / R² 曲線同時繪製於同一畫布，展現跨演算法之收斂對比。
- **即時回歸推論試算器**：內建滑桿與單選控制項，即時試算各項支出配置下之預期獲利。
- **原生圖表雙軌切換**：可自由於「Chart.js 向量動態圖表」與「Python Matplotlib 300 DPI 原生科研圖檔」間切換。

---

## 📂 專案目錄結構

```text
DIC Multiple Linear Regression/
├── index.html               # GitHub Pages 根目錄重定向入口
├── docs/                    # GitHub Pages 發布目錄 (包含完整靜態網頁)
├── web/                     # 實證視覺化前端
│   ├── index.html           # Aether Quant Lab 互動儀表板
│   ├── style.css            # 現代科技感獨立樣式系統 (深淺色雙主題)
│   ├── app.js               # 動態五演算法切換與 OLS 推論控制器
│   ├── chart.umd.min.js     # 本地 Chart.js 核心庫 (免外網依賴)
│   ├── data.json            # 完整真實計算結構化 JSON
│   ├── default_data.js      # 離線回退內嵌資料集
│   ├── images/              # Python 原生 300 DPI 繪圖檔 (3 張)
│   └── downloads/           # PKL.GZ 模型檔與 CSV 預測明細下載庫
├── src/                     # Python 核心實作原始碼
│   ├── download_data.py     # 資料獲取與品質完整性檢核
│   ├── part1_crispdm.py     # Part 1 CRISP-DM 六階段回歸基準
│   ├── part2_forward_selection.py # Part 2 前向選擇與奧卡姆剃刀實測
│   ├── export_web_data.py   # 真實數據匯出模組
│   └── pipeline.py          # 端到端一鍵執行流程
├── data/                    # 原始數據集 (50_Startups.csv)
├── outputs/                 # 序列化模型與評估報告產出
├── tests/                   # 自動化測試套件 (test_pipeline.py)
├── results.md               # 完整成果報告與統計檢定紀錄
├── RunLog.md                # 實作執行歷程記錄
└── requirements.txt         # Python 依賴清單
```

---

## 🚀 本機重現與執行指南

### 1. 安裝環境依賴
```bash
pip install -r requirements.txt
```

### 2. 端到端一鍵執行全流程
```bash
python3 src/pipeline.py
```
*此指令將自動執行：資料下載檢核 $\rightarrow$ CRISP-DM 基準回歸 $\rightarrow$ 前向特徵選擇 $\rightarrow$ 繪製 300 DPI 圖表 $\rightarrow$ 序列化保存模型 $\rightarrow$ 匯出 Web 結構化數據。*

### 3. 執行數值一致性測試套件
```bash
python3 tests/test_pipeline.py
```

### 4. 本機啟動互動網頁展示平台
```bash
python3 -m http.server 8000 --directory web
```
開啟瀏覽器訪問：`http://localhost:8000` 即可體驗完整互動儀表板。
