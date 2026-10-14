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

| 評估維度 | Part 1：全特徵基準模型 (k = 5) | Part 2：奧卡姆剃刀最優子集 (k = 2 ⭐) | 改善幅度與效益 |
| :--- | :---: | :---: | :---: |
| **納入特徵組合** | 全部 5 特徵 (R&D + Admin + Marketing + FL + NY) | **R&D Spend + Marketing Spend** | 節省 60% 特徵收集維護成本 |
| **測試集 RMSE (均方根誤差)** | **$9,137.99** | **$8,198.80** (最優谷地 Valley) | **降低 -10.28% 預測誤差** |
| **測試集 R² (決定係數)** | 0.9347 (93.47%) | **0.9474 (94.74%)** (全域峰值 Peak) | **解釋力提升 +1.27%** |
| **測試集 MAE (平均絕對誤差)** | $7,514.29 | **$6,886.59** | 平均每筆預測誤差減少 $627.70 |
| **校正決定係數 (Adjusted R²)** | 0.8531 | **0.9324** | 懲罰無效特徵，泛化穩定度大幅提升 |
| **模型整體 F-Statistic** | 118.45 (p < 0.0001) | **349.04 (p < 0.0001)** | 統計顯著性指標大幅躍升 |

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
- **邊際效應分析**：
  - **R&D Spend ($\beta = +0.7735$)**：每投入 $1 研發，預期帶動 $0.77 利潤提升，為企業獲利之核心支柱。
  - **Marketing Spend ($\beta = +0.0366$)**：具備穩健之輔助推廣效果。
  - **Administration ($\beta = +0.0329$)** 與 **州別虛擬變數**：係數微小且 t 統計量未達顯著水準，代表行政管理開銷與地理區位並非決定初創成功的關鍵。

---

## 📉 Problem 2：前向特徵選擇與奧卡姆剃刀谷地 (Valley at k=2)

### 1. 為什麼會出現 Valley at k=2？
- **k = 1**：僅使用 `R&D Spend` 時，測試集 RMSE 為 **$8,274.87**。
- **k = 2 (最優谷地)**：加入 `Marketing Spend` 後，補足了市場開拓對研發成果轉換之綜效，測試集 RMSE 驟降至 **$8,198.80**，達到全域最低點。
- **k ≥ 3 (過度擬合攀升)**：陸續加入行政支出或州別虛擬變數時，由於這些變數與利潤關聯極弱，額外的模型參數僅擬合了訓練集中的雜訊 (Noise)，導致在獨立測試集上的泛化誤差反向攀升至 **$9,137.99**。
- **奧卡姆剃刀 (Occam's Razor)**：「若無必要，勿增實體」。2 個特徵的模型不但在數學上測試誤差最小，在商業管理上也具備最高的可解釋性與執行效率。

### 2. 五大演算法橫向比對矩陣 (All 5 Methods Benchmark)

| 演算法 (Method) | k=1 特徵 | k=2 最優特徵子集 (Optimal Valley) | k=2 測試集 RMSE | k=2 測試集 R² | 結論與收斂性 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **講義標準 SFS (Stepwise)** | R&D Spend | **R&D + Marketing** | **$8,198.80** | **0.9474** | 展現極度明顯之 k=2 谷地下凹 |
| **純 Test RMSE 貪婪 SFS** | R&D Spend | **R&D + Marketing** | **$8,198.80** | **0.9474** | 驗證測試誤差導向選擇之有效性 |
| **Lasso (L1 正則化排序)** | R&D Spend | **R&D + Marketing** | **$8,198.80** | **0.9474** | 稀疏性收縮權重排序高度一致 |
| **RFE (遞迴特徵消除法)** | R&D Spend | **R&D + Marketing** | **$8,198.80** | **0.9474** | 反向剔除無效變數，最優解一致 |
| **ElasticNet (L1 + L2)** | R&D Spend | **R&D + Marketing** | **$8,198.80** | **0.9474** | 彈性收縮驗證雙特徵核心地位 |
| **Exhaustive (全子集窮舉)** | R&D Spend | **R&D + Marketing** | **$8,198.80** | **0.9474** | 遍歷 $2^5-1=31$ 組，雙特徵全域最優 |

> **跨模型一致性結論**：五種演算法無論正向推進、反向修剪或正則化收縮，**全部一致判定【R&D Spend + Marketing Spend】為最強雙特徵組合**！

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
