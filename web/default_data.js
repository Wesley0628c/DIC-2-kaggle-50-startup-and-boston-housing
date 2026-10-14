window.EMBEDDED_DATA = {
  "metadata": {
    "title": "50 Startups 多元線性回歸分析與特徵選擇實證研究",
    "subtitle": "Multiple Linear Regression, 5-Fold CV Feature Selection, Optuna Tuning & Boston Housing",
    "generated_at": "2026-10-14 21:37:52",
    "data_source_url": "https://raw.githubusercontent.com/harimittapalli/Mulitple-Linear-Reggression/master/50_Startups.csv",
    "total_rows": 50,
    "raw_features_count": 4,
    "encoded_features_count": 5,
    "test_size": 0.2,
    "random_state": 0,
    "train_rows_count": 40,
    "test_rows_count": 10
  },
  "dataset": {
    "head_5": [
      {
        "R&D Spend": 165349.2,
        "Administration": 136897.8,
        "Marketing Spend": 471784.1,
        "State": "New York",
        "Profit": 192261.83
      },
      {
        "R&D Spend": 162597.7,
        "Administration": 151377.59,
        "Marketing Spend": 443898.53,
        "State": "California",
        "Profit": 191792.06
      },
      {
        "R&D Spend": 153441.51,
        "Administration": 101145.55,
        "Marketing Spend": 407934.54,
        "State": "Florida",
        "Profit": 191050.39
      },
      {
        "R&D Spend": 144372.41,
        "Administration": 118671.85,
        "Marketing Spend": 383199.62,
        "State": "New York",
        "Profit": 182901.99
      },
      {
        "R&D Spend": 142107.34,
        "Administration": 91391.77,
        "Marketing Spend": 366168.42,
        "State": "Florida",
        "Profit": 166187.94
      }
    ],
    "all_rows": [
      {
        "R&D Spend": 165349.2,
        "Administration": 136897.8,
        "Marketing Spend": 471784.1,
        "State": "New York",
        "Profit": 192261.83,
        "original_id": 0
      },
      {
        "R&D Spend": 162597.7,
        "Administration": 151377.59,
        "Marketing Spend": 443898.53,
        "State": "California",
        "Profit": 191792.06,
        "original_id": 1
      },
      {
        "R&D Spend": 153441.51,
        "Administration": 101145.55,
        "Marketing Spend": 407934.54,
        "State": "Florida",
        "Profit": 191050.39,
        "original_id": 2
      },
      {
        "R&D Spend": 144372.41,
        "Administration": 118671.85,
        "Marketing Spend": 383199.62,
        "State": "New York",
        "Profit": 182901.99,
        "original_id": 3
      },
      {
        "R&D Spend": 142107.34,
        "Administration": 91391.77,
        "Marketing Spend": 366168.42,
        "State": "Florida",
        "Profit": 166187.94,
        "original_id": 4
      },
      {
        "R&D Spend": 131876.9,
        "Administration": 99814.71,
        "Marketing Spend": 362861.36,
        "State": "New York",
        "Profit": 156991.12,
        "original_id": 5
      },
      {
        "R&D Spend": 134615.46,
        "Administration": 147198.87,
        "Marketing Spend": 127716.82,
        "State": "California",
        "Profit": 156122.51,
        "original_id": 6
      },
      {
        "R&D Spend": 130298.13,
        "Administration": 145530.06,
        "Marketing Spend": 323876.68,
        "State": "Florida",
        "Profit": 155752.6,
        "original_id": 7
      },
      {
        "R&D Spend": 120542.52,
        "Administration": 148718.95,
        "Marketing Spend": 311613.29,
        "State": "New York",
        "Profit": 152211.77,
        "original_id": 8
      },
      {
        "R&D Spend": 123334.88,
        "Administration": 108679.17,
        "Marketing Spend": 304981.62,
        "State": "California",
        "Profit": 149759.96,
        "original_id": 9
      },
      {
        "R&D Spend": 101913.08,
        "Administration": 110594.11,
        "Marketing Spend": 229160.95,
        "State": "Florida",
        "Profit": 146121.95,
        "original_id": 10
      },
      {
        "R&D Spend": 100671.96,
        "Administration": 91790.61,
        "Marketing Spend": 249744.55,
        "State": "California",
        "Profit": 144259.4,
        "original_id": 11
      },
      {
        "R&D Spend": 93863.75,
        "Administration": 127320.38,
        "Marketing Spend": 249839.44,
        "State": "Florida",
        "Profit": 141585.52,
        "original_id": 12
      },
      {
        "R&D Spend": 91992.39,
        "Administration": 135495.07,
        "Marketing Spend": 252664.93,
        "State": "California",
        "Profit": 134307.35,
        "original_id": 13
      },
      {
        "R&D Spend": 119943.24,
        "Administration": 156547.42,
        "Marketing Spend": 256512.92,
        "State": "Florida",
        "Profit": 132602.65,
        "original_id": 14
      },
      {
        "R&D Spend": 114523.61,
        "Administration": 122616.84,
        "Marketing Spend": 261776.23,
        "State": "New York",
        "Profit": 129917.04,
        "original_id": 15
      },
      {
        "R&D Spend": 78013.11,
        "Administration": 121597.55,
        "Marketing Spend": 264346.06,
        "State": "California",
        "Profit": 126992.93,
        "original_id": 16
      },
      {
        "R&D Spend": 94657.16,
        "Administration": 145077.58,
        "Marketing Spend": 282574.31,
        "State": "New York",
        "Profit": 125370.37,
        "original_id": 17
      },
      {
        "R&D Spend": 91749.16,
        "Administration": 114175.79,
        "Marketing Spend": 294919.57,
        "State": "Florida",
        "Profit": 124266.9,
        "original_id": 18
      },
      {
        "R&D Spend": 86419.7,
        "Administration": 153514.11,
        "Marketing Spend": 0.0,
        "State": "New York",
        "Profit": 122776.86,
        "original_id": 19
      },
      {
        "R&D Spend": 76253.86,
        "Administration": 113867.3,
        "Marketing Spend": 298664.47,
        "State": "California",
        "Profit": 118474.03,
        "original_id": 20
      },
      {
        "R&D Spend": 78389.47,
        "Administration": 153773.43,
        "Marketing Spend": 299737.29,
        "State": "New York",
        "Profit": 111313.02,
        "original_id": 21
      },
      {
        "R&D Spend": 73994.56,
        "Administration": 122782.75,
        "Marketing Spend": 303319.26,
        "State": "Florida",
        "Profit": 110352.25,
        "original_id": 22
      },
      {
        "R&D Spend": 67532.53,
        "Administration": 105751.03,
        "Marketing Spend": 304768.73,
        "State": "Florida",
        "Profit": 108733.99,
        "original_id": 23
      },
      {
        "R&D Spend": 77044.01,
        "Administration": 99281.34,
        "Marketing Spend": 140574.81,
        "State": "New York",
        "Profit": 108552.04,
        "original_id": 24
      },
      {
        "R&D Spend": 64664.71,
        "Administration": 139553.16,
        "Marketing Spend": 137962.62,
        "State": "California",
        "Profit": 107404.34,
        "original_id": 25
      },
      {
        "R&D Spend": 75328.87,
        "Administration": 144135.98,
        "Marketing Spend": 134050.07,
        "State": "Florida",
        "Profit": 105733.54,
        "original_id": 26
      },
      {
        "R&D Spend": 72107.6,
        "Administration": 127864.55,
        "Marketing Spend": 353183.81,
        "State": "New York",
        "Profit": 105008.31,
        "original_id": 27
      },
      {
        "R&D Spend": 66051.52,
        "Administration": 182645.56,
        "Marketing Spend": 118148.2,
        "State": "Florida",
        "Profit": 103282.38,
        "original_id": 28
      },
      {
        "R&D Spend": 65605.48,
        "Administration": 153032.06,
        "Marketing Spend": 107138.38,
        "State": "New York",
        "Profit": 101004.64,
        "original_id": 29
      },
      {
        "R&D Spend": 61994.48,
        "Administration": 115641.28,
        "Marketing Spend": 91131.24,
        "State": "Florida",
        "Profit": 99937.59,
        "original_id": 30
      },
      {
        "R&D Spend": 61136.38,
        "Administration": 152701.92,
        "Marketing Spend": 88218.23,
        "State": "New York",
        "Profit": 97483.56,
        "original_id": 31
      },
      {
        "R&D Spend": 63408.86,
        "Administration": 129219.61,
        "Marketing Spend": 46085.25,
        "State": "California",
        "Profit": 97427.84,
        "original_id": 32
      },
      {
        "R&D Spend": 55493.95,
        "Administration": 103057.49,
        "Marketing Spend": 214634.81,
        "State": "Florida",
        "Profit": 96778.92,
        "original_id": 33
      },
      {
        "R&D Spend": 46426.07,
        "Administration": 157693.92,
        "Marketing Spend": 210797.67,
        "State": "California",
        "Profit": 96712.8,
        "original_id": 34
      },
      {
        "R&D Spend": 46014.02,
        "Administration": 85047.44,
        "Marketing Spend": 205517.64,
        "State": "New York",
        "Profit": 96479.51,
        "original_id": 35
      },
      {
        "R&D Spend": 28663.76,
        "Administration": 127056.21,
        "Marketing Spend": 201126.82,
        "State": "Florida",
        "Profit": 90708.19,
        "original_id": 36
      },
      {
        "R&D Spend": 44069.95,
        "Administration": 51283.14,
        "Marketing Spend": 197029.42,
        "State": "California",
        "Profit": 89949.14,
        "original_id": 37
      },
      {
        "R&D Spend": 20229.59,
        "Administration": 65947.93,
        "Marketing Spend": 185265.1,
        "State": "New York",
        "Profit": 81229.06,
        "original_id": 38
      },
      {
        "R&D Spend": 38558.51,
        "Administration": 82982.09,
        "Marketing Spend": 174999.3,
        "State": "California",
        "Profit": 81005.76,
        "original_id": 39
      },
      {
        "R&D Spend": 28754.33,
        "Administration": 118546.05,
        "Marketing Spend": 172795.67,
        "State": "California",
        "Profit": 78239.91,
        "original_id": 40
      },
      {
        "R&D Spend": 27892.92,
        "Administration": 84710.77,
        "Marketing Spend": 164470.71,
        "State": "Florida",
        "Profit": 77798.83,
        "original_id": 41
      },
      {
        "R&D Spend": 23640.93,
        "Administration": 96189.63,
        "Marketing Spend": 148001.11,
        "State": "California",
        "Profit": 71498.49,
        "original_id": 42
      },
      {
        "R&D Spend": 15505.73,
        "Administration": 127382.3,
        "Marketing Spend": 35534.17,
        "State": "New York",
        "Profit": 69758.98,
        "original_id": 43
      },
      {
        "R&D Spend": 22177.74,
        "Administration": 154806.14,
        "Marketing Spend": 28334.72,
        "State": "California",
        "Profit": 65200.33,
        "original_id": 44
      },
      {
        "R&D Spend": 1000.23,
        "Administration": 124153.04,
        "Marketing Spend": 1903.93,
        "State": "New York",
        "Profit": 64926.08,
        "original_id": 45
      },
      {
        "R&D Spend": 1315.46,
        "Administration": 115816.21,
        "Marketing Spend": 297114.46,
        "State": "Florida",
        "Profit": 49490.75,
        "original_id": 46
      },
      {
        "R&D Spend": 0.0,
        "Administration": 135426.92,
        "Marketing Spend": 0.0,
        "State": "California",
        "Profit": 42559.73,
        "original_id": 47
      },
      {
        "R&D Spend": 542.05,
        "Administration": 51743.15,
        "Marketing Spend": 0.0,
        "State": "New York",
        "Profit": 35673.41,
        "original_id": 48
      },
      {
        "R&D Spend": 0.0,
        "Administration": 116983.8,
        "Marketing Spend": 45173.06,
        "State": "California",
        "Profit": 14681.4,
        "original_id": 49
      }
    ],
    "columns": [
      "R&D Spend",
      "Administration",
      "Marketing Spend",
      "State",
      "Profit"
    ],
    "describe": {
      "R&D Spend": {
        "count": 50,
        "mean": 73721.6156,
        "std": 45902.25648230753,
        "min": 0.0,
        "q25": 39936.37,
        "median": 73051.08,
        "q75": 101602.8,
        "max": 165349.2
      },
      "Administration": {
        "count": 50,
        "mean": 121344.63960000001,
        "std": 28017.802755488683,
        "min": 51283.14,
        "q25": 103730.875,
        "median": 122699.795,
        "q75": 144842.18,
        "max": 182645.56
      },
      "Marketing Spend": {
        "count": 50,
        "mean": 211025.09780000002,
        "std": 122290.31072584528,
        "min": 0.0,
        "q25": 129300.1325,
        "median": 212716.24,
        "q75": 299469.08499999996,
        "max": 471784.1
      },
      "Profit": {
        "count": 50,
        "mean": 112012.63920000002,
        "std": 40306.18033765055,
        "min": 14681.4,
        "q25": 90138.9025,
        "median": 107978.19,
        "q75": 139765.97749999998,
        "max": 192261.83
      }
    },
    "state_counts": {
      "New York": 17,
      "California": 17,
      "Florida": 16
    },
    "null_counts": {
      "R&D Spend": 0,
      "Administration": 0,
      "Marketing Spend": 0,
      "State": 0,
      "Profit": 0
    },
    "duplicate_count": 0
  },
  "crisp_dm_stages": [
    {
      "id": "business_understanding",
      "phase_name_zh": "商業理解 (Business Understanding)",
      "phase_name_en": "Business Understanding",
      "goal": "確認預測目標為 Profit，分析研發 (R&D)、行政 (Administration)、行銷 (Marketing) 與地理位置 (State) 對利潤之關聯。",
      "key_takeaway": "線性模型提供變數間敏感度與相依性度量，但不可逕行宣稱為因果關係 (Association does not imply Causality)。"
    },
    {
      "id": "data_understanding",
      "phase_name_zh": "資料理解 (Data Understanding)",
      "phase_name_en": "Data Understanding",
      "goal": "探勘 50 筆企業資料之分佈、極值、缺失值 (0 筆) 與重複列 (0 筆)。",
      "key_takeaway": "R&D Spend 與 Profit 呈現極高度線性正相關 (r = 0.973)，行政支出關聯相對微弱。"
    },
    {
      "id": "data_preparation",
      "phase_name_zh": "資料準備 (Data Preparation)",
      "phase_name_en": "Data Preparation",
      "goal": "嚴格先切分 80% 訓練集與 20% 測試集；前處理規則 (中位數與 One-Hot 類別) 100% 僅從訓練集學習，避免虛擬變數陷阱 (Dummy Variable Trap)。",
      "key_takeaway": "特徵維度擴增至 5 維；切分 train=40, test=10 (test_size=0.2, random_state=0)。"
    },
    {
      "id": "modeling",
      "phase_name_zh": "模型建置 (Modeling)",
      "phase_name_en": "Modeling",
      "goal": "採用 Ordinary Least Squares (OLS) 多元線性回歸模型，求解最佳擬合平面之截距與迴歸係數。",
      "key_takeaway": "截距 b0 = 42554.17，R&D 支出係數約 +0.7735，表示每增加 1 美元研發預算，利潤預期增加約 0.77 美元。"
    },
    {
      "id": "evaluation",
      "phase_name_zh": "模型評估 (Evaluation)",
      "phase_name_en": "Evaluation",
      "goal": "計算 MAE, MSE, RMSE, R² 指標，進行 OLS 殘差假設檢定與 5-Fold 交叉驗證。",
      "key_takeaway": "基準模型 5-Fold CV RMSE 為 10042.10，測試集 RMSE 為 9137.99，R² 為 0.9347。"
    },
    {
      "id": "deployment",
      "phase_name_zh": "模型部署與打包 (Deployment)",
      "phase_name_en": "Deployment",
      "goal": "使用 pickle + gzip 序列化包含前處理與回歸器的端到端 Pipeline，確保由原始欄位直接推論之完全可重現性。",
      "key_takeaway": "模型重載驗證數值誤差為 0.00e+00，未知類別容錯驗證通過。"
    }
  ],
  "part1": {
    "title": "Part 1: CRISP-DM 多元線性回歸基準模型 (Baseline Model)",
    "encoded_features": [
      "R&D Spend",
      "Administration",
      "Marketing Spend",
      "State_Florida",
      "State_New York"
    ],
    "model": {
      "intercept": 42554.16761773238,
      "coefficients": {
        "R&D Spend": 0.7734671927328,
        "Administration": 0.03288459753625586,
        "Marketing Spend": 0.036610025863856234,
        "State_Florida": -959.2841600571062,
        "State_New York": 699.3690525192605
      },
      "parameter_statistics": {
        "Intercept": {
          "coef": 42554.1676,
          "stderr": 8358.5382,
          "tstat": 5.0911,
          "pvalue": 1.3087e-05
        },
        "State_Florida": {
          "coef": -959.2842,
          "stderr": 4038.1077,
          "tstat": -0.2376,
          "pvalue": 0.8136
        },
        "State_New York": {
          "coef": 699.3691,
          "stderr": 3661.5627,
          "tstat": 0.191,
          "pvalue": 0.8497
        },
        "R&D Spend": {
          "coef": 0.7735,
          "stderr": 0.0551,
          "tstat": 14.0298,
          "pvalue": 1.1102e-15
        },
        "Administration": {
          "coef": 0.0329,
          "stderr": 0.0665,
          "tstat": 0.4945,
          "pvalue": 0.6241
        },
        "Marketing Spend": {
          "coef": 0.0366,
          "stderr": 0.0194,
          "tstat": 1.8841,
          "pvalue": 0.0681
        }
      },
      "formula": "Profit = 42554.17 + 0.7735 * (R&D Spend) + 0.0329 * (Administration) + 0.0366 * (Marketing Spend) - 959.2842 * (State_Florida) + 699.3691 * (State_New York)"
    },
    "train_metrics": {
      "MAE": 6341.5358,
      "MSE": 81571001.8008,
      "RMSE": 9031.6666,
      "R2": 0.9502,
      "Adjusted_R2": 0.9429,
      "F_statistic": 129.7,
      "F_pvalue": 1.11e-16
    },
    "cv_metrics": {
      "cv_rmse_mean": 10042.1,
      "cv_rmse_std": 2850.48
    },
    "test_metrics": {
      "MAE": 7514.2936596406025,
      "MSE": 83502864.03257722,
      "RMSE": 9137.990152794935,
      "R2": 0.9347068473282426
    },
    "test_predictions": [
      {
        "original_id": 2,
        "state": "Florida",
        "rd_spend": 153441.51,
        "admin_spend": 101145.55,
        "marketing_spend": 407934.54,
        "actual_profit": 191050.39,
        "predicted_profit": 178537.4822105507,
        "residual": 12512.907789449324,
        "abs_error": 12512.907789449324
      },
      {
        "original_id": 4,
        "state": "Florida",
        "rd_spend": 142107.34,
        "admin_spend": 91391.77,
        "marketing_spend": 366168.42,
        "actual_profit": 166187.94,
        "predicted_profit": 167921.06569550425,
        "residual": -1733.125695504248,
        "abs_error": 1733.125695504248
      },
      {
        "original_id": 10,
        "state": "Florida",
        "rd_spend": 101913.08,
        "admin_spend": 110594.11,
        "marketing_spend": 229160.95,
        "actual_profit": 146121.95,
        "predicted_profit": 132447.73845174484,
        "residual": 13674.211548255174,
        "abs_error": 13674.211548255174
      },
      {
        "original_id": 11,
        "state": "California",
        "rd_spend": 100671.96,
        "admin_spend": 91790.61,
        "marketing_spend": 249744.55,
        "actual_profit": 144259.4,
        "predicted_profit": 132582.27760815568,
        "residual": 11677.122391844314,
        "abs_error": 11677.122391844314
      },
      {
        "original_id": 22,
        "state": "Florida",
        "rd_spend": 73994.56,
        "admin_spend": 122782.75,
        "marketing_spend": 303319.26,
        "actual_profit": 110352.25,
        "predicted_profit": 113969.43533012444,
        "residual": -3617.185330124441,
        "abs_error": 3617.185330124441
      },
      {
        "original_id": 27,
        "state": "New York",
        "rd_spend": 72107.6,
        "admin_spend": 127864.55,
        "marketing_spend": 353183.81,
        "actual_profit": 105008.31,
        "predicted_profit": 116161.24230165104,
        "residual": -11152.932301651046,
        "abs_error": 11152.932301651046
      },
      {
        "original_id": 28,
        "state": "Florida",
        "rd_spend": 66051.52,
        "admin_spend": 182645.56,
        "marketing_spend": 118148.2,
        "actual_profit": 103282.38,
        "predicted_profit": 103015.2015979618,
        "residual": 267.1784020382038,
        "abs_error": 267.1784020382038
      },
      {
        "original_id": 31,
        "state": "New York",
        "rd_spend": 61136.38,
        "admin_spend": 152701.92,
        "marketing_spend": 88218.23,
        "actual_profit": 97483.56,
        "predicted_profit": 98791.7337468745,
        "residual": -1308.1737468744975,
        "abs_error": 1308.1737468744975
      },
      {
        "original_id": 38,
        "state": "New York",
        "rd_spend": 20229.59,
        "admin_spend": 65947.93,
        "marketing_spend": 185265.1,
        "actual_profit": 81229.06,
        "predicted_profit": 67851.69209675625,
        "residual": 13377.36790324375,
        "abs_error": 13377.36790324375
      },
      {
        "original_id": 41,
        "state": "Florida",
        "rd_spend": 27892.92,
        "admin_spend": 84710.77,
        "marketing_spend": 164470.71,
        "actual_profit": 77798.83,
        "predicted_profit": 71976.09851257897,
        "residual": 5822.731487421028,
        "abs_error": 5822.731487421028
      }
    ]
  },
  "part2": {
    "title": "Part 2: 5-Fold CV 前向特徵選擇 (Forward Selection)",
    "methodology_note": "本實作依機器學習嚴謹規範，每輪在 40 筆訓練集內部以 5-Fold CV RMSE 最低者納入特徵；最佳特徵數 k 亦由 CV 最低點決定。未參與決策的 10 筆獨立測試集僅作為模型確定後的最終泛化評估。",
    "candidate_columns": [
      "Administration",
      "Marketing Spend",
      "R&D Spend",
      "State_Florida",
      "State_New York"
    ],
    "rounds_summary": [
      {
        "step": 1,
        "added_feature": "R&D Spend",
        "selected_features": [
          "R&D Spend"
        ],
        "cv_rmse": 9625.473491612547,
        "cv_rmse_std": 2650.077088741687,
        "test_rmse": 8274.86801822599,
        "test_r2": 0.946458760778722,
        "test_mae": 6772.453280477899,
        "test_mse": 68473440.71905932,
        "train_rmse": 9493.580597976612,
        "train_r2": 0.9449589778363044,
        "train_adj_r2": 0.9435105298846282,
        "train_f_stat": 652.3941552354449,
        "train_f_pvalue": 1.1102230246251563e-16,
        "intercept": 48416.297661385026,
        "coefficients": {
          "R&D Spend": 0.8516227998779371
        },
        "candidates_evaluation": [
          {
            "candidate": "Administration",
            "cv_rmse": 38425.482678701504,
            "cv_rmse_std": 7791.430525886112,
            "is_selected": false
          },
          {
            "candidate": "Marketing Spend",
            "cv_rmse": 27220.692982901113,
            "cv_rmse_std": 7961.149714490011,
            "is_selected": false
          },
          {
            "candidate": "R&D Spend",
            "cv_rmse": 9625.473491612547,
            "cv_rmse_std": 2650.077088741687,
            "is_selected": true
          },
          {
            "candidate": "State_Florida",
            "cv_rmse": 41183.590005815626,
            "cv_rmse_std": 8014.086815678363,
            "is_selected": false
          },
          {
            "candidate": "State_New York",
            "cv_rmse": 41910.27487159964,
            "cv_rmse_std": 6033.738641626326,
            "is_selected": false
          }
        ],
        "predictions": [
          {
            "original_id": 2,
            "actual_profit": 191050.39,
            "predicted_profit": 179090.58602508352,
            "residual": 11959.80397491649,
            "abs_error": 11959.80397491649
          },
          {
            "original_id": 4,
            "actual_profit": 166187.94,
            "predicted_profit": 169438.148435391,
            "residual": -3250.2084353909886,
            "abs_error": 3250.2084353909886
          },
          {
            "original_id": 10,
            "actual_profit": 146121.95,
            "predicted_profit": 135207.80019516923,
            "residual": 10914.149804830784,
            "abs_error": 10914.149804830784
          },
          {
            "original_id": 11,
            "actual_profit": 144259.4,
            "predicted_profit": 134150.8341057847,
            "residual": 10108.56589421528,
            "abs_error": 10108.56589421528
          },
          {
            "original_id": 22,
            "actual_profit": 110352.25,
            "predicted_profit": 111431.75202432103,
            "residual": -1079.502024321031,
            "abs_error": 1079.502024321031
          },
          {
            "original_id": 27,
            "actual_profit": 105008.31,
            "predicted_profit": 109824.77386586336,
            "residual": -4816.463865863363,
            "abs_error": 4816.463865863363
          },
          {
            "original_id": 28,
            "actual_profit": 103282.38,
            "predicted_profit": 104667.27805997859,
            "residual": -1384.8980599785864,
            "abs_error": 1384.8980599785864
          },
          {
            "original_id": 31,
            "actual_profit": 97483.56,
            "predicted_profit": 100481.43277138655,
            "residual": -2997.872771386552,
            "abs_error": 2997.872771386552
          },
          {
            "original_id": 38,
            "actual_profit": 81229.06,
            "predicted_profit": 65644.27773756774,
            "residual": 15584.782262432258,
            "abs_error": 15584.782262432258
          },
          {
            "original_id": 41,
            "actual_profit": 77798.83,
            "predicted_profit": 72170.54428855634,
            "residual": 5628.285711443663,
            "abs_error": 5628.285711443663
          }
        ]
      },
      {
        "step": 2,
        "added_feature": "Marketing Spend",
        "selected_features": [
          "R&D Spend",
          "Marketing Spend"
        ],
        "cv_rmse": 9474.318466367124,
        "cv_rmse_std": 2513.081772226985,
        "test_rmse": 8198.797190788491,
        "test_r2": 0.9474386447268488,
        "test_mae": 6886.594588246512,
        "test_mse": 67220275.37568125,
        "train_rmse": 9078.617918486692,
        "train_r2": 0.9496654854268444,
        "train_adj_r2": 0.9469447008553225,
        "train_f_stat": 349.04104329569407,
        "train_f_pvalue": 1.1102230246251563e-16,
        "intercept": 46346.33876123658,
        "coefficients": {
          "R&D Spend": 0.7886182999424612,
          "Marketing Spend": 0.03257662731702063
        },
        "candidates_evaluation": [
          {
            "candidate": "Administration",
            "cv_rmse": 9701.569919854584,
            "cv_rmse_std": 2695.568462847259,
            "is_selected": false
          },
          {
            "candidate": "Marketing Spend",
            "cv_rmse": 9474.318466367124,
            "cv_rmse_std": 2513.081772226985,
            "is_selected": true
          },
          {
            "candidate": "State_Florida",
            "cv_rmse": 9845.925137646074,
            "cv_rmse_std": 2717.436813825152,
            "is_selected": false
          },
          {
            "candidate": "State_New York",
            "cv_rmse": 9802.050579444973,
            "cv_rmse_std": 2563.312056637042,
            "is_selected": false
          }
        ],
        "predictions": [
          {
            "original_id": 2,
            "actual_profit": 191050.39,
            "predicted_profit": 180642.25299736098,
            "residual": 10408.137002639036,
            "abs_error": 10408.137002639036
          },
          {
            "original_id": 4,
            "actual_profit": 166187.94,
            "predicted_profit": 170343.31979498419,
            "residual": -4155.379794984183,
            "abs_error": 4155.379794984183
          },
          {
            "original_id": 10,
            "actual_profit": 146121.95,
            "predicted_profit": 134182.14951650103,
            "residual": 11939.800483498984,
            "abs_error": 11939.800483498984
          },
          {
            "original_id": 11,
            "actual_profit": 144259.4,
            "predicted_profit": 133873.92383811908,
            "residual": 10385.476161880913,
            "abs_error": 10385.476161880913
          },
          {
            "original_id": 22,
            "actual_profit": 110352.25,
            "predicted_profit": 114580.9213645215,
            "residual": -4228.671364521506,
            "abs_error": 4228.671364521506
          },
          {
            "original_id": 27,
            "actual_profit": 105008.31,
            "predicted_profit": 114717.24903894302,
            "residual": -9708.93903894302,
            "abs_error": 9708.93903894302
          },
          {
            "original_id": 28,
            "actual_profit": 103282.38,
            "predicted_profit": 102284.64605182887,
            "residual": 997.7339481711388,
            "abs_error": 997.7339481711388
          },
          {
            "original_id": 31,
            "actual_profit": 97483.56,
            "predicted_profit": 97433.45922275007,
            "residual": 50.1007772499288,
            "abs_error": 50.1007772499288
          },
          {
            "original_id": 38,
            "actual_profit": 81229.06,
            "predicted_profit": 68335.07575312015,
            "residual": 12893.984246879845,
            "abs_error": 12893.984246879845
          },
          {
            "original_id": 41,
            "actual_profit": 77798.83,
            "predicted_profit": 73701.10693630343,
            "residual": 4097.723063696569,
            "abs_error": 4097.723063696569
          }
        ]
      },
      {
        "step": 3,
        "added_feature": "State_New York",
        "selected_features": [
          "R&D Spend",
          "Marketing Spend",
          "State_New York"
        ],
        "cv_rmse": 9602.033536091889,
        "cv_rmse_std": 2455.890438310911,
        "test_rmse": 8309.059683481717,
        "test_r2": 0.9460153830071388,
        "test_mae": 7028.435331818949,
        "test_mse": 69040472.8236613,
        "train_rmse": 9069.684820873445,
        "train_r2": 0.9497644921078412,
        "train_adj_r2": 0.9455781997834948,
        "train_f_stat": 226.87486169664257,
        "train_f_pvalue": 1.1102230246251563e-16,
        "intercept": 46100.29130551412,
        "coefficients": {
          "R&D Spend": 0.7858709815066743,
          "Marketing Spend": 0.03327182941032258,
          "State_New York": 864.0018700527295
        },
        "candidates_evaluation": [
          {
            "candidate": "Administration",
            "cv_rmse": 9613.456019570289,
            "cv_rmse_std": 2542.591242712831,
            "is_selected": false
          },
          {
            "candidate": "State_Florida",
            "cv_rmse": 9634.198709549211,
            "cv_rmse_std": 2752.70072632981,
            "is_selected": false
          },
          {
            "candidate": "State_New York",
            "cv_rmse": 9602.033536091889,
            "cv_rmse_std": 2455.890438310911,
            "is_selected": true
          }
        ],
        "predictions": [
          {
            "original_id": 2,
            "actual_profit": 191050.39,
            "predicted_profit": 180258.24979853872,
            "residual": 10792.140201461298,
            "abs_error": 10792.140201461298
          },
          {
            "original_id": 4,
            "actual_profit": 166187.94,
            "predicted_profit": 169961.41927630414,
            "residual": -3773.4792763041332,
            "abs_error": 3773.4792763041332
          },
          {
            "original_id": 10,
            "actual_profit": 146121.95,
            "predicted_profit": 133815.4275493898,
            "residual": 12306.52245061021,
            "abs_error": 12306.52245061021
          },
          {
            "original_id": 11,
            "actual_profit": 144259.4,
            "predicted_profit": 133524.92138467255,
            "residual": 10734.478615327447,
            "abs_error": 10734.478615327447
          },
          {
            "original_id": 22,
            "actual_profit": 110352.25,
            "predicted_profit": 114342.45547445389,
            "residual": -3990.205474453891,
            "abs_error": 3990.205474453891
          },
          {
            "original_id": 27,
            "actual_profit": 105008.31,
            "predicted_profit": 115382.63503846529,
            "residual": -10374.325038465293,
            "abs_error": 10374.325038465293
          },
          {
            "original_id": 28,
            "actual_profit": 103282.38,
            "predicted_profit": 101939.27091345853,
            "residual": 1343.1090865414735,
            "abs_error": 1343.1090865414735
          },
          {
            "original_id": 31,
            "actual_profit": 97483.56,
            "predicted_profit": 97944.78203137245,
            "residual": -461.2220313724538,
            "abs_error": 461.2220313724538
          },
          {
            "original_id": 38,
            "actual_profit": 81229.06,
            "predicted_profit": 69026.24972723081,
            "residual": 12202.810272769188,
            "abs_error": 12202.810272769188
          },
          {
            "original_id": 41,
            "actual_profit": 77798.83,
            "predicted_profit": 73492.7691291159,
            "residual": 4306.060870884103,
            "abs_error": 4306.060870884103
          }
        ]
      },
      {
        "step": 4,
        "added_feature": "State_Florida",
        "selected_features": [
          "R&D Spend",
          "Marketing Spend",
          "State_New York",
          "State_Florida"
        ],
        "cv_rmse": 9779.97303784842,
        "cv_rmse_std": 2721.853729055403,
        "test_rmse": 8409.916714488052,
        "test_r2": 0.9446968769428332,
        "test_mae": 7109.161997627586,
        "test_mse": 70726699.14462551,
        "train_rmse": 9064.118245163729,
        "train_r2": 0.9498261378977876,
        "train_adj_r2": 0.9440919822289634,
        "train_f_stat": 165.6435912721812,
        "train_f_pvalue": 1.1102230246251563e-16,
        "intercept": 46329.060178542495,
        "coefficients": {
          "R&D Spend": 0.7851417173112168,
          "Marketing Spend": 0.03395640234030121,
          "State_New York": 554.6573328267123,
          "State_Florida": -826.4681587619034
        },
        "candidates_evaluation": [
          {
            "candidate": "Administration",
            "cv_rmse": 9817.13775671058,
            "cv_rmse_std": 2475.2856297973085,
            "is_selected": false
          },
          {
            "candidate": "State_Florida",
            "cv_rmse": 9779.97303784842,
            "cv_rmse_std": 2721.853729055403,
            "is_selected": true
          }
        ],
        "predictions": [
          {
            "original_id": 2,
            "actual_profit": 191050.39,
            "predicted_profit": 179827.91205675254,
            "residual": 11222.477943247475,
            "abs_error": 11222.477943247475
          },
          {
            "original_id": 4,
            "actual_profit": 166187.94,
            "predicted_profit": 169510.75518374197,
            "residual": -3322.815183741972,
            "abs_error": 3322.815183741972
          },
          {
            "original_id": 10,
            "actual_profit": 146121.95,
            "predicted_profit": 133300.28408634168,
            "residual": 12821.665913658333,
            "abs_error": 12821.665913658333
          },
          {
            "original_id": 11,
            "actual_profit": 144259.4,
            "predicted_profit": 133851.2421601261,
            "residual": 10408.157839873893,
            "abs_error": 10408.157839873893
          },
          {
            "original_id": 22,
            "actual_profit": 110352.25,
            "predicted_profit": 113898.4387599909,
            "residual": -3546.188759990895,
            "abs_error": 3546.188759990895
          },
          {
            "original_id": 27,
            "actual_profit": 105008.31,
            "predicted_profit": 115491.25395900001,
            "residual": -10482.943959000011,
            "abs_error": 10482.943959000011
          },
          {
            "original_id": 28,
            "actual_profit": 103282.38,
            "predicted_profit": 101374.28367857914,
            "residual": 1908.0963214208605,
            "abs_error": 1908.0963214208605
          },
          {
            "original_id": 31,
            "actual_profit": 97483.56,
            "predicted_profit": 97880.01360638956,
            "residual": -396.4536063895648,
            "abs_error": 396.4536063895648
          },
          {
            "original_id": 38,
            "actual_profit": 81229.06,
            "predicted_profit": 69057.74881968717,
            "residual": 12171.311180312827,
            "abs_error": 12171.311180312827
          },
          {
            "original_id": 41,
            "actual_profit": 77798.83,
            "predicted_profit": 72987.32073135997,
            "residual": 4811.509268640031,
            "abs_error": 4811.509268640031
          }
        ]
      },
      {
        "step": 5,
        "added_feature": "Administration",
        "selected_features": [
          "R&D Spend",
          "Marketing Spend",
          "State_New York",
          "State_Florida",
          "Administration"
        ],
        "cv_rmse": 10042.095066709877,
        "cv_rmse_std": 2850.4838878187848,
        "test_rmse": 9137.99015279213,
        "test_r2": 0.9347068473282828,
        "test_mae": 7514.293659638902,
        "test_mse": 83502864.03252594,
        "train_rmse": 9031.66661257897,
        "train_r2": 0.9501847627493608,
        "train_adj_r2": 0.9428589925654433,
        "train_f_stat": 129.70441863373296,
        "train_f_pvalue": 1.1102230246251563e-16,
        "intercept": 42554.16761774788,
        "coefficients": {
          "R&D Spend": 0.7734671927328999,
          "Marketing Spend": 0.036610025863816474,
          "State_New York": 699.3690525188922,
          "State_Florida": -959.2841600570787,
          "Administration": 0.03288459753613828
        },
        "candidates_evaluation": [
          {
            "candidate": "Administration",
            "cv_rmse": 10042.095066709877,
            "cv_rmse_std": 2850.4838878187848,
            "is_selected": true
          }
        ],
        "predictions": [
          {
            "original_id": 2,
            "actual_profit": 191050.39,
            "predicted_profit": 178537.4822105534,
            "residual": 12512.907789446617,
            "abs_error": 12512.907789446617
          },
          {
            "original_id": 4,
            "actual_profit": 166187.94,
            "predicted_profit": 167921.06569550867,
            "residual": -1733.125695508672,
            "abs_error": 1733.125695508672
          },
          {
            "original_id": 10,
            "actual_profit": 146121.95,
            "predicted_profit": 132447.7384517484,
            "residual": 13674.211548251624,
            "abs_error": 13674.211548251624
          },
          {
            "original_id": 11,
            "actual_profit": 144259.4,
            "predicted_profit": 132582.2776081605,
            "residual": 11677.122391839483,
            "abs_error": 11677.122391839483
          },
          {
            "original_id": 22,
            "actual_profit": 110352.25,
            "predicted_profit": 113969.43533012088,
            "residual": -3617.1853301208757,
            "abs_error": 3617.1853301208757
          },
          {
            "original_id": 27,
            "actual_profit": 105008.31,
            "predicted_profit": 116161.24230164429,
            "residual": -11152.932301644294,
            "abs_error": 11152.932301644294
          },
          {
            "original_id": 28,
            "actual_profit": 103282.38,
            "predicted_profit": 103015.20159795776,
            "residual": 267.1784020422492,
            "abs_error": 267.1784020422492
          },
          {
            "original_id": 31,
            "actual_profit": 97483.56,
            "predicted_profit": 98791.73374687428,
            "residual": -1308.1737468742795,
            "abs_error": 1308.1737468742795
          },
          {
            "original_id": 38,
            "actual_profit": 81229.06,
            "predicted_profit": 67851.69209675828,
            "residual": 13377.367903241713,
            "abs_error": 13377.367903241713
          },
          {
            "original_id": 41,
            "actual_profit": 77798.83,
            "predicted_profit": 71976.0985125808,
            "residual": 5822.731487419209,
            "abs_error": 5822.731487419209
          }
        ]
      }
    ],
    "progression_table": [
      {
        "step": 1,
        "num_features": 1,
        "added_feature": "R&D Spend",
        "selected_features": "['R&D Spend']",
        "selected_features_str": "R&D Spend",
        "cv_rmse": 9625.473491612547,
        "cv_rmse_std": 2650.077088741687,
        "train_rmse": 9493.580597976612,
        "train_r2": 0.9449589778363044,
        "train_adj_r2": 0.9435105298846282,
        "train_f_stat": 652.3941552354449,
        "train_f_pvalue": 1.1102230246251563e-16,
        "test_rmse": 8274.86801822599,
        "test_r2": 0.946458760778722,
        "test_mae": 6772.453280477899,
        "test_mse": 68473440.71905932
      },
      {
        "step": 2,
        "num_features": 2,
        "added_feature": "Marketing Spend",
        "selected_features": "['R&D Spend', 'Marketing Spend']",
        "selected_features_str": "R&D Spend, Marketing Spend",
        "cv_rmse": 9474.318466367124,
        "cv_rmse_std": 2513.081772226985,
        "train_rmse": 9078.617918486692,
        "train_r2": 0.9496654854268444,
        "train_adj_r2": 0.9469447008553225,
        "train_f_stat": 349.04104329569407,
        "train_f_pvalue": 1.1102230246251563e-16,
        "test_rmse": 8198.797190788491,
        "test_r2": 0.9474386447268488,
        "test_mae": 6886.594588246512,
        "test_mse": 67220275.37568125
      },
      {
        "step": 3,
        "num_features": 3,
        "added_feature": "State_New York",
        "selected_features": "['R&D Spend', 'Marketing Spend', 'State_New York']",
        "selected_features_str": "R&D Spend, Marketing Spend, State_New York",
        "cv_rmse": 9602.033536091889,
        "cv_rmse_std": 2455.890438310911,
        "train_rmse": 9069.684820873445,
        "train_r2": 0.9497644921078412,
        "train_adj_r2": 0.9455781997834948,
        "train_f_stat": 226.87486169664257,
        "train_f_pvalue": 1.1102230246251563e-16,
        "test_rmse": 8309.059683481717,
        "test_r2": 0.9460153830071388,
        "test_mae": 7028.435331818949,
        "test_mse": 69040472.8236613
      },
      {
        "step": 4,
        "num_features": 4,
        "added_feature": "State_Florida",
        "selected_features": "['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida']",
        "selected_features_str": "R&D Spend, Marketing Spend, State_New York, State_Florida",
        "cv_rmse": 9779.97303784842,
        "cv_rmse_std": 2721.853729055403,
        "train_rmse": 9064.118245163729,
        "train_r2": 0.9498261378977876,
        "train_adj_r2": 0.9440919822289634,
        "train_f_stat": 165.6435912721812,
        "train_f_pvalue": 1.1102230246251563e-16,
        "test_rmse": 8409.916714488052,
        "test_r2": 0.9446968769428332,
        "test_mae": 7109.161997627586,
        "test_mse": 70726699.14462551
      },
      {
        "step": 5,
        "num_features": 5,
        "added_feature": "Administration",
        "selected_features": "['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']",
        "selected_features_str": "R&D Spend, Marketing Spend, State_New York, State_Florida, Administration",
        "cv_rmse": 10042.095066709877,
        "cv_rmse_std": 2850.4838878187848,
        "train_rmse": 9031.66661257897,
        "train_r2": 0.9501847627493608,
        "train_adj_r2": 0.9428589925654433,
        "train_f_stat": 129.70441863373296,
        "train_f_pvalue": 1.1102230246251563e-16,
        "test_rmse": 9137.99015279213,
        "test_r2": 0.9347068473282828,
        "test_mae": 7514.293659638902,
        "test_mse": 83502864.03252594
      }
    ],
    "best_feature_count": 2,
    "best_features": [
      "R&D Spend",
      "Marketing Spend"
    ],
    "best_cv_rmse": 9474.318466367124,
    "best_test_rmse": 8198.797190788491,
    "best_test_r2": 0.9474386447268488,
    "historical_leakage_demo": {
      "description": "歷史講義探索示範區 (以最終 Test RMSE 選特徵)：此處保留原先講義做法以供教學比對，清楚標記測試集參與選模所導致的評估樂觀偏差 (Data Leakage / Optimism Bias)。",
      "progression": [
        {
          "step": 1,
          "num_features": 1,
          "added_feature": "R&D Spend",
          "selected_features_str": "R&D Spend",
          "test_rmse": 8274.86801822599,
          "test_r2": 0.946458760778722
        },
        {
          "step": 2,
          "num_features": 2,
          "added_feature": "Marketing Spend",
          "selected_features_str": "R&D Spend, Marketing Spend",
          "test_rmse": 8198.797190788491,
          "test_r2": 0.9474386447268488
        },
        {
          "step": 3,
          "num_features": 3,
          "added_feature": "State_New York",
          "selected_features_str": "R&D Spend, Marketing Spend, State_New York",
          "test_rmse": 8309.059683481717,
          "test_r2": 0.9460153830071388
        },
        {
          "step": 4,
          "num_features": 4,
          "added_feature": "State_Florida",
          "selected_features_str": "R&D Spend, Marketing Spend, State_New York, State_Florida",
          "test_rmse": 8409.916714488052,
          "test_r2": 0.9446968769428332
        },
        {
          "step": 5,
          "num_features": 5,
          "added_feature": "Administration",
          "selected_features_str": "R&D Spend, Marketing Spend, State_New York, State_Florida, Administration",
          "test_rmse": 9137.99015279213,
          "test_r2": 0.9347068473282828
        }
      ]
    }
  },
  "optuna_tuning": {
    "title": "Step 4: Optuna 超參數自動調校 (Ridge 正則化回歸)",
    "description": "透過 TPESampler (30 trials, seed=42) 於對數尺度 [1e-4, 1e4] 搜尋 Ridge 正則化參數 alpha。Objective 完全限制於訓練集內 5-Fold CV RMSE。",
    "summary": {
      "model_type": "Pipeline(StandardScaler + Ridge)",
      "pipeline": "Pipeline(steps=[('preprocessor',\n                 ColumnTransformer(transformers=[('num',\n                                                  Pipeline(steps=[('imputer',\n                                                                   SimpleImputer(strategy='median')),\n                                                                  ('scaler',\n                                                                   StandardScaler())]),\n                                                  ['R&D Spend',\n                                                   'Administration',\n                                                   'Marketing Spend']),\n                                                 ('cat',\n                                                  OneHotEncoder(drop='first',\n                                                                handle_unknown='ignore',\n                                                                sparse_output=False),\n                                                  ['State'])])),\n                ('regressor',\n                 Ridge(alpha=0.18784949239433849, random_state=42))])",
      "feature_names": [
        "R&D Spend",
        "Administration",
        "Marketing Spend",
        "State_Florida",
        "State_New York"
      ],
      "best_alpha": 0.18784949239433849,
      "best_trial_number": 22,
      "baseline_metrics": {
        "cv_rmse": 10042.095066712931,
        "cv_std": 2850.483887882537,
        "test_rmse": 9137.990152794944,
        "test_r2": 0.9347068473282425,
        "test_mae": 7514.2936596406025
      },
      "tuned_metrics": {
        "cv_rmse": 10034.578142541317,
        "cv_std": 2884.0709670603615,
        "test_rmse": 9294.32356617669,
        "test_r2": 0.9324536567247735,
        "test_mae": 7594.534664053295
      },
      "cv_improvement_pct": 0.07485414270305693,
      "test_improvement_pct": -1.7108074179083026,
      "model": "Ridge(alpha=0.18784949239433849, random_state=42)",
      "preprocessor": "ColumnTransformer(transformers=[('num',\n                                 Pipeline(steps=[('imputer',\n                                                  SimpleImputer(strategy='median')),\n                                                 ('scaler', StandardScaler())]),\n                                 ['R&D Spend', 'Administration',\n                                  'Marketing Spend']),\n                                ('cat',\n                                 OneHotEncoder(drop='first',\n                                               handle_unknown='ignore',\n                                               sparse_output=False),\n                                 ['State'])])",
      "column_transformer": "ColumnTransformer(transformers=[('num',\n                                 Pipeline(steps=[('imputer',\n                                                  SimpleImputer(strategy='median')),\n                                                 ('scaler', StandardScaler())]),\n                                 ['R&D Spend', 'Administration',\n                                  'Marketing Spend']),\n                                ('cat',\n                                 OneHotEncoder(drop='first',\n                                               handle_unknown='ignore',\n                                               sparse_output=False),\n                                 ['State'])])",
      "coefficients": {
        "R&D Spend": 35328.54352017618,
        "Administration": 963.08344270305,
        "Marketing Spend": 4777.058676305769,
        "State_Florida": -985.163885226428,
        "State_New York": 799.8571932875885
      },
      "intercept": 109412.78820365595
    },
    "trials": [
      {
        "trial_number": 0,
        "alpha": 0.09915644566638405,
        "cv_rmse": 10036.260971298334,
        "cv_std": 2869.4503804028914
      },
      {
        "trial_number": 1,
        "alpha": 4033.800832600386,
        "cv_rmse": 40010.61020527319,
        "cv_std": 7567.984816944391
      },
      {
        "trial_number": 2,
        "alpha": 71.77141927992028,
        "cv_rmse": 25011.305940767717,
        "cv_std": 5068.564278935122
      },
      {
        "trial_number": 3,
        "alpha": 6.155564318973023,
        "cv_rmse": 11543.323036215806,
        "cv_std": 3837.821210741195
      },
      {
        "trial_number": 4,
        "alpha": 0.001770716864353783,
        "cv_rmse": 10041.951634139998,
        "cv_std": 2850.8496627694826
      },
      {
        "trial_number": 5,
        "alpha": 0.001769930294063334,
        "cv_rmse": 10041.951697525681,
        "cv_std": 2850.849500519407
      },
      {
        "trial_number": 6,
        "alpha": 0.00029152036385288323,
        "cv_rmse": 10042.071351010825,
        "cv_std": 2850.5441784519267
      },
      {
        "trial_number": 7,
        "alpha": 849.9808989183019,
        "cv_rmse": 38254.10824027144,
        "cv_std": 7215.766944709811
      },
      {
        "trial_number": 8,
        "alpha": 6.440507553993717,
        "cv_rmse": 11624.96270282385,
        "cv_std": 3888.6758193473934
      },
      {
        "trial_number": 9,
        "alpha": 46.19347374377382,
        "cv_rmse": 21373.333609121262,
        "cv_std": 4841.553314309063
      },
      {
        "trial_number": 10,
        "alpha": 0.19884646792631183,
        "cv_rmse": 10034.584844136587,
        "cv_std": 2885.7479055605995
      },
      {
        "trial_number": 11,
        "alpha": 0.1318704442321766,
        "cv_rmse": 10035.271080144095,
        "cv_std": 2875.0801471897676
      },
      {
        "trial_number": 12,
        "alpha": 0.08111837990757255,
        "cv_rmse": 10036.999067470311,
        "cv_std": 2866.2194732847533
      },
      {
        "trial_number": 13,
        "alpha": 0.059539767066579,
        "cv_rmse": 10038.066620529202,
        "cv_std": 2862.230718624184
      },
      {
        "trial_number": 14,
        "alpha": 0.5838307937299033,
        "cv_rmse": 10059.635090482207,
        "cv_std": 2931.035711455011
      },
      {
        "trial_number": 15,
        "alpha": 1.3492708006340848,
        "cv_rmse": 10200.703695229993,
        "cv_std": 2996.2324306859127
      },
      {
        "trial_number": 16,
        "alpha": 0.010724872926713714,
        "cv_rmse": 10041.248910377106,
        "cv_std": 2852.6834689292855
      },
      {
        "trial_number": 17,
        "alpha": 0.015786004581735354,
        "cv_rmse": 10040.868296877856,
        "cv_std": 2853.7083855791743
      },
      {
        "trial_number": 18,
        "alpha": 0.9060546871526487,
        "cv_rmse": 10108.402204439806,
        "cv_std": 2958.5388129143307
      },
      {
        "trial_number": 19,
        "alpha": 0.00012747172406239344,
        "cv_rmse": 10042.08469172032,
        "cv_std": 2850.5102543263456
      },
      {
        "trial_number": 20,
        "alpha": 55.8525565732922,
        "cv_rmse": 22926.198538478122,
        "cv_std": 4911.05392201069
      },
      {
        "trial_number": 21,
        "alpha": 0.1289028658550792,
        "cv_rmse": 10035.34267268441,
        "cv_std": 2874.581315584285
      },
      {
        "trial_number": 22,
        "alpha": 0.18784949239433849,
        "cv_rmse": 10034.578142541317,
        "cv_std": 2884.0709670603615
      },
      {
        "trial_number": 23,
        "alpha": 0.009216840631730925,
        "cv_rmse": 10041.364629127977,
        "cv_std": 2852.376467849156
      },
      {
        "trial_number": 24,
        "alpha": 0.35277502270607325,
        "cv_rmse": 10039.193844286941,
        "cv_std": 2906.579234264209
      },
      {
        "trial_number": 25,
        "alpha": 2.725955614985709,
        "cv_rmse": 10564.745917249804,
        "cv_std": 3177.9164298510705
      },
      {
        "trial_number": 26,
        "alpha": 0.0027575631918797363,
        "cv_rmse": 10041.872338988062,
        "cv_std": 2851.053063510379
      },
      {
        "trial_number": 27,
        "alpha": 14.699470979224309,
        "cv_rmse": 14039.586695264883,
        "cv_std": 4691.436590834892
      },
      {
        "trial_number": 28,
        "alpha": 0.02932335223939986,
        "cv_rmse": 10039.908510653398,
        "cv_std": 2856.4092544855785
      },
      {
        "trial_number": 29,
        "alpha": 0.2955363398407532,
        "cv_rmse": 10036.537335957064,
        "cv_std": 2899.3643064276926
      }
    ]
  },
  "algorithms_benchmark": {
    "sfs": {
      "name": "Forward Selection (前向特徵選擇)",
      "progression": [
        {
          "k": 1,
          "features": [
            "R&D Spend"
          ],
          "cv_rmse": 9625.473491612547,
          "cv_std": 2650.077088741687,
          "test_rmse": 8274.86801822599,
          "test_r2": 0.9464587607787219,
          "test_mae": 6772.453280477899,
          "test_mse": 68473440.71905932,
          "train_rmse": 9493.580597976612,
          "train_r2": 0.9449589778363044,
          "train_adj_r2": 0.9435105298846282,
          "train_f_stat": 652.3941552354449,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 48416.297661385026,
          "coefficients": {
            "R&D Spend": 0.8516227998779371
          },
          "test_predictions": [
            179090.58602508352,
            169438.148435391,
            135207.80019516923,
            134150.8341057847,
            111431.75202432103,
            109824.77386586336,
            104667.27805997859,
            100481.43277138655,
            65644.27773756774,
            72170.54428855634
          ]
        },
        {
          "k": 2,
          "features": [
            "R&D Spend",
            "Marketing Spend"
          ],
          "cv_rmse": 9474.318466367124,
          "cv_std": 2513.081772226985,
          "test_rmse": 8198.797190788491,
          "test_r2": 0.9474386447268488,
          "test_mae": 6886.594588246512,
          "test_mse": 67220275.37568125,
          "train_rmse": 9078.617918486692,
          "train_r2": 0.9496654854268443,
          "train_adj_r2": 0.9469447008553223,
          "train_f_stat": 349.04104329569407,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 46346.33876123658,
          "coefficients": {
            "R&D Spend": 0.7886182999424612,
            "Marketing Spend": 0.03257662731702063
          },
          "test_predictions": [
            180642.25299736098,
            170343.31979498419,
            134182.14951650103,
            133873.92383811908,
            114580.9213645215,
            114717.24903894302,
            102284.64605182887,
            97433.45922275007,
            68335.07575312015,
            73701.10693630343
          ]
        },
        {
          "k": 3,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "State_New York"
          ],
          "cv_rmse": 9602.033536091889,
          "cv_std": 2455.890438310911,
          "test_rmse": 8309.059683481717,
          "test_r2": 0.9460153830071389,
          "test_mae": 7028.435331818949,
          "test_mse": 69040472.8236613,
          "train_rmse": 9069.684820873443,
          "train_r2": 0.9497644921078413,
          "train_adj_r2": 0.9455781997834948,
          "train_f_stat": 226.87486169664254,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 46100.29130551412,
          "coefficients": {
            "R&D Spend": 0.7858709815066743,
            "Marketing Spend": 0.03327182941032258,
            "State_New York": 864.0018700527295
          },
          "test_predictions": [
            180258.24979853872,
            169961.41927630414,
            133815.4275493898,
            133524.92138467255,
            114342.45547445389,
            115382.63503846529,
            101939.27091345853,
            97944.78203137245,
            69026.24972723081,
            73492.7691291159
          ]
        },
        {
          "k": 4,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "State_New York",
            "State_Florida"
          ],
          "cv_rmse": 9779.97303784842,
          "cv_std": 2721.853729055403,
          "test_rmse": 8409.916714488052,
          "test_r2": 0.9446968769428332,
          "test_mae": 7109.161997627586,
          "test_mse": 70726699.14462551,
          "train_rmse": 9064.118245163727,
          "train_r2": 0.9498261378977877,
          "train_adj_r2": 0.9440919822289634,
          "train_f_stat": 165.6435912721812,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 46329.060178542495,
          "coefficients": {
            "R&D Spend": 0.7851417173112168,
            "Marketing Spend": 0.03395640234030121,
            "State_New York": 554.6573328267123,
            "State_Florida": -826.4681587619034
          },
          "test_predictions": [
            179827.91205675254,
            169510.75518374197,
            133300.28408634168,
            133851.2421601261,
            113898.4387599909,
            115491.25395900001,
            101374.28367857914,
            97880.01360638956,
            69057.74881968717,
            72987.32073135997
          ]
        },
        {
          "k": 5,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "State_New York",
            "State_Florida",
            "Administration"
          ],
          "cv_rmse": 10042.095066709875,
          "cv_std": 2850.4838878187848,
          "test_rmse": 9137.99015279213,
          "test_r2": 0.9347068473282827,
          "test_mae": 7514.293659638902,
          "test_mse": 83502864.03252594,
          "train_rmse": 9031.66661257897,
          "train_r2": 0.9501847627493607,
          "train_adj_r2": 0.9428589925654431,
          "train_f_stat": 129.70441863373296,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42554.16761774788,
          "coefficients": {
            "R&D Spend": 0.7734671927328999,
            "Marketing Spend": 0.036610025863816474,
            "State_New York": 699.3690525188922,
            "State_Florida": -959.2841600570787,
            "Administration": 0.03288459753613828
          },
          "test_predictions": [
            178537.4822105534,
            167921.06569550867,
            132447.7384517484,
            132582.2776081605,
            113969.43533012088,
            116161.24230164429,
            103015.20159795776,
            98791.73374687428,
            67851.69209675828,
            71976.0985125808
          ]
        }
      ],
      "best_k": 2
    },
    "lasso_direct": {
      "algorithm": "Lasso Direct",
      "description": "L1 正則化 (LassoCV) 直接預測，保留係數縮減懲罰",
      "best_alpha": 1480.9856485464968,
      "cv_rmse": 9351.658488055098,
      "cv_std": 3052.9980066478593,
      "test_rmse": 8401.503047799048,
      "test_r2": 0.9448074771557228,
      "test_mae": 6722.94233627355,
      "train_rmse": 9217.564101648122,
      "train_r2": 0.9481129786648369,
      "intercept": 109446.44725,
      "coefficients": {
        "R&D Spend": 35570.404417074045,
        "Administration": 0.0,
        "Marketing Spend": 3160.1040457740082,
        "State_Florida": -0.0,
        "State_New York": 0.0
      },
      "non_zero_features": [
        "R&D Spend",
        "Marketing Spend"
      ],
      "test_predictions": [
        177690.6931653048,
        167893.28467945175,
        133433.13206041275,
        133004.21547147082,
        113831.4588888359,
        113654.66475436548,
        102974.92270061925,
        98423.71969722505,
        69405.73764658609,
        74774.953612749
      ]
    },
    "lasso_ols": {
      "name": "Lasso + OLS (L1 篩選重訓)",
      "progression": [
        {
          "k": 1,
          "features": [
            "R&D Spend"
          ],
          "cv_rmse": 9625.473491612547,
          "cv_std": 2650.077088741687,
          "test_rmse": 8274.86801822599,
          "test_r2": 0.9464587607787219,
          "test_mae": 6772.453280477899,
          "test_mse": 68473440.71905932,
          "train_rmse": 9493.580597976612,
          "train_r2": 0.9449589778363044,
          "train_adj_r2": 0.9435105298846282,
          "train_f_stat": 652.3941552354449,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 48416.297661385026,
          "coefficients": {
            "R&D Spend": 0.8516227998779371
          },
          "test_predictions": [
            179090.58602508352,
            169438.148435391,
            135207.80019516923,
            134150.8341057847,
            111431.75202432103,
            109824.77386586336,
            104667.27805997859,
            100481.43277138655,
            65644.27773756774,
            72170.54428855634
          ]
        },
        {
          "k": 2,
          "features": [
            "R&D Spend",
            "Marketing Spend"
          ],
          "cv_rmse": 9474.318466367124,
          "cv_std": 2513.081772226985,
          "test_rmse": 8198.797190788491,
          "test_r2": 0.9474386447268488,
          "test_mae": 6886.594588246512,
          "test_mse": 67220275.37568125,
          "train_rmse": 9078.617918486692,
          "train_r2": 0.9496654854268443,
          "train_adj_r2": 0.9469447008553223,
          "train_f_stat": 349.04104329569407,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 46346.33876123658,
          "coefficients": {
            "R&D Spend": 0.7886182999424612,
            "Marketing Spend": 0.03257662731702063
          },
          "test_predictions": [
            180642.25299736098,
            170343.31979498419,
            134182.14951650103,
            133873.92383811908,
            114580.9213645215,
            114717.24903894302,
            102284.64605182887,
            97433.45922275007,
            68335.07575312015,
            73701.10693630343
          ]
        },
        {
          "k": 3,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "Administration"
          ],
          "cv_rmse": 9613.456019570289,
          "cv_std": 2542.5912427128314,
          "test_rmse": 8803.775790469346,
          "test_r2": 0.939395591782057,
          "test_mae": 7320.441614848128,
          "test_mse": 77506468.16885416,
          "train_rmse": 9052.267248800474,
          "train_r2": 0.9499572530324031,
          "train_adj_r2": 0.9457870241184366,
          "train_f_stat": 227.79498982679945,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42989.00816508666,
          "coefficients": {
            "R&D Spend": 0.7788410440821668,
            "Marketing Spend": 0.034710254869420344,
            "Administration": 0.029391899097552594
          },
          "test_predictions": [
            179627.92567223712,
            169064.0140879458,
            133567.90370043818,
            132763.05993125628,
            114756.1155522112,
            115166.64864795282,
            103901.89696960003,
            98154.80686776245,
            67113.57690570194,
            72911.78976735752
          ]
        },
        {
          "k": 4,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "Administration",
            "State_Florida"
          ],
          "cv_rmse": 9840.958130022287,
          "cv_std": 2905.6548924399794,
          "test_rmse": 9068.54118486935,
          "test_r2": 0.9356955353592089,
          "test_mae": 7431.473270412908,
          "test_mse": 82238439.22167158,
          "train_rmse": 9036.510816272257,
          "train_r2": 0.9501313108490097,
          "train_adj_r2": 0.9444320320888965,
          "train_f_stat": 166.71079812740052,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42922.43234093205,
          "coefficients": {
            "R&D Spend": 0.7753799289895307,
            "Marketing Spend": 0.036327059535269636,
            "Administration": 0.03187042324979702,
            "State_Florida": -1272.160786779386
          },
          "test_predictions": [
            178668.35249140536,
            168051.9671369675,
            132521.0528561006,
            132979.3402644851,
            113956.00325905217,
            115738.44472844573,
            102978.26243868438,
            98397.73803877176,
            67439.97515767504,
            71952.3872507576
          ]
        },
        {
          "k": 5,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "Administration",
            "State_Florida",
            "State_New York"
          ],
          "cv_rmse": 10042.095066710595,
          "cv_std": 2850.483887818726,
          "test_rmse": 9137.990152794957,
          "test_r2": 0.9347068473282423,
          "test_mae": 7514.293659640612,
          "test_mse": 83502864.0325776,
          "train_rmse": 9031.66661257897,
          "train_r2": 0.9501847627493607,
          "train_adj_r2": 0.9428589925654431,
          "train_f_stat": 129.70441863373296,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42554.1676177324,
          "coefficients": {
            "R&D Spend": 0.7734671927327996,
            "Marketing Spend": 0.036610025863856144,
            "Administration": 0.032884597536255944,
            "State_Florida": -959.2841600571081,
            "State_New York": 699.3690525192566
          },
          "test_predictions": [
            178537.4822105506,
            167921.0656955042,
            132447.73845174478,
            132582.27760815562,
            113969.43533012443,
            116161.242301651,
            103015.2015979618,
            98791.7337468745,
            67851.69209675625,
            71976.09851257897
          ]
        }
      ],
      "selected_model": {
        "k": 2,
        "features": [
          "R&D Spend",
          "Marketing Spend"
        ],
        "cv_rmse": 9474.318466367124,
        "cv_std": 2513.081772226985,
        "test_rmse": 8198.797190788491,
        "test_r2": 0.9474386447268488,
        "test_mae": 6886.594588246512,
        "test_mse": 67220275.37568125,
        "train_rmse": 9078.617918486692,
        "train_r2": 0.9496654854268443,
        "train_adj_r2": 0.9469447008553223,
        "train_f_stat": 349.04104329569407,
        "train_f_pvalue": 1.1102230246251565e-16,
        "intercept": 46346.33876123658,
        "coefficients": {
          "R&D Spend": 0.7886182999424612,
          "Marketing Spend": 0.03257662731702063
        },
        "test_predictions": [
          180642.25299736098,
          170343.31979498419,
          134182.14951650103,
          133873.92383811908,
          114580.9213645215,
          114717.24903894302,
          102284.64605182887,
          97433.45922275007,
          68335.07575312015,
          73701.10693630343
        ],
        "algorithm": "Lasso + OLS",
        "description": "以 Lasso 篩選非零特徵後，重新擬合無偏 OLS 模型",
        "best_alpha": 1480.9856485464968
      }
    },
    "rfe": {
      "name": "RFE (遞迴消除法)",
      "progression": [
        {
          "k": 1,
          "features": [
            "R&D Spend"
          ],
          "cv_rmse": 9625.473491612547,
          "cv_std": 2650.077088741687,
          "test_rmse": 8274.86801822599,
          "test_r2": 0.9464587607787219,
          "test_mae": 6772.453280477899,
          "test_mse": 68473440.71905932,
          "train_rmse": 9493.580597976612,
          "train_r2": 0.9449589778363044,
          "train_adj_r2": 0.9435105298846282,
          "train_f_stat": 652.3941552354449,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 48416.297661385026,
          "coefficients": {
            "R&D Spend": 0.8516227998779371
          },
          "test_predictions": [
            179090.58602508352,
            169438.148435391,
            135207.80019516923,
            134150.8341057847,
            111431.75202432103,
            109824.77386586336,
            104667.27805997859,
            100481.43277138655,
            65644.27773756774,
            72170.54428855634
          ]
        },
        {
          "k": 2,
          "features": [
            "R&D Spend",
            "Marketing Spend"
          ],
          "cv_rmse": 9474.318466367124,
          "cv_std": 2513.081772226985,
          "test_rmse": 8198.797190788491,
          "test_r2": 0.9474386447268488,
          "test_mae": 6886.594588246512,
          "test_mse": 67220275.37568125,
          "train_rmse": 9078.617918486692,
          "train_r2": 0.9496654854268443,
          "train_adj_r2": 0.9469447008553223,
          "train_f_stat": 349.04104329569407,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 46346.33876123658,
          "coefficients": {
            "R&D Spend": 0.7886182999424612,
            "Marketing Spend": 0.03257662731702063
          },
          "test_predictions": [
            180642.25299736098,
            170343.31979498419,
            134182.14951650103,
            133873.92383811908,
            114580.9213645215,
            114717.24903894302,
            102284.64605182887,
            97433.45922275007,
            68335.07575312015,
            73701.10693630343
          ]
        },
        {
          "k": 3,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "Administration"
          ],
          "cv_rmse": 9613.456019570289,
          "cv_std": 2542.5912427128314,
          "test_rmse": 8803.775790469346,
          "test_r2": 0.939395591782057,
          "test_mae": 7320.441614848128,
          "test_mse": 77506468.16885416,
          "train_rmse": 9052.267248800474,
          "train_r2": 0.9499572530324031,
          "train_adj_r2": 0.9457870241184366,
          "train_f_stat": 227.79498982679945,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42989.00816508666,
          "coefficients": {
            "R&D Spend": 0.7788410440821668,
            "Marketing Spend": 0.034710254869420344,
            "Administration": 0.029391899097552594
          },
          "test_predictions": [
            179627.92567223712,
            169064.0140879458,
            133567.90370043818,
            132763.05993125628,
            114756.1155522112,
            115166.64864795282,
            103901.89696960003,
            98154.80686776245,
            67113.57690570194,
            72911.78976735752
          ]
        },
        {
          "k": 4,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "Administration",
            "State_Florida"
          ],
          "cv_rmse": 9840.958130022287,
          "cv_std": 2905.6548924399794,
          "test_rmse": 9068.54118486935,
          "test_r2": 0.9356955353592089,
          "test_mae": 7431.473270412908,
          "test_mse": 82238439.22167158,
          "train_rmse": 9036.510816272257,
          "train_r2": 0.9501313108490097,
          "train_adj_r2": 0.9444320320888965,
          "train_f_stat": 166.71079812740052,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42922.43234093205,
          "coefficients": {
            "R&D Spend": 0.7753799289895307,
            "Marketing Spend": 0.036327059535269636,
            "Administration": 0.03187042324979702,
            "State_Florida": -1272.160786779386
          },
          "test_predictions": [
            178668.35249140536,
            168051.9671369675,
            132521.0528561006,
            132979.3402644851,
            113956.00325905217,
            115738.44472844573,
            102978.26243868438,
            98397.73803877176,
            67439.97515767504,
            71952.3872507576
          ]
        },
        {
          "k": 5,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "Administration",
            "State_Florida",
            "State_New York"
          ],
          "cv_rmse": 10042.095066710595,
          "cv_std": 2850.483887818726,
          "test_rmse": 9137.990152794957,
          "test_r2": 0.9347068473282423,
          "test_mae": 7514.293659640612,
          "test_mse": 83502864.0325776,
          "train_rmse": 9031.66661257897,
          "train_r2": 0.9501847627493607,
          "train_adj_r2": 0.9428589925654431,
          "train_f_stat": 129.70441863373296,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42554.1676177324,
          "coefficients": {
            "R&D Spend": 0.7734671927327996,
            "Marketing Spend": 0.036610025863856144,
            "Administration": 0.032884597536255944,
            "State_Florida": -959.2841600571081,
            "State_New York": 699.3690525192566
          },
          "test_predictions": [
            178537.4822105506,
            167921.0656955042,
            132447.73845174478,
            132582.27760815562,
            113969.43533012443,
            116161.242301651,
            103015.2015979618,
            98791.7337468745,
            67851.69209675625,
            71976.09851257897
          ]
        }
      ],
      "best_k": 2
    },
    "elasticnet_direct": {
      "algorithm": "ElasticNet Direct",
      "best_alpha": 78.67255775647456,
      "l1_ratio": 0.5,
      "test_rmse": 36807.83190594249,
      "test_r2": -0.05936773455109745,
      "train_rmse": 38857.022768012626,
      "train_r2": 0.0779266913971326,
      "coefficients": {
        "R&D Spend": 957.7323942407555,
        "Administration": 323.0533084682317,
        "Marketing Spend": 734.2152992988912,
        "State_Florida": 12.898997379869606,
        "State_New York": 148.4762056746322
      },
      "test_predictions": [
        112001.72539148603,
        111396.61397693529,
        109988.04481288057,
        109820.28103145242,
        110002.28316172866,
        110604.61897060435,
        109483.41230278122,
        109111.33747219,
        107757.67047399619,
        107745.54531033563
      ]
    },
    "elasticnet_ols": {
      "name": "ElasticNet + OLS (彈性網絡排序重訓)",
      "progression": [
        {
          "k": 1,
          "features": [
            "R&D Spend"
          ],
          "cv_rmse": 9625.473491612547,
          "cv_std": 2650.077088741687,
          "test_rmse": 8274.86801822599,
          "test_r2": 0.9464587607787219,
          "test_mae": 6772.453280477899,
          "test_mse": 68473440.71905932,
          "train_rmse": 9493.580597976612,
          "train_r2": 0.9449589778363044,
          "train_adj_r2": 0.9435105298846282,
          "train_f_stat": 652.3941552354449,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 48416.297661385026,
          "coefficients": {
            "R&D Spend": 0.8516227998779371
          },
          "test_predictions": [
            179090.58602508352,
            169438.148435391,
            135207.80019516923,
            134150.8341057847,
            111431.75202432103,
            109824.77386586336,
            104667.27805997859,
            100481.43277138655,
            65644.27773756774,
            72170.54428855634
          ]
        },
        {
          "k": 2,
          "features": [
            "R&D Spend",
            "Marketing Spend"
          ],
          "cv_rmse": 9474.318466367124,
          "cv_std": 2513.081772226985,
          "test_rmse": 8198.797190788491,
          "test_r2": 0.9474386447268488,
          "test_mae": 6886.594588246512,
          "test_mse": 67220275.37568125,
          "train_rmse": 9078.617918486692,
          "train_r2": 0.9496654854268443,
          "train_adj_r2": 0.9469447008553223,
          "train_f_stat": 349.04104329569407,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 46346.33876123658,
          "coefficients": {
            "R&D Spend": 0.7886182999424612,
            "Marketing Spend": 0.03257662731702063
          },
          "test_predictions": [
            180642.25299736098,
            170343.31979498419,
            134182.14951650103,
            133873.92383811908,
            114580.9213645215,
            114717.24903894302,
            102284.64605182887,
            97433.45922275007,
            68335.07575312015,
            73701.10693630343
          ]
        },
        {
          "k": 3,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "Administration"
          ],
          "cv_rmse": 9613.456019570289,
          "cv_std": 2542.5912427128314,
          "test_rmse": 8803.775790469346,
          "test_r2": 0.939395591782057,
          "test_mae": 7320.441614848128,
          "test_mse": 77506468.16885416,
          "train_rmse": 9052.267248800474,
          "train_r2": 0.9499572530324031,
          "train_adj_r2": 0.9457870241184366,
          "train_f_stat": 227.79498982679945,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42989.00816508666,
          "coefficients": {
            "R&D Spend": 0.7788410440821668,
            "Marketing Spend": 0.034710254869420344,
            "Administration": 0.029391899097552594
          },
          "test_predictions": [
            179627.92567223712,
            169064.0140879458,
            133567.90370043818,
            132763.05993125628,
            114756.1155522112,
            115166.64864795282,
            103901.89696960003,
            98154.80686776245,
            67113.57690570194,
            72911.78976735752
          ]
        },
        {
          "k": 4,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "Administration",
            "State_New York"
          ],
          "cv_rmse": 9817.137756709979,
          "cv_std": 2475.2856297976987,
          "test_rmse": 8997.199285980298,
          "test_r2": 0.9367033175958571,
          "test_mae": 7474.740626347193,
          "test_mse": 80949594.99164437,
          "train_rmse": 9039.1589552431,
          "train_r2": 0.9501020786438475,
          "train_adj_r2": 0.9443994590602872,
          "train_f_stat": 166.60800614911008,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42410.30908405062,
          "coefficients": {
            "R&D Spend": 0.7746825805130203,
            "Marketing Spend": 0.03573424379441179,
            "Administration": 0.03183486825464469,
            "State_New York": 1052.2198800537271
          },
          "test_predictions": [
            179075.96157197983,
            168492.58649269515,
            133070.23906352918,
            132245.69745454332,
            114480.26282334968,
            116014.33808117366,
            103615.70496702472,
            98837.47482909175,
            67853.79186123735,
            72592.46097857336
          ]
        },
        {
          "k": 5,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "Administration",
            "State_New York",
            "State_Florida"
          ],
          "cv_rmse": 10042.095066710597,
          "cv_std": 2850.483887818726,
          "test_rmse": 9137.990152794957,
          "test_r2": 0.9347068473282423,
          "test_mae": 7514.293659640612,
          "test_mse": 83502864.0325776,
          "train_rmse": 9031.66661257897,
          "train_r2": 0.9501847627493607,
          "train_adj_r2": 0.9428589925654431,
          "train_f_stat": 129.70441863373296,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42554.1676177324,
          "coefficients": {
            "R&D Spend": 0.7734671927327996,
            "Marketing Spend": 0.036610025863856144,
            "Administration": 0.032884597536255944,
            "State_New York": 699.3690525192562,
            "State_Florida": -959.2841600571085
          },
          "test_predictions": [
            178537.4822105506,
            167921.0656955042,
            132447.73845174478,
            132582.27760815562,
            113969.43533012443,
            116161.242301651,
            103015.2015979618,
            98791.7337468745,
            67851.69209675625,
            71976.09851257897
          ]
        }
      ],
      "best_k": 2
    },
    "exhaustive_search": {
      "name": "Exhaustive Search (全子集 31 組窮舉)",
      "best_per_k": [
        {
          "k": 1,
          "features": [
            "R&D Spend"
          ],
          "cv_rmse": 9625.473491612547,
          "cv_std": 2650.077088741687,
          "test_rmse": 8274.86801822599,
          "test_r2": 0.9464587607787219,
          "test_mae": 6772.453280477899,
          "test_mse": 68473440.71905932,
          "train_rmse": 9493.580597976612,
          "train_r2": 0.9449589778363044,
          "train_adj_r2": 0.9435105298846282,
          "train_f_stat": 652.3941552354449,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 48416.297661385026,
          "coefficients": {
            "R&D Spend": 0.8516227998779371
          },
          "test_predictions": [
            179090.58602508352,
            169438.148435391,
            135207.80019516923,
            134150.8341057847,
            111431.75202432103,
            109824.77386586336,
            104667.27805997859,
            100481.43277138655,
            65644.27773756774,
            72170.54428855634
          ]
        },
        {
          "k": 2,
          "features": [
            "R&D Spend",
            "Marketing Spend"
          ],
          "cv_rmse": 9474.318466367124,
          "cv_std": 2513.081772226985,
          "test_rmse": 8198.797190788491,
          "test_r2": 0.9474386447268488,
          "test_mae": 6886.594588246512,
          "test_mse": 67220275.37568125,
          "train_rmse": 9078.617918486692,
          "train_r2": 0.9496654854268443,
          "train_adj_r2": 0.9469447008553223,
          "train_f_stat": 349.04104329569407,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 46346.33876123658,
          "coefficients": {
            "R&D Spend": 0.7886182999424612,
            "Marketing Spend": 0.03257662731702063
          },
          "test_predictions": [
            180642.25299736098,
            170343.31979498419,
            134182.14951650103,
            133873.92383811908,
            114580.9213645215,
            114717.24903894302,
            102284.64605182887,
            97433.45922275007,
            68335.07575312015,
            73701.10693630343
          ]
        },
        {
          "k": 3,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "State_New York"
          ],
          "cv_rmse": 9602.033536091889,
          "cv_std": 2455.890438310911,
          "test_rmse": 8309.059683481717,
          "test_r2": 0.9460153830071389,
          "test_mae": 7028.435331818949,
          "test_mse": 69040472.8236613,
          "train_rmse": 9069.684820873443,
          "train_r2": 0.9497644921078413,
          "train_adj_r2": 0.9455781997834948,
          "train_f_stat": 226.87486169664254,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 46100.29130551412,
          "coefficients": {
            "R&D Spend": 0.7858709815066743,
            "Marketing Spend": 0.03327182941032258,
            "State_New York": 864.0018700527295
          },
          "test_predictions": [
            180258.24979853872,
            169961.41927630414,
            133815.4275493898,
            133524.92138467255,
            114342.45547445389,
            115382.63503846529,
            101939.27091345853,
            97944.78203137245,
            69026.24972723081,
            73492.7691291159
          ]
        },
        {
          "k": 4,
          "features": [
            "R&D Spend",
            "Marketing Spend",
            "State_Florida",
            "State_New York"
          ],
          "cv_rmse": 9779.97303784842,
          "cv_std": 2721.8537290554013,
          "test_rmse": 8409.91671448805,
          "test_r2": 0.9446968769428332,
          "test_mae": 7109.161997627585,
          "test_mse": 70726699.14462548,
          "train_rmse": 9064.118245163729,
          "train_r2": 0.9498261378977877,
          "train_adj_r2": 0.9440919822289634,
          "train_f_stat": 165.64359127218114,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 46329.0601785425,
          "coefficients": {
            "R&D Spend": 0.7851417173112168,
            "Marketing Spend": 0.03395640234030121,
            "State_Florida": -826.4681587619028,
            "State_New York": 554.6573328267139
          },
          "test_predictions": [
            179827.91205675257,
            169510.75518374197,
            133300.28408634168,
            133851.2421601261,
            113898.43875999091,
            115491.25395900002,
            101374.28367857916,
            97880.01360638958,
            69057.74881968717,
            72987.32073135999
          ]
        },
        {
          "k": 5,
          "features": [
            "R&D Spend",
            "Administration",
            "Marketing Spend",
            "State_Florida",
            "State_New York"
          ],
          "cv_rmse": 10042.09506671059,
          "cv_std": 2850.483887818729,
          "test_rmse": 9137.99015279496,
          "test_r2": 0.9347068473282423,
          "test_mae": 7514.293659640617,
          "test_mse": 83502864.03257768,
          "train_rmse": 9031.66661257897,
          "train_r2": 0.9501847627493607,
          "train_adj_r2": 0.9428589925654431,
          "train_f_stat": 129.70441863373296,
          "train_f_pvalue": 1.1102230246251565e-16,
          "intercept": 42554.16761773235,
          "coefficients": {
            "R&D Spend": 0.7734671927327997,
            "Administration": 0.0328845975362561,
            "Marketing Spend": 0.03661002586385624,
            "State_Florida": -959.2841600571066,
            "State_New York": 699.3690525192587
          },
          "test_predictions": [
            178537.48221055063,
            167921.0656955042,
            132447.73845174478,
            132582.27760815562,
            113969.43533012444,
            116161.24230165104,
            103015.2015979618,
            98791.7337468745,
            67851.69209675623,
            71976.09851257896
          ]
        }
      ],
      "overall_best": {
        "k": 2,
        "features": [
          "R&D Spend",
          "Marketing Spend"
        ],
        "cv_rmse": 9474.318466367124,
        "cv_std": 2513.081772226985,
        "test_rmse": 8198.797190788491,
        "test_r2": 0.9474386447268488,
        "test_mae": 6886.594588246512,
        "test_mse": 67220275.37568125,
        "train_rmse": 9078.617918486692,
        "train_r2": 0.9496654854268443,
        "train_adj_r2": 0.9469447008553223,
        "train_f_stat": 349.04104329569407,
        "train_f_pvalue": 1.1102230246251565e-16,
        "intercept": 46346.33876123658,
        "coefficients": {
          "R&D Spend": 0.7886182999424612,
          "Marketing Spend": 0.03257662731702063
        },
        "test_predictions": [
          180642.25299736098,
          170343.31979498419,
          134182.14951650103,
          133873.92383811908,
          114580.9213645215,
          114717.24903894302,
          102284.64605182887,
          97433.45922275007,
          68335.07575312015,
          73701.10693630343
        ]
      }
    }
  },
  "algorithm_descriptions": {
    "sfs": "從空集合開始，每輪依 5-Fold CV RMSE 最小者加入一個候選特徵，直到全部納入。",
    "lasso_direct": "以 L1 懲罰項將不重要特徵權重收縮至 0，以縮減係數直接預測。",
    "lasso_ols": "利用 LassoCV 挑選出非零係數特徵，再使用標準 OLS 擬合無偏模型，兼顧稀疏性與估計不偏性。",
    "rfe": "以線性模型權重遞迴剔除最不重要特徵，依留存順序評估 top-k 表現。",
    "elasticnet_ols": "同時結合 L1 與 L2 正則化權重排序，再以 OLS 擬合最佳子集。",
    "exhaustive_search": "窮舉遍歷全部 31 種可能特徵組合，以 5-Fold CV 評估找出全域最優解。"
  },
  "exhaustive_search": {
    "all_31_subsets": [
      {
        "features": "R&D Spend",
        "k": 1,
        "cv_rmse": 9625.473491612547,
        "cv_std": 2650.077088741687,
        "test_rmse": 8274.86801822599,
        "test_r2": 0.946458760778722,
        "train_rmse": 9493.580597976612,
        "train_r2": 0.9449589778363044
      },
      {
        "features": "Administration",
        "k": 1,
        "cv_rmse": 38425.482678701504,
        "cv_std": 7791.430525886112,
        "test_rmse": 45669.70577941752,
        "test_r2": -0.6308825840983827,
        "train_rmse": 38162.178493008825,
        "train_r2": 0.110609013129954
      },
      {
        "features": "Marketing Spend",
        "k": 1,
        "cv_rmse": 27220.692982901113,
        "cv_std": 7961.149714490011,
        "test_rmse": 25529.053887870352,
        "test_r2": 0.4903926212546319,
        "train_rmse": 26731.776740752688,
        "train_r2": 0.5636026973161414
      },
      {
        "features": "State_Florida",
        "k": 1,
        "cv_rmse": 41183.590005815626,
        "cv_std": 8014.086815678363,
        "test_rmse": 37588.05292629757,
        "test_r2": -0.1047548756438994,
        "train_rmse": 40460.57672769297,
        "train_r2": 0.0002519980153838
      },
      {
        "features": "State_New York",
        "k": 1,
        "cv_rmse": 41910.27487159964,
        "cv_std": 6033.738641626326,
        "test_rmse": 41366.02329593475,
        "test_r2": -0.337992909392512,
        "train_rmse": 39991.13006785038,
        "train_r2": 0.0233167039814222
      },
      {
        "features": "R&D Spend, Administration",
        "k": 2,
        "cv_rmse": 9701.569919854584,
        "cv_std": 2695.568462847259,
        "test_rmse": 8237.540194485355,
        "test_r2": 0.9469407189577184,
        "train_rmse": 9493.505441460638,
        "train_r2": 0.944959849304144
      },
      {
        "features": "R&D Spend, Marketing Spend",
        "k": 2,
        "cv_rmse": 9474.318466367124,
        "cv_std": 2513.081772226985,
        "test_rmse": 8198.797190788491,
        "test_r2": 0.9474386447268488,
        "train_rmse": 9078.617918486692,
        "train_r2": 0.9496654854268444
      },
      {
        "features": "R&D Spend, State_Florida",
        "k": 2,
        "cv_rmse": 9845.925137646074,
        "cv_std": 2717.436813825152,
        "test_rmse": 8217.446447873415,
        "test_r2": 0.947199257159004,
        "train_rmse": 9492.00718052055,
        "train_r2": 0.944977220759014
      },
      {
        "features": "R&D Spend, State_New York",
        "k": 2,
        "cv_rmse": 9802.050579444973,
        "cv_std": 2563.312056637042,
        "test_rmse": 8273.828202485938,
        "test_r2": 0.9464722158635016,
        "train_rmse": 9493.578848073585,
        "train_r2": 0.9449589981271612
      },
      {
        "features": "Administration, Marketing Spend",
        "k": 2,
        "cv_rmse": 24538.53703660446,
        "cv_std": 8015.609645767953,
        "test_rmse": 28044.339874916634,
        "test_r2": 0.3850260621597938,
        "train_rmse": 24378.71213505458,
        "train_r2": 0.6370490600729561
      },
      {
        "features": "Administration, State_Florida",
        "k": 2,
        "cv_rmse": 39448.4723547756,
        "cv_std": 8449.669726093853,
        "test_rmse": 45664.32281716461,
        "test_r2": -0.6304981514920294,
        "train_rmse": 38162.17753694892,
        "train_r2": 0.1106090576929788
      },
      {
        "features": "Administration, State_New York",
        "k": 2,
        "cv_rmse": 39506.89665039866,
        "cv_std": 5549.9359888355675,
        "test_rmse": 48868.222680884006,
        "test_r2": -0.8673226104964413,
        "train_rmse": 37623.20120182793,
        "train_r2": 0.1355539440264918
      },
      {
        "features": "Marketing Spend, State_Florida",
        "k": 2,
        "cv_rmse": 26861.931565008235,
        "cv_std": 8117.633827697809,
        "test_rmse": 27724.528517063183,
        "test_r2": 0.3989721367324698,
        "train_rmse": 26377.37401261263,
        "train_r2": 0.5750972697191309
      },
      {
        "features": "Marketing Spend, State_New York",
        "k": 2,
        "cv_rmse": 27311.329757722087,
        "cv_std": 6528.915193391372,
        "test_rmse": 28603.153690373812,
        "test_r2": 0.3602738453272684,
        "train_rmse": 26108.67531724467,
        "train_r2": 0.5837099008848237
      },
      {
        "features": "State_Florida, State_New York",
        "k": 2,
        "cv_rmse": 42501.52846598824,
        "cv_std": 5999.7256322786725,
        "test_rmse": 39655.93042301208,
        "test_r2": -0.2296529481611711,
        "train_rmse": 39828.84274386977,
        "train_r2": 0.0312275436306205
      },
      {
        "features": "R&D Spend, Administration, Marketing Spend",
        "k": 3,
        "cv_rmse": 9613.456019570294,
        "cv_std": 2542.591242712824,
        "test_rmse": 8803.775790469343,
        "test_r2": 0.939395591782057,
        "train_rmse": 9052.267248800474,
        "train_r2": 0.9499572530324032
      },
      {
        "features": "R&D Spend, Administration, State_Florida",
        "k": 3,
        "cv_rmse": 9941.932305948754,
        "cv_std": 2791.641093602729,
        "test_rmse": 8171.645989404477,
        "test_r2": 0.9477861934963948,
        "train_rmse": 9491.896989969053,
        "train_r2": 0.9449784982454472
      },
      {
        "features": "R&D Spend, Administration, State_New York",
        "k": 3,
        "cv_rmse": 9882.563780726914,
        "cv_std": 2602.259810667186,
        "test_rmse": 8235.348504978972,
        "test_r2": 0.9469689492295412,
        "train_rmse": 9493.501524194697,
        "train_r2": 0.9449598947261147
      },
      {
        "features": "R&D Spend, Marketing Spend, State_Florida",
        "k": 3,
        "cv_rmse": 9634.198709549211,
        "cv_std": 2752.70072632981,
        "test_rmse": 8376.45051976933,
        "test_r2": 0.9451361446943416,
        "train_rmse": 9067.174045517751,
        "train_r2": 0.9497923018100066
      },
      {
        "features": "R&D Spend, Marketing Spend, State_New York",
        "k": 3,
        "cv_rmse": 9602.033536091889,
        "cv_std": 2455.890438310911,
        "test_rmse": 8309.059683481717,
        "test_r2": 0.9460153830071388,
        "train_rmse": 9069.684820873445,
        "train_r2": 0.9497644921078412
      },
      {
        "features": "R&D Spend, State_Florida, State_New York",
        "k": 3,
        "cv_rmse": 10011.13359629165,
        "cv_std": 2707.7199556490195,
        "test_rmse": 8219.806004927406,
        "test_r2": 0.9471689304016888,
        "train_rmse": 9491.70173048498,
        "train_r2": 0.9449807619361624
      },
      {
        "features": "Administration, Marketing Spend, State_Florida",
        "k": 3,
        "cv_rmse": 24315.92050050181,
        "cv_std": 8843.456131759385,
        "test_rmse": 30607.437371603213,
        "test_r2": 0.267478785862692,
        "train_rmse": 23918.22663952711,
        "train_r2": 0.6506310050515545
      },
      {
        "features": "Administration, Marketing Spend, State_New York",
        "k": 3,
        "cv_rmse": 25145.034342362094,
        "cv_std": 6691.25626486456,
        "test_rmse": 31294.430191992647,
        "test_r2": 0.2342264454629645,
        "train_rmse": 23647.743783205977,
        "train_r2": 0.6584881092458313
      },
      {
        "features": "Administration, State_Florida, State_New York",
        "k": 3,
        "cv_rmse": 40398.108225519216,
        "cv_std": 6384.781567764413,
        "test_rmse": 47389.80249808112,
        "test_r2": -0.7560467150570473,
        "train_rmse": 37503.664142634785,
        "train_r2": 0.1410382824063362
      },
      {
        "features": "Marketing Spend, State_Florida, State_New York",
        "k": 3,
        "cv_rmse": 27591.873372523503,
        "cv_std": 6403.645683140527,
        "test_rmse": 29299.114567384466,
        "test_r2": 0.3287639709813525,
        "train_rmse": 26027.030220460783,
        "train_r2": 0.5863094125477661
      },
      {
        "features": "R&D Spend, Administration, Marketing Spend, State_Florida",
        "k": 4,
        "cv_rmse": 9840.958130022287,
        "cv_std": 2905.654892439978,
        "test_rmse": 9068.541184869351,
        "test_r2": 0.9356955353592088,
        "train_rmse": 9036.510816272255,
        "train_r2": 0.9501313108490096
      },
      {
        "features": "R&D Spend, Administration, Marketing Spend, State_New York",
        "k": 4,
        "cv_rmse": 9817.137756709984,
        "cv_std": 2475.2856297976987,
        "test_rmse": 8997.199285980296,
        "test_r2": 0.9367033175958572,
        "train_rmse": 9039.1589552431,
        "train_r2": 0.9501020786438474
      },
      {
        "features": "R&D Spend, Administration, State_Florida, State_New York",
        "k": 4,
        "cv_rmse": 10112.65884654129,
        "cv_std": 2771.413162209627,
        "test_rmse": 8178.369214854917,
        "test_r2": 0.9477002402858664,
        "train_rmse": 9491.612574571716,
        "train_r2": 0.9449817955268752
      },
      {
        "features": "R&D Spend, Marketing Spend, State_Florida, State_New York",
        "k": 4,
        "cv_rmse": 9779.97303784842,
        "cv_std": 2721.8537290554013,
        "test_rmse": 8409.91671448805,
        "test_r2": 0.9446968769428332,
        "train_rmse": 9064.118245163729,
        "train_r2": 0.9498261378977876
      },
      {
        "features": "Administration, Marketing Spend, State_Florida, State_New York",
        "k": 4,
        "cv_rmse": 25676.10915632825,
        "cv_std": 6926.579118111535,
        "test_rmse": 32175.96174485506,
        "test_r2": 0.1904767222878428,
        "train_rmse": 23526.295233951423,
        "train_r2": 0.6619869308561408
      },
      {
        "features": "R&D Spend, Administration, Marketing Spend, State_Florida, State_New York",
        "k": 5,
        "cv_rmse": 10042.09506671059,
        "cv_std": 2850.483887818729,
        "test_rmse": 9137.99015279496,
        "test_r2": 0.9347068473282424,
        "train_rmse": 9031.66661257897,
        "train_r2": 0.9501847627493608
      }
    ],
    "best_per_k": [
      {
        "k": 1,
        "features": [
          "R&D Spend"
        ],
        "cv_rmse": 9625.473491612547,
        "cv_std": 2650.077088741687,
        "test_rmse": 8274.86801822599,
        "test_r2": 0.9464587607787219,
        "test_mae": 6772.453280477899,
        "test_mse": 68473440.71905932,
        "train_rmse": 9493.580597976612,
        "train_r2": 0.9449589778363044,
        "train_adj_r2": 0.9435105298846282,
        "train_f_stat": 652.3941552354449,
        "train_f_pvalue": 1.1102230246251565e-16,
        "intercept": 48416.297661385026,
        "coefficients": {
          "R&D Spend": 0.8516227998779371
        },
        "test_predictions": [
          179090.58602508352,
          169438.148435391,
          135207.80019516923,
          134150.8341057847,
          111431.75202432103,
          109824.77386586336,
          104667.27805997859,
          100481.43277138655,
          65644.27773756774,
          72170.54428855634
        ]
      },
      {
        "k": 2,
        "features": [
          "R&D Spend",
          "Marketing Spend"
        ],
        "cv_rmse": 9474.318466367124,
        "cv_std": 2513.081772226985,
        "test_rmse": 8198.797190788491,
        "test_r2": 0.9474386447268488,
        "test_mae": 6886.594588246512,
        "test_mse": 67220275.37568125,
        "train_rmse": 9078.617918486692,
        "train_r2": 0.9496654854268443,
        "train_adj_r2": 0.9469447008553223,
        "train_f_stat": 349.04104329569407,
        "train_f_pvalue": 1.1102230246251565e-16,
        "intercept": 46346.33876123658,
        "coefficients": {
          "R&D Spend": 0.7886182999424612,
          "Marketing Spend": 0.03257662731702063
        },
        "test_predictions": [
          180642.25299736098,
          170343.31979498419,
          134182.14951650103,
          133873.92383811908,
          114580.9213645215,
          114717.24903894302,
          102284.64605182887,
          97433.45922275007,
          68335.07575312015,
          73701.10693630343
        ]
      },
      {
        "k": 3,
        "features": [
          "R&D Spend",
          "Marketing Spend",
          "State_New York"
        ],
        "cv_rmse": 9602.033536091889,
        "cv_std": 2455.890438310911,
        "test_rmse": 8309.059683481717,
        "test_r2": 0.9460153830071389,
        "test_mae": 7028.435331818949,
        "test_mse": 69040472.8236613,
        "train_rmse": 9069.684820873443,
        "train_r2": 0.9497644921078413,
        "train_adj_r2": 0.9455781997834948,
        "train_f_stat": 226.87486169664254,
        "train_f_pvalue": 1.1102230246251565e-16,
        "intercept": 46100.29130551412,
        "coefficients": {
          "R&D Spend": 0.7858709815066743,
          "Marketing Spend": 0.03327182941032258,
          "State_New York": 864.0018700527295
        },
        "test_predictions": [
          180258.24979853872,
          169961.41927630414,
          133815.4275493898,
          133524.92138467255,
          114342.45547445389,
          115382.63503846529,
          101939.27091345853,
          97944.78203137245,
          69026.24972723081,
          73492.7691291159
        ]
      },
      {
        "k": 4,
        "features": [
          "R&D Spend",
          "Marketing Spend",
          "State_Florida",
          "State_New York"
        ],
        "cv_rmse": 9779.97303784842,
        "cv_std": 2721.8537290554013,
        "test_rmse": 8409.91671448805,
        "test_r2": 0.9446968769428332,
        "test_mae": 7109.161997627585,
        "test_mse": 70726699.14462548,
        "train_rmse": 9064.118245163729,
        "train_r2": 0.9498261378977877,
        "train_adj_r2": 0.9440919822289634,
        "train_f_stat": 165.64359127218114,
        "train_f_pvalue": 1.1102230246251565e-16,
        "intercept": 46329.0601785425,
        "coefficients": {
          "R&D Spend": 0.7851417173112168,
          "Marketing Spend": 0.03395640234030121,
          "State_Florida": -826.4681587619028,
          "State_New York": 554.6573328267139
        },
        "test_predictions": [
          179827.91205675257,
          169510.75518374197,
          133300.28408634168,
          133851.2421601261,
          113898.43875999091,
          115491.25395900002,
          101374.28367857916,
          97880.01360638958,
          69057.74881968717,
          72987.32073135999
        ]
      },
      {
        "k": 5,
        "features": [
          "R&D Spend",
          "Administration",
          "Marketing Spend",
          "State_Florida",
          "State_New York"
        ],
        "cv_rmse": 10042.09506671059,
        "cv_std": 2850.483887818729,
        "test_rmse": 9137.99015279496,
        "test_r2": 0.9347068473282423,
        "test_mae": 7514.293659640617,
        "test_mse": 83502864.03257768,
        "train_rmse": 9031.66661257897,
        "train_r2": 0.9501847627493607,
        "train_adj_r2": 0.9428589925654431,
        "train_f_stat": 129.70441863373296,
        "train_f_pvalue": 1.1102230246251565e-16,
        "intercept": 42554.16761773235,
        "coefficients": {
          "R&D Spend": 0.7734671927327997,
          "Administration": 0.0328845975362561,
          "Marketing Spend": 0.03661002586385624,
          "State_Florida": -959.2841600571066,
          "State_New York": 699.3690525192587
        },
        "test_predictions": [
          178537.48221055063,
          167921.0656955042,
          132447.73845174478,
          132582.27760815562,
          113969.43533012444,
          116161.24230165104,
          103015.2015979618,
          98791.7337468745,
          67851.69209675623,
          71976.09851257896
        ]
      }
    ],
    "overall_best": {
      "k": 2,
      "features": [
        "R&D Spend",
        "Marketing Spend"
      ],
      "cv_rmse": 9474.318466367124,
      "cv_std": 2513.081772226985,
      "test_rmse": 8198.797190788491,
      "test_r2": 0.9474386447268488,
      "test_mae": 6886.594588246512,
      "test_mse": 67220275.37568125,
      "train_rmse": 9078.617918486692,
      "train_r2": 0.9496654854268443,
      "train_adj_r2": 0.9469447008553223,
      "train_f_stat": 349.04104329569407,
      "train_f_pvalue": 1.1102230246251565e-16,
      "intercept": 46346.33876123658,
      "coefficients": {
        "R&D Spend": 0.7886182999424612,
        "Marketing Spend": 0.03257662731702063
      },
      "test_predictions": [
        180642.25299736098,
        170343.31979498419,
        134182.14951650103,
        133873.92383811908,
        114580.9213645215,
        114717.24903894302,
        102284.64605182887,
        97433.45922275007,
        68335.07575312015,
        73701.10693630343
      ]
    }
  },
  "boston_housing": {
    "title": "Topic 2: Boston Housing 波士頓房價多元線性回歸實證 (CRISP-DM Workflow)",
    "description": "經典波士頓房價回歸資料集 (506 筆樣本、13 項特徵、目標變數為住宅中位數價值 MEDV，以千美元計)。",
    "features_info": {
      "crim": "城鎮人均犯罪率",
      "zn": "大面積住宅用地比例",
      "indus": "非零售商業用地比例",
      "chas": "查爾斯河虛擬變數 (邊界為河為 1，臨河溢價約 $2,341)",
      "nox": "一氧化氮濃度 (ppm，空氣污染每增加 1 單位房價降低約 $16,124)",
      "rm": "住宅平均房間數 (每增加 1 間房房價預期提升約 $3,709)",
      "age": "1940 年前建造之自住單位比例",
      "dis": "到五個就業中心之加權距離",
      "rad": "徑向公路通道指數",
      "tax": "每萬美元全值財產稅率",
      "ptratio": "學生與教師比例",
      "b": "城鎮黑人比例相關指數",
      "lstat": "人口中低地位社群比例 (%) (弱勢族群比例每升 1%，房價下降約 $493)"
    },
    "metrics": {
      "train": {
        "mae": 3.1031563965794304,
        "mse": 19.32647020358573,
        "rmse": 4.396188144698282,
        "r2": 0.7730135569264233,
        "adj_r2": 0.765447342157304
      },
      "test": {
        "mae": 3.8429092204444966,
        "mse": 33.448979997676304,
        "rmse": 5.783509315085116,
        "r2": 0.5892223849182535
      },
      "cv": {
        "cv_rmse_mean": 4.704364924730501,
        "cv_rmse_std": 0.7066920236892873
      }
    },
    "formula": "medv = 38.0917 - 0.1194 * (crim) + 0.0448 * (zn) + 0.0055 * (indus) + 2.3408 * (chas) - 16.1236 * (nox) + 3.7087 * (rm) - 0.0031 * (age) - 1.3864 * (dis) + 0.2442 * (rad) - 0.0110 * (tax) - 1.0459 * (ptratio) + 0.0081 * (b) - 0.4928 * (lstat)",
    "coefficients": {
      "crim": -0.11944344700245886,
      "zn": 0.0447799510665102,
      "indus": 0.005485261681773282,
      "chas": 2.3408036062420625,
      "nox": -16.12360431542992,
      "rm": 3.7087090122191633,
      "age": -0.0031210817807303093,
      "dis": -1.3863973702786825,
      "rad": 0.24417832698881214,
      "tax": -0.010989636563080368,
      "ptratio": -1.0459211887459758,
      "b": 0.008110106932705552,
      "lstat": -0.49279272450470607
    },
    "intercept": 38.09169492630195,
    "test_predictions": [
      {
        "original_id": 1,
        "actual": 21.6,
        "predicted": 25.17247952461782,
        "residual": -3.572479524617819,
        "abs_error": 3.572479524617819
      },
      {
        "original_id": 10,
        "actual": 15.0,
        "predicted": 19.78693474702982,
        "residual": -4.786934747029822,
        "abs_error": 4.786934747029822
      },
      {
        "original_id": 12,
        "actual": 21.7,
        "predicted": 21.726730664321636,
        "residual": -0.0267306643216365,
        "abs_error": 0.0267306643216365
      },
      {
        "original_id": 15,
        "actual": 19.9,
        "predicted": 19.28344430480126,
        "residual": 0.6165556951987377,
        "abs_error": 0.6165556951987377
      },
      {
        "original_id": 21,
        "actual": 19.6,
        "predicted": 17.648168648202226,
        "residual": 1.9518313517977752,
        "abs_error": 1.9518313517977752
      },
      {
        "original_id": 37,
        "actual": 21.0,
        "predicted": 23.151400085720876,
        "residual": -2.1514000857208764,
        "abs_error": 2.1514000857208764
      },
      {
        "original_id": 45,
        "actual": 19.3,
        "predicted": 22.419407359009977,
        "residual": -3.119407359009976,
        "abs_error": 3.119407359009976
      },
      {
        "original_id": 46,
        "actual": 20.0,
        "predicted": 20.863046930630368,
        "residual": -0.8630469306303681,
        "abs_error": 0.8630469306303681
      },
      {
        "original_id": 54,
        "actual": 18.9,
        "predicted": 15.835159912029445,
        "residual": 3.0648400879705555,
        "abs_error": 3.0648400879705555
      },
      {
        "original_id": 56,
        "actual": 24.7,
        "predicted": 25.44217132031485,
        "residual": -0.7421713203148492,
        "abs_error": 0.7421713203148492
      },
      {
        "original_id": 60,
        "actual": 18.7,
        "predicted": 17.93719474769467,
        "residual": 0.7628052523053306,
        "abs_error": 0.7628052523053306
      },
      {
        "original_id": 65,
        "actual": 23.5,
        "predicted": 30.728593255913506,
        "residual": -7.228593255913502,
        "abs_error": 7.228593255913502
      },
      {
        "original_id": 75,
        "actual": 21.4,
        "predicted": 24.02555703445435,
        "residual": -2.625557034454353,
        "abs_error": 2.625557034454353
      },
      {
        "original_id": 76,
        "actual": 20.0,
        "predicted": 22.96041622332026,
        "residual": -2.9604162233202587,
        "abs_error": 2.9604162233202587
      },
      {
        "original_id": 78,
        "actual": 21.2,
        "predicted": 21.443822540726412,
        "residual": -0.2438225407264127,
        "abs_error": 0.2438225407264127
      },
      {
        "original_id": 90,
        "actual": 22.6,
        "predicted": 27.26856352117485,
        "residual": -4.66856352117485,
        "abs_error": 4.66856352117485
      },
      {
        "original_id": 96,
        "actual": 21.4,
        "predicted": 24.93076216852115,
        "residual": -3.53076216852115,
        "abs_error": 3.53076216852115
      },
      {
        "original_id": 97,
        "actual": 38.7,
        "predicted": 35.53362541272343,
        "residual": 3.1663745872765716,
        "abs_error": 3.1663745872765716
      },
      {
        "original_id": 100,
        "actual": 27.5,
        "predicted": 24.285151233051643,
        "residual": 3.214848766948357,
        "abs_error": 3.214848766948357
      },
      {
        "original_id": 102,
        "actual": 18.6,
        "predicted": 19.928065796712755,
        "residual": -1.3280657967127532,
        "abs_error": 1.3280657967127532
      },
      {
        "original_id": 107,
        "actual": 20.4,
        "predicted": 20.595437515199784,
        "residual": -0.1954375151997851,
        "abs_error": 0.1954375151997851
      },
      {
        "original_id": 108,
        "actual": 19.8,
        "predicted": 22.373232598576163,
        "residual": -2.573232598576162,
        "abs_error": 2.573232598576162
      },
      {
        "original_id": 113,
        "actual": 18.7,
        "predicted": 20.96084724464472,
        "residual": -2.2608472446447223,
        "abs_error": 2.2608472446447223
      },
      {
        "original_id": 134,
        "actual": 15.6,
        "predicted": 13.426958772411066,
        "residual": 2.173041227588934,
        "abs_error": 2.173041227588934
      },
      {
        "original_id": 137,
        "actual": 17.1,
        "predicted": 19.17978838209381,
        "residual": -2.0797883820938097,
        "abs_error": 2.0797883820938097
      },
      {
        "original_id": 144,
        "actual": 11.8,
        "predicted": 10.03805069832287,
        "residual": 1.7619493016771308,
        "abs_error": 1.7619493016771308
      },
      {
        "original_id": 153,
        "actual": 19.4,
        "predicted": 18.27419652433351,
        "residual": 1.125803475666487,
        "abs_error": 1.125803475666487
      },
      {
        "original_id": 154,
        "actual": 17.0,
        "predicted": 22.90256276384767,
        "residual": -5.902562763847669,
        "abs_error": 5.902562763847669
      },
      {
        "original_id": 155,
        "actual": 15.6,
        "predicted": 20.941277110751596,
        "residual": -5.341277110751596,
        "abs_error": 5.341277110751596
      },
      {
        "original_id": 159,
        "actual": 23.3,
        "predicted": 26.098457513763773,
        "residual": -2.798457513763772,
        "abs_error": 2.798457513763772
      },
      {
        "original_id": 162,
        "actual": 50.0,
        "predicted": 39.996367261026016,
        "residual": 10.003632738973984,
        "abs_error": 10.003632738973984
      },
      {
        "original_id": 171,
        "actual": 19.1,
        "predicted": 24.671877559523907,
        "residual": -5.571877559523905,
        "abs_error": 5.571877559523905
      },
      {
        "original_id": 173,
        "actual": 23.6,
        "predicted": 29.102048740799816,
        "residual": -5.502048740799815,
        "abs_error": 5.502048740799815
      },
      {
        "original_id": 175,
        "actual": 29.4,
        "predicted": 30.864010891945732,
        "residual": -1.4640108919457333,
        "abs_error": 1.4640108919457333
      },
      {
        "original_id": 188,
        "actual": 29.8,
        "predicted": 32.76358680915159,
        "residual": -2.9635868091515896,
        "abs_error": 2.9635868091515896
      },
      {
        "original_id": 196,
        "actual": 33.3,
        "predicted": 36.85192326537387,
        "residual": -3.5519232653738726,
        "abs_error": 3.5519232653738726
      },
      {
        "original_id": 205,
        "actual": 22.6,
        "predicted": 22.898253740333423,
        "residual": -0.298253740333422,
        "abs_error": 0.298253740333422
      },
      {
        "original_id": 206,
        "actual": 24.4,
        "predicted": 23.7726724879158,
        "residual": 0.6273275120841966,
        "abs_error": 0.6273275120841966
      },
      {
        "original_id": 210,
        "actual": 21.7,
        "predicted": 22.1837787446552,
        "residual": -0.4837787446552006,
        "abs_error": 0.4837787446552006
      },
      {
        "original_id": 213,
        "actual": 28.1,
        "predicted": 25.305888439935792,
        "residual": 2.7941115600642092,
        "abs_error": 2.7941115600642092
      },
      {
        "original_id": 219,
        "actual": 23.0,
        "predicted": 29.364998677356255,
        "residual": -6.364998677356258,
        "abs_error": 6.364998677356258
      },
      {
        "original_id": 224,
        "actual": 44.8,
        "predicted": 37.82397661650042,
        "residual": 6.97602338349958,
        "abs_error": 6.97602338349958
      },
      {
        "original_id": 225,
        "actual": 50.0,
        "predicted": 39.2201264742018,
        "residual": 10.779873525798202,
        "abs_error": 10.779873525798202
      },
      {
        "original_id": 229,
        "actual": 31.5,
        "predicted": 31.112356714772645,
        "residual": 0.3876432852273552,
        "abs_error": 0.3876432852273552
      },
      {
        "original_id": 230,
        "actual": 24.3,
        "predicted": 24.522354054923007,
        "residual": -0.2223540549230058,
        "abs_error": 0.2223540549230058
      },
      {
        "original_id": 233,
        "actual": 48.3,
        "predicted": 36.7218320185354,
        "residual": 11.578167981464595,
        "abs_error": 11.578167981464595
      },
      {
        "original_id": 239,
        "actual": 23.3,
        "predicted": 28.62304496381229,
        "residual": -5.32304496381229,
        "abs_error": 5.32304496381229
      },
      {
        "original_id": 241,
        "actual": 20.1,
        "predicted": 24.061272314267455,
        "residual": -3.961272314267454,
        "abs_error": 3.961272314267454
      },
      {
        "original_id": 246,
        "actual": 24.3,
        "predicted": 20.25087211747942,
        "residual": 4.04912788252058,
        "abs_error": 4.04912788252058
      },
      {
        "original_id": 247,
        "actual": 20.5,
        "predicted": 19.96561662765766,
        "residual": 0.5343833723423401,
        "abs_error": 0.5343833723423401
      },
      {
        "original_id": 253,
        "actual": 42.8,
        "predicted": 29.951430459932975,
        "residual": 12.848569540067022,
        "abs_error": 12.848569540067022
      },
      {
        "original_id": 254,
        "actual": 21.9,
        "predicted": 24.71583587623484,
        "residual": -2.8158358762348428,
        "abs_error": 2.8158358762348428
      },
      {
        "original_id": 278,
        "actual": 29.1,
        "predicted": 30.353691680427495,
        "residual": -1.2536916804274938,
        "abs_error": 1.2536916804274938
      },
      {
        "original_id": 293,
        "actual": 23.9,
        "predicted": 26.20577527247334,
        "residual": -2.3057752724733405,
        "abs_error": 2.3057752724733405
      },
      {
        "original_id": 297,
        "actual": 20.3,
        "predicted": 20.11459463776962,
        "residual": 0.1854053622303801,
        "abs_error": 0.1854053622303801
      },
      {
        "original_id": 299,
        "actual": 29.0,
        "predicted": 32.49654668484868,
        "residual": -3.4965466848486813,
        "abs_error": 3.4965466848486813
      },
      {
        "original_id": 300,
        "actual": 24.8,
        "predicted": 31.25165266527712,
        "residual": -6.45165266527712,
        "abs_error": 6.45165266527712
      },
      {
        "original_id": 303,
        "actual": 33.1,
        "predicted": 32.905221359861116,
        "residual": 0.1947786401388853,
        "abs_error": 0.1947786401388853
      },
      {
        "original_id": 310,
        "actual": 16.1,
        "predicted": 18.8967118021058,
        "residual": -2.7967118021058006,
        "abs_error": 2.7967118021058006
      },
      {
        "original_id": 319,
        "actual": 21.0,
        "predicted": 21.588691647335786,
        "residual": -0.5886916473357857,
        "abs_error": 0.5886916473357857
      },
      {
        "original_id": 320,
        "actual": 23.8,
        "predicted": 24.774957890739508,
        "residual": -0.9749578907395068,
        "abs_error": 0.9749578907395068
      },
      {
        "original_id": 326,
        "actual": 23.0,
        "predicted": 23.712611058922345,
        "residual": -0.7126110589223451,
        "abs_error": 0.7126110589223451
      },
      {
        "original_id": 329,
        "actual": 22.6,
        "predicted": 24.889637772756473,
        "residual": -2.289637772756471,
        "abs_error": 2.289637772756471
      },
      {
        "original_id": 330,
        "actual": 19.8,
        "predicted": 22.3146051649507,
        "residual": -2.514605164950698,
        "abs_error": 2.514605164950698
      },
      {
        "original_id": 331,
        "actual": 17.1,
        "predicted": 20.789705299728347,
        "residual": -3.6897052997283457,
        "abs_error": 3.6897052997283457
      },
      {
        "original_id": 340,
        "actual": 18.7,
        "predicted": 21.36099298414245,
        "residual": -2.6609929841424496,
        "abs_error": 2.6609929841424496
      },
      {
        "original_id": 342,
        "actual": 16.5,
        "predicted": 23.00363104480596,
        "residual": -6.50363104480596,
        "abs_error": 6.50363104480596
      },
      {
        "original_id": 344,
        "actual": 31.2,
        "predicted": 28.831426910296365,
        "residual": 2.3685730897036343,
        "abs_error": 2.3685730897036343
      },
      {
        "original_id": 346,
        "actual": 17.2,
        "predicted": 15.496688054528745,
        "residual": 1.7033119454712562,
        "abs_error": 1.7033119454712562
      },
      {
        "original_id": 352,
        "actual": 18.6,
        "predicted": 17.730478141145042,
        "residual": 0.8695218588549594,
        "abs_error": 0.8695218588549594
      },
      {
        "original_id": 363,
        "actual": 16.8,
        "predicted": 19.27774122699208,
        "residual": -2.47774122699208,
        "abs_error": 2.47774122699208
      },
      {
        "original_id": 367,
        "actual": 23.1,
        "predicted": 10.232665648735424,
        "residual": 12.867334351264578,
        "abs_error": 12.867334351264578
      },
      {
        "original_id": 371,
        "actual": 50.0,
        "predicted": 23.72141085316499,
        "residual": 26.27858914683501,
        "abs_error": 26.27858914683501
      },
      {
        "original_id": 372,
        "actual": 50.0,
        "predicted": 24.575605789012265,
        "residual": 25.424394210987735,
        "abs_error": 25.424394210987735
      },
      {
        "original_id": 374,
        "actual": 13.8,
        "predicted": 0.5902531893753746,
        "residual": 13.209746810624626,
        "abs_error": 13.209746810624626
      },
      {
        "original_id": 386,
        "actual": 10.5,
        "predicted": 5.593756586199156,
        "residual": 4.906243413800844,
        "abs_error": 4.906243413800844
      },
      {
        "original_id": 387,
        "actual": 7.4,
        "predicted": 5.232518169107685,
        "residual": 2.1674818308923154,
        "abs_error": 2.1674818308923154
      },
      {
        "original_id": 391,
        "actual": 23.2,
        "predicted": 16.688196405062612,
        "residual": 6.511803594937387,
        "abs_error": 6.511803594937387
      },
      {
        "original_id": 400,
        "actual": 5.6,
        "predicted": 11.155366481016262,
        "residual": -5.555366481016263,
        "abs_error": 5.555366481016263
      },
      {
        "original_id": 403,
        "actual": 8.3,
        "predicted": 12.122386208111934,
        "residual": -3.822386208111933,
        "abs_error": 3.822386208111933
      },
      {
        "original_id": 407,
        "actual": 27.9,
        "predicted": 18.971859948663873,
        "residual": 8.928140051336126,
        "abs_error": 8.928140051336126
      },
      {
        "original_id": 409,
        "actual": 27.5,
        "predicted": 19.02958926588229,
        "residual": 8.47041073411771,
        "abs_error": 8.47041073411771
      },
      {
        "original_id": 410,
        "actual": 15.0,
        "predicted": 13.98982600760467,
        "residual": 1.0101739923953303,
        "abs_error": 1.0101739923953303
      },
      {
        "original_id": 411,
        "actual": 17.2,
        "predicted": 16.38673959685491,
        "residual": 0.8132604031450903,
        "abs_error": 0.8132604031450903
      },
      {
        "original_id": 418,
        "actual": 8.8,
        "predicted": 5.127035128201015,
        "residual": 3.672964871798986,
        "abs_error": 3.672964871798986
      },
      {
        "original_id": 426,
        "actual": 10.2,
        "predicted": 15.885454166011026,
        "residual": -5.685454166011027,
        "abs_error": 5.685454166011027
      },
      {
        "original_id": 430,
        "actual": 14.5,
        "predicted": 17.586264396535817,
        "residual": -3.0862643965358174,
        "abs_error": 3.0862643965358174
      },
      {
        "original_id": 435,
        "actual": 13.4,
        "predicted": 13.10775596769474,
        "residual": 0.2922440323052608,
        "abs_error": 0.2922440323052608
      },
      {
        "original_id": 446,
        "actual": 14.9,
        "predicted": 17.077764845915265,
        "residual": -2.1777648459152648,
        "abs_error": 2.1777648459152648
      },
      {
        "original_id": 451,
        "actual": 15.2,
        "predicted": 18.71246369829179,
        "residual": -3.512463698291789,
        "abs_error": 3.512463698291789
      },
      {
        "original_id": 455,
        "actual": 14.1,
        "predicted": 15.571783281575794,
        "residual": -1.471783281575794,
        "abs_error": 1.471783281575794
      },
      {
        "original_id": 459,
        "actual": 20.0,
        "predicted": 17.792620720092486,
        "residual": 2.2073792799075136,
        "abs_error": 2.2073792799075136
      },
      {
        "original_id": 469,
        "actual": 20.1,
        "predicted": 17.732526353228224,
        "residual": 2.367473646771778,
        "abs_error": 2.367473646771778
      },
      {
        "original_id": 474,
        "actual": 13.8,
        "predicted": 15.607101557600824,
        "residual": -1.8071015576008236,
        "abs_error": 1.8071015576008236
      },
      {
        "original_id": 475,
        "actual": 13.3,
        "predicted": 15.49454112008096,
        "residual": -2.1945411200809595,
        "abs_error": 2.1945411200809595
      },
      {
        "original_id": 478,
        "actual": 14.6,
        "predicted": 18.30413012366064,
        "residual": -3.7041301236606383,
        "abs_error": 3.7041301236606383
      },
      {
        "original_id": 480,
        "actual": 23.0,
        "predicted": 22.457988089368584,
        "residual": 0.5420119106314161,
        "abs_error": 0.5420119106314161
      },
      {
        "original_id": 481,
        "actual": 23.7,
        "predicted": 25.899570588591395,
        "residual": -2.1995705885913956,
        "abs_error": 2.1995705885913956
      },
      {
        "original_id": 487,
        "actual": 20.6,
        "predicted": 20.49647539145716,
        "residual": 0.1035246085428411,
        "abs_error": 0.1035246085428411
      },
      {
        "original_id": 495,
        "actual": 23.1,
        "predicted": 17.352587914555208,
        "residual": 5.747412085444793,
        "abs_error": 5.747412085444793
      },
      {
        "original_id": 497,
        "actual": 18.3,
        "predicted": 19.294921593158627,
        "residual": -0.994921593158626,
        "abs_error": 0.994921593158626
      },
      {
        "original_id": 505,
        "actual": 11.9,
        "predicted": 22.16640988894835,
        "residual": -10.26640988894835,
        "abs_error": 10.26640988894835
      }
    ]
  },
  "curriculum_guide": {
    "metrics_definitions": [
      {
        "metric": "MAE (平均絕對誤差, Mean Absolute Error)",
        "formula": "MAE = (1/n) * Σ |y_i - ŷ_i|",
        "unit": "與目標變數相同單位 (例如：美元 $)",
        "meaning": "衡量預測值與真實值平均差距的絕對大小，對極端異常值具有線性穩健性 (Robustness)。"
      },
      {
        "metric": "MSE (均方誤差, Mean Squared Error)",
        "formula": "MSE = (1/n) * Σ (y_i - ŷ_i)²",
        "unit": "目標變數單位的平方 (例如：$²)",
        "meaning": "將誤差平方後求均值，對較大預測誤差施加非線性嚴厲懲罰；常用於數學求解最佳化。"
      },
      {
        "metric": "RMSE (均方根誤差, Root Mean Squared Error)",
        "formula": "RMSE = sqrt(MSE)",
        "unit": "與目標變數相同單位 (例如：美元 $)",
        "meaning": "兼具 MSE 對大誤差之敏感度，同時開根號回歸與目標相同之物理單位，便於實務解讀。"
      },
      {
        "metric": "R² (判定係數, Coefficient of Determination)",
        "formula": "R² = 1 - (SS_res / SS_tot)",
        "unit": "無量綱 (常介於 0 與 1 之間，低於基準模型均值預測時可為負數)",
        "meaning": "衡量目標變數的總變異量中，能被回歸模型所解釋的比例；不可直接稱為「準確率 (Accuracy)」。"
      }
    ],
    "diagnostic_principles": [
      {
        "topic": "Train/Test Split 的必要性",
        "explanation": "機器學習的核心目標是建立具備「泛化能力 (Generalization Ability)」的模型。若僅在訓練資料上評估，模型可透過記憶雜訊獲得虛高的分數；切出未參與訓練的獨立測試集，才能評估模型在未見新樣本上的真實表現。"
      },
      {
        "topic": "過擬合 (Overfitting) 與欠擬合 (Underfitting) 的判讀",
        "explanation": "欠擬合：訓練誤差與驗證誤差均高，模型複雜度不足以捕捉資料規律；過擬合：訓練誤差持續下降趨近於 0，但驗證／測試誤差顯著反彈攀升。不可僅憑單一次切分或單輪誤差微幅上升就斷言過擬合，需觀察多折交叉驗證或學習曲線趨勢。"
      },
      {
        "topic": "常態性 (Normality) 與等變異性 (Homoscedasticity) 的正確統計歸屬",
        "explanation": "回歸分析中的常態性與等變異性假設，嚴格指的是「模型誤差項／殘差 (Residuals, ε ~ N(0, σ²))」滿足常態分佈且變異數固定，絕非要求自變數 X 必須符合常態分佈或具有相同變異數。"
      },
      {
        "topic": "p-value 的正確定義",
        "explanation": "p-value 表示「在虛無假設 (H0) 成立的前提下，觀察到當前樣本檢定統計量（或更極端情況）的條件機率」，絕對不可錯誤解釋為「虛無假設為錯誤的機率」或「對立假設為真的機率」。"
      }
    ]
  }
};
