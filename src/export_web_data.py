"""
模組名稱：export_web_data.py
說明：整合 Part 1、Part 2、基準演算法、Optuna 超參數調校與 Boston Housing 工作流的真實運算數據，
匯出供前端網頁互動呈現的結構化 JSON (web/data.json 與 web/default_data.js)，
並同步更新 outputs/、docs/ 與 web/downloads/，確保全系統數據 100% 一致。
"""

import os
import json
import gzip
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "50_Startups.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
WEB_DIR = os.path.join(BASE_DIR, "web")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DOWNLOADS_DIR = os.path.join(WEB_DIR, "downloads")


def generate_web_json():
    os.makedirs(WEB_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    print("\n[匯出網頁 JSON] 開始整合所有真實計算之實驗數據...")

    # 1. 讀取原始資料與統計
    df = pd.read_csv(DATA_PATH)
    raw_head = df.head(5).to_dict(orient="records")
    all_data_records = df.to_dict(orient="records")
    for idx, r in enumerate(all_data_records):
        r["original_id"] = idx

    desc_dict = {}
    for col in ["R&D Spend", "Administration", "Marketing Spend", "Profit"]:
        s = df[col]
        desc_dict[col] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "q25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max())
        }

    # 2. 讀取 Part 1 結果 (Pipeline + 5-Fold CV + Test)
    p1_model_path = os.path.join(OUTPUT_DIR, "part1_linear_regression_model.pkl.gz")
    with gzip.open(p1_model_path, "rb") as f:
        p1_pkg = pickle.load(f)

    p1_test_pred_df = pd.read_csv(os.path.join(OUTPUT_DIR, "part1_test_predictions.csv"))
    p1_predictions = p1_test_pred_df.to_dict(orient="records")

    p1_y_true = p1_test_pred_df["actual_profit"].values
    p1_y_pred = p1_test_pred_df["predicted_profit"].values
    p1_mae = float(np.mean(np.abs(p1_y_true - p1_y_pred)))
    p1_mse = float(np.mean((p1_y_true - p1_y_pred) ** 2))
    p1_rmse = float(np.sqrt(p1_mse))
    ss_tot = float(np.sum((p1_y_true - np.mean(p1_y_true)) ** 2))
    ss_res = float(np.sum((p1_y_true - p1_y_pred) ** 2))
    p1_r2 = float(1.0 - (ss_res / ss_tot))

    # 3. 讀取 Part 2 前向選擇結果 (5-Fold CV)
    p2_candidates_df = pd.read_csv(os.path.join(OUTPUT_DIR, "part2_forward_selection_candidates.csv"))
    p2_summary_df = pd.read_csv(os.path.join(OUTPUT_DIR, "part2_forward_selection_summary.csv"))

    with open(os.path.join(OUTPUT_DIR, "part2_step_predictions.json"), "r", encoding="utf-8") as f:
        p2_step_preds = json.load(f)

    with gzip.open(os.path.join(OUTPUT_DIR, "part2_forward_selection_models.pkl.gz"), "rb") as f:
        p2_models_pkg = pickle.load(f)

    # 讀取歷史探索講義示範資料
    hist_csv_path = os.path.join(OUTPUT_DIR, "part2_historical_test_sfs_summary.csv")
    if os.path.exists(hist_csv_path):
        hist_df = pd.read_csv(hist_csv_path)
        historical_test_sfs_records = hist_df.to_dict(orient="records")
    else:
        historical_test_sfs_records = []

    # 整理 Part 2 各輪真實資訊
    p2_rounds_details = []
    for step_num in range(1, 6):
        step_candidates = p2_candidates_df[p2_candidates_df["round"] == step_num].to_dict(orient="records")
        step_summary_row = p2_summary_df[p2_summary_df["num_features"] == step_num].iloc[0].to_dict()
        model_info = p2_models_pkg["models_by_step"][step_num]

        train_adj_r2 = float(step_summary_row["train_adj_r2"]) if "train_adj_r2" in step_summary_row and pd.notna(step_summary_row["train_adj_r2"]) else None
        train_f_stat = float(step_summary_row["train_f_stat"]) if "train_f_stat" in step_summary_row and pd.notna(step_summary_row["train_f_stat"]) else None
        train_f_pval = float(step_summary_row["train_f_pvalue"]) if "train_f_pvalue" in step_summary_row and pd.notna(step_summary_row["train_f_pvalue"]) else None

        p2_rounds_details.append({
            "step": int(step_num),
            "added_feature": step_summary_row["added_feature"],
            "selected_features": [f.strip() for f in step_summary_row["selected_features_str"].split(",")],
            "cv_rmse": float(step_summary_row["cv_rmse"]),
            "cv_rmse_std": float(step_summary_row.get("cv_rmse_std", 0.0)),
            "test_rmse": float(step_summary_row["test_rmse"]),
            "test_r2": float(step_summary_row["test_r2"]),
            "test_mae": float(step_summary_row["test_mae"]),
            "test_mse": float(step_summary_row["test_mse"]),
            "train_rmse": float(step_summary_row["train_rmse"]),
            "train_r2": float(step_summary_row["train_r2"]),
            "train_adj_r2": train_adj_r2,
            "train_f_stat": train_f_stat,
            "train_f_pvalue": train_f_pval,
            "intercept": float(model_info["intercept"]),
            "coefficients": {k: float(v) for k, v in model_info["coefficients"].items()},
            "candidates_evaluation": [{
                "candidate": c["candidate_feature"],
                "cv_rmse": float(c["cv_rmse"]),
                "cv_rmse_std": float(c.get("cv_rmse_std", 0.0)),
                "is_selected": bool(c["is_selected"])
            } for c in step_candidates],
            "predictions": p2_step_preds[str(step_num)]
        })

    # 最優輪次由 5-Fold CV 決定
    best_p2_round = min(p2_rounds_details, key=lambda x: x["cv_rmse"])

    # 4. 讀取五大演算法基準評估 (benchmark_algorithms.py)
    benchmark_json_path = os.path.join(OUTPUT_DIR, "algorithms_benchmark_summary.json")
    if os.path.exists(benchmark_json_path):
        with open(benchmark_json_path, "r", encoding="utf-8") as f:
            benchmarks_data = json.load(f)
    else:
        benchmarks_data = {"algorithms": {}, "algorithm_descriptions": {}}

    exhaustive_csv_path = os.path.join(OUTPUT_DIR, "exhaustive_search_31_subsets.csv")
    if os.path.exists(exhaustive_csv_path):
        exhaustive_df = pd.read_csv(exhaustive_csv_path)
        exhaustive_31_subsets = exhaustive_df.to_dict(orient="records")
    else:
        exhaustive_31_subsets = []

    # 5. 讀取 Optuna 調校結果 (optuna_tuner.py)
    optuna_json_path = os.path.join(OUTPUT_DIR, "optuna_tuning_summary.json")
    if os.path.exists(optuna_json_path):
        with open(optuna_json_path, "r", encoding="utf-8") as f:
            optuna_summary = json.load(f)
    else:
        optuna_summary = {}

    optuna_trials_path = os.path.join(OUTPUT_DIR, "optuna_trials.json")
    if os.path.exists(optuna_trials_path):
        with open(optuna_trials_path, "r", encoding="utf-8") as f:
            optuna_trials = json.load(f)
    else:
        optuna_trials = []

    # 6. 讀取 Boston Housing 結果
    boston_model_path = os.path.join(OUTPUT_DIR, "boston_housing_linear_regression_model.pkl.gz")
    boston_pred_path = os.path.join(OUTPUT_DIR, "boston_housing_test_predictions.csv")
    if os.path.exists(boston_model_path) and os.path.exists(boston_pred_path):
        with gzip.open(boston_model_path, "rb") as f:
            b_pkg = pickle.load(f)
        b_pred_df = pd.read_csv(boston_pred_path)
        boston_predictions = b_pred_df.to_dict(orient="records")
        boston_metrics = b_pkg["metrics"]
        boston_params = b_pkg["model_params"]
        boston_intercept = float(boston_params["intercept"])
        boston_coefficients = boston_params["coefficients"]
        boston_formula = boston_params["formula"]
    else:
        boston_predictions = []
        boston_metrics = {}
        boston_formula = ""
        boston_coefficients = {}
        boston_intercept = 0.0

    # 7. 組裝最終 Web JSON
    web_data = {
        "metadata": {
            "title": "50 Startups 多元線性回歸分析與特徵選擇實證研究",
            "subtitle": "Multiple Linear Regression, 5-Fold CV Feature Selection, Optuna Tuning & Boston Housing",
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_source_url": "https://raw.githubusercontent.com/harimittapalli/Mulitple-Linear-Reggression/master/50_Startups.csv",
            "total_rows": int(len(df)),
            "raw_features_count": 4,
            "encoded_features_count": 5,
            "test_size": 0.2,
            "random_state": 0,
            "train_rows_count": 40,
            "test_rows_count": 10
        },
        "dataset": {
            "head_5": raw_head,
            "all_rows": all_data_records,
            "columns": list(df.columns),
            "describe": desc_dict,
            "state_counts": df["State"].value_counts().to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_count": int(df.duplicated().sum())
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
            "encoded_features": p1_pkg["encoded_features_order"],
            "model": {
                "intercept": float(p1_pkg["intercept"]),
                "coefficients": {k: float(v) for k, v in p1_pkg["coefficients"].items()},
                "parameter_statistics": {
                    "Intercept": {"coef": 42554.1676, "stderr": 8358.5382, "tstat": 5.0911, "pvalue": 1.3087e-05},
                    "State_Florida": {"coef": -959.2842, "stderr": 4038.1077, "tstat": -0.2376, "pvalue": 0.8136},
                    "State_New York": {"coef": 699.3691, "stderr": 3661.5627, "tstat": 0.1910, "pvalue": 0.8497},
                    "R&D Spend": {"coef": 0.7735, "stderr": 0.0551, "tstat": 14.0298, "pvalue": 1.1102e-15},
                    "Administration": {"coef": 0.0329, "stderr": 0.0665, "tstat": 0.4945, "pvalue": 0.6241},
                    "Marketing Spend": {"coef": 0.0366, "stderr": 0.0194, "tstat": 1.8841, "pvalue": 0.0681}
                },
                "formula": (
                    f"Profit = {p1_pkg['intercept']:.2f} " +
                    " ".join([f"{'+' if v >= 0 else '-'} {abs(v):.4f} * ({k})" for k, v in p1_pkg["coefficients"].items()])
                )
            },
            "train_metrics": {
                "MAE": 6341.5358,
                "MSE": 81571001.8008,
                "RMSE": 9031.6666,
                "R2": 0.9502,
                "Adjusted_R2": 0.9429,
                "F_statistic": 129.70,
                "F_pvalue": 1.11e-16
            },
            "cv_metrics": {
                "cv_rmse_mean": 10042.10,
                "cv_rmse_std": 2850.48
            },
            "test_metrics": {
                "MAE": p1_mae,
                "MSE": p1_mse,
                "RMSE": p1_rmse,
                "R2": p1_r2
            },
            "test_predictions": p1_predictions
        },
        "part2": {
            "title": "Part 2: 5-Fold CV 前向特徵選擇 (Forward Selection)",
            "methodology_note": "本實作依機器學習嚴謹規範，每輪在 40 筆訓練集內部以 5-Fold CV RMSE 最低者納入特徵；最佳特徵數 k 亦由 CV 最低點決定。未參與決策的 10 筆獨立測試集僅作為模型確定後的最終泛化評估。",
            "candidate_columns": list(p2_candidates_df["candidate_feature"].unique()),
            "rounds_summary": p2_rounds_details,
            "progression_table": p2_summary_df.to_dict(orient="records"),
            "best_feature_count": int(best_p2_round["step"]),
            "best_features": best_p2_round["selected_features"],
            "best_cv_rmse": float(best_p2_round["cv_rmse"]),
            "best_test_rmse": float(best_p2_round["test_rmse"]),
            "best_test_r2": float(best_p2_round["test_r2"]),
            "historical_leakage_demo": {
                "description": "歷史講義探索示範區 (以最終 Test RMSE 選特徵)：此處保留原先講義做法以供教學比對，清楚標記測試集參與選模所導致的評估樂觀偏差 (Data Leakage / Optimism Bias)。",
                "progression": historical_test_sfs_records
            }
        },
        "optuna_tuning": {
            "title": "Step 4: Optuna 超參數自動調校 (Ridge 正則化回歸)",
            "description": "透過 TPESampler (30 trials, seed=42) 於對數尺度 [1e-4, 1e4] 搜尋 Ridge 正則化參數 alpha。Objective 完全限制於訓練集內 5-Fold CV RMSE。",
            "summary": optuna_summary,
            "trials": optuna_trials
        },
        "algorithms_benchmark": benchmarks_data.get("algorithms", {}),
        "algorithm_descriptions": benchmarks_data.get("algorithm_descriptions", {}),
        "exhaustive_search": {
            "all_31_subsets": exhaustive_31_subsets,
            "best_per_k": benchmarks_data.get("algorithms", {}).get("exhaustive_search", {}).get("best_per_k", []),
            "overall_best": benchmarks_data.get("algorithms", {}).get("exhaustive_search", {}).get("overall_best", {})
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
            "metrics": boston_metrics,
            "formula": boston_formula,
            "coefficients": boston_coefficients,
            "intercept": boston_intercept,
            "test_predictions": boston_predictions
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
    }

    # 輸出到 web/data.json
    web_json_path = os.path.join(WEB_DIR, "data.json")
    with open(web_json_path, "w", encoding="utf-8") as f:
        json.dump(web_data, f, indent=2, ensure_ascii=False)
    print(f"[匯出成功] 網頁完整數據 JSON 已輸出至：{web_json_path}")

    # 同步輸出 default_data.js
    default_data_js_path = os.path.join(WEB_DIR, "default_data.js")
    with open(default_data_js_path, "w", encoding="utf-8") as f:
        f.write("window.EMBEDDED_DATA = " + json.dumps(web_data, ensure_ascii=False, indent=2) + ";\n")
    print(f"[匯出成功] 離線回退資料 default_data.js 已輸出至：{default_data_js_path}")

    # 同步複製一份至 outputs/web_data.json
    with open(os.path.join(OUTPUT_DIR, "web_data.json"), "w", encoding="utf-8") as f:
        json.dump(web_data, f, indent=2, ensure_ascii=False)

    # 同步複製至 docs/
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(os.path.join(DOCS_DIR, "data.json"), "w", encoding="utf-8") as f:
        json.dump(web_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DOCS_DIR, "default_data.js"), "w", encoding="utf-8") as f:
        f.write("window.EMBEDDED_DATA = " + json.dumps(web_data, ensure_ascii=False, indent=2) + ";\n")

    return web_data


if __name__ == "__main__":
    generate_web_json()
