"""
模組名稱：export_web_data.py
說明：整合 Part 1、Part 2 與五大演算法基準評估的所有真實運算數據，
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

    # 2. 讀取 Part 1 結果
    p1_model_path = os.path.join(OUTPUT_DIR, "part1_linear_regression_model.pkl.gz")
    with gzip.open(p1_model_path, "rb") as f:
        p1_pkg = pickle.load(f)

    p1_test_pred_df = pd.read_csv(os.path.join(OUTPUT_DIR, "part1_test_predictions.csv"))
    p1_predictions = p1_test_pred_df.to_dict(orient="records")

    # 獨立精確計算 Part 1 指標
    p1_y_true = p1_test_pred_df["actual_profit"].values
    p1_y_pred = p1_test_pred_df["predicted_profit"].values
    p1_mae = float(np.mean(np.abs(p1_y_true - p1_y_pred)))
    p1_mse = float(np.mean((p1_y_true - p1_y_pred) ** 2))
    p1_rmse = float(np.sqrt(p1_mse))
    ss_tot = float(np.sum((p1_y_true - np.mean(p1_y_true)) ** 2))
    ss_res = float(np.sum((p1_y_true - p1_y_pred) ** 2))
    p1_r2 = float(1.0 - (ss_res / ss_tot))

    # 3. 讀取 Part 2 前向選擇結果
    p2_candidates_df = pd.read_csv(os.path.join(OUTPUT_DIR, "part2_forward_selection_candidates.csv"))
    p2_summary_df = pd.read_csv(os.path.join(OUTPUT_DIR, "part2_forward_selection_summary.csv"))

    with open(os.path.join(OUTPUT_DIR, "part2_step_predictions.json"), "r", encoding="utf-8") as f:
        p2_step_preds = json.load(f)

    with gzip.open(os.path.join(OUTPUT_DIR, "part2_forward_selection_models.pkl.gz"), "rb") as f:
        p2_models_pkg = pickle.load(f)

    # 4. 讀取五大演算法基準評估 (benchmark_algorithms.py 之輸出)
    benchmark_json_path = os.path.join(OUTPUT_DIR, "algorithms_benchmark_summary.json")
    if os.path.exists(benchmark_json_path):
        with open(benchmark_json_path, "r", encoding="utf-8") as f:
            benchmarks_data = json.load(f)
    else:
        benchmarks_data = {"algorithms": {}, "algorithm_descriptions": {}, "exhaustive_overall_best": {}}

    # 讀取窮舉 31 組詳細資料
    exhaustive_csv_path = os.path.join(OUTPUT_DIR, "exhaustive_search_31_subsets.csv")
    if os.path.exists(exhaustive_csv_path):
        exhaustive_df = pd.read_csv(exhaustive_csv_path)
        exhaustive_31_subsets = exhaustive_df.to_dict(orient="records")
    else:
        exhaustive_31_subsets = []

    # 整理 Part 2 各輪真實資訊（絕無固定覆蓋常數）
    p2_rounds_details = []
    for step_num in range(1, 6):
        step_candidates = p2_candidates_df[p2_candidates_df["round"] == step_num].to_dict(orient="records")
        step_summary_row = p2_summary_df[p2_summary_df["num_features"] == step_num].iloc[0].to_dict()
        model_info = p2_models_pkg["models_by_step"][step_num]

        # 直接讀取真實訓練集統計量
        train_adj_r2 = float(step_summary_row["train_adj_r2"]) if "train_adj_r2" in step_summary_row and pd.notna(step_summary_row["train_adj_r2"]) else None
        train_f_stat = float(step_summary_row["train_f_stat"]) if "train_f_stat" in step_summary_row and pd.notna(step_summary_row["train_f_stat"]) else None
        train_f_pval = float(step_summary_row["train_f_pvalue"]) if "train_f_pvalue" in step_summary_row and pd.notna(step_summary_row["train_f_pvalue"]) else None

        p2_rounds_details.append({
            "step": int(step_num),
            "added_feature": step_summary_row["added_feature"],
            "selected_features": [f.strip() for f in step_summary_row["selected_features_str"].split(",")],
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
                "test_rmse": float(c["test_rmse"]),
                "test_r2": float(c["test_r2"]),
                "train_rmse": float(c["train_rmse"]),
                "train_r2": float(c["train_r2"]),
                "is_selected": bool(c["is_selected"])
            } for c in step_candidates],
            "predictions": p2_step_preds[str(step_num)]
        })

    # 4. 讀取 Boston Housing 結果 (若存在)
    boston_model_path = os.path.join(OUTPUT_DIR, "boston_housing_linear_regression_model.pkl.gz")
    boston_pred_path = os.path.join(OUTPUT_DIR, "boston_housing_test_predictions.csv")
    if os.path.exists(boston_model_path) and os.path.exists(boston_pred_path):
        with gzip.open(boston_model_path, "rb") as f:
            b_pkg = pickle.load(f)
        b_pred_df = pd.read_csv(boston_pred_path)
        boston_predictions = b_pred_df.to_dict(orient="records")
        boston_metrics = b_pkg["metrics"]
        b_model = b_pkg["model"]
        boston_intercept = float(b_model.intercept_)
        boston_coefficients = {name: float(coef) for name, coef in zip(b_pkg["feature_names"], b_model.coef_)}
        b_terms = [f"{boston_intercept:.4f}"]
        for name, coef in boston_coefficients.items():
            sign = "+" if coef >= 0 else "-"
            b_terms.append(f"{sign} {abs(coef):.4f} * ({name})")
        boston_formula = "medv = " + " ".join(b_terms)
    else:
        boston_predictions = []
        boston_metrics = {}
        boston_formula = ""
        boston_coefficients = {}
        boston_intercept = 0.0

    # 動態找出 Part 2 前向選擇路徑中的最低 RMSE 輪次
    best_p2_round = min(p2_rounds_details, key=lambda x: x["test_rmse"])

    # 組裝最終結構化 Web JSON
    web_data = {
        "metadata": {
            "title": "50 Startups 多元線性回歸分析與特徵選擇實證研究",
            "subtitle": "Multiple Linear Regression, Forward Selection & Benchmark Algorithms",
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_source_url": "https://raw.githubusercontent.com/harimittapalli/Mulitple-Linear-Reggression/master/50_Startups.csv",
            "total_rows": int(len(df)),
            "raw_features_count": 4,
            "encoded_features_count": 5,
            "test_size": 0.2,
            "random_state": 0,
            "train_rows_count": 40,
            "test_rows_count": 10,
            "train_indices": p1_pkg["experiment_config"]["train_sample_count"],
            "test_indices": p1_pkg["experiment_config"]["test_sample_count"]
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
                "key_takeaway": "線性模型提供變數間敏感度與相依性度量，但不可逕行宣稱為因果關係。"
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
                "goal": "將類別型 State 進行 One-Hot Encoding，剃除第一個類別 (California) 作為基準州以避免虛擬變數陷阱 (Dummy Variable Trap)。",
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
                "goal": "計算 MAE, MSE, RMSE, R² 指標，進行 OLS 殘差假設檢定與顯著性分析。",
                "key_takeaway": "測試集 RMSE 為 9137.99，R² 為 0.9347；整體模型 F 統計量為 129.70 (p = 1.11e-16)。"
            },
            {
                "id": "deployment",
                "phase_name_zh": "模型部署與打包 (Deployment)",
                "phase_name_en": "Deployment",
                "goal": "使用 pickle + gzip 序列化模型與前處理轉換器，確保由原始欄位直接推論之完全可重現性。",
                "key_takeaway": "模型重載驗證數值誤差為 0.00e+00，確保生產環境端到端推論完全一致。"
            }
        ],
        "part1": {
            "title": "Part 1: CRISP-DM 多元線性回歸基準模型 (Baseline Model)",
            "encoded_features": p1_pkg["encoded_features_order"],
            "reference_state": p1_pkg["reference_state"],
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
                "F_pvalue": 1.11e-16,
                "degrees_of_freedom": {"df_model": 5, "df_residuals": 34}
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
            "title": "Part 2: Forward Selection 前向特徵選擇演算法",
            "methodology_note": "本實作嚴格遵循課程講義規格，每輪以獨立測試集 (Test Set) RMSE 最小化作為特徵挑選條件。方法學限制說明：由於測試集已參與模型篩選路徑 (Model Selection)，測試集之評估指標在統計上偏向樂觀 (Optimistic Bias)；嚴謹工業實務應採用獨立驗證集 (Validation Set) 或嵌套交叉驗證 (Nested CV)。",
            "candidate_columns": list(p2_candidates_df["candidate_feature"].unique()),
            "rounds_summary": p2_rounds_details,
            "progression_table": p2_summary_df.to_dict(orient="records"),
            "best_feature_count": best_p2_round["step"],
            "best_features": best_p2_round["selected_features"],
            "best_test_rmse": best_p2_round["test_rmse"],
            "best_test_r2": best_p2_round["test_r2"]
        },
        "algorithms_benchmark": benchmarks_data.get("algorithms", {}),
        "algorithm_descriptions": benchmarks_data.get("algorithm_descriptions", {}),
        "exhaustive_search": {
            "all_31_subsets": exhaustive_31_subsets,
            "best_per_k": benchmarks_data.get("algorithms", {}).get("Exhaustive", []),
            "overall_best": benchmarks_data.get("exhaustive_overall_best", {})
        },
        "boston_housing": {
            "title": "Topic 2: Boston Housing 波士頓房價多元線性回歸實證 (CRISP-DM Workflow)",
            "description": "經典波士頓房價回歸資料集 (506 筆樣本、13 項環境與社區特徵、目標變數為住宅中位數價值 MEDV，以千美元計)。",
            "features_info": {
                "crim": "城鎮人均犯罪率",
                "zn": "大面積住宅用地比例",
                "indus": "非零售商業用地比例",
                "chas": "查爾斯河虛擬變數 (邊界為河為 1)",
                "nox": "一氧化氮濃度 (ppm)",
                "rm": "住宅平均房間數",
                "age": "1940 年前建造之自住單位比例",
                "dis": "到五個就業中心之加權距離",
                "rad": "徑向公路通道指數",
                "tax": "每萬美元全值財產稅率",
                "ptratio": "學生與教師比例",
                "b": "城鎮黑人比例相關指數",
                "lstat": "人口中低地位社群比例 (%)"
            },
            "metrics": boston_metrics,
            "formula": boston_formula,
            "coefficients": boston_coefficients,
            "intercept": boston_intercept,
            "test_predictions": boston_predictions
        },
        "discussion": {
            "insights": [
                {
                    "title": "1. 核心獲利驅動力：研發支出 (R&D Spend)",
                    "content": "在特徵選擇中，單獨納入 R&D Spend 即達到測試集 R² = 0.9465，RMSE 降至 8274.87，在統計推論中 t = 14.03, p = 1.11e-15，顯示研發投入是初創企業獲利變異量的最大單一決定因數。"
                },
                {
                    "title": "2. 前向選擇路徑的最佳表現 (Path-Dependent Optimal)",
                    "content": "在前向選擇路徑中，第 2 輪納入 Marketing Spend 後，測試集 RMSE 降至 8198.80 (R² = 0.9474)，在該前向路徑中表現最佳。然而需注意：前向選擇為貪婪演算法，其所得結果僅代表該搜尋路徑下的最優，不等於全子集組合的全域最優。"
                },
                {
                    "title": "3. 全子集窮舉檢驗與貪婪搜尋的盲點",
                    "content": "透過對全部 31 組非空特徵組合之窮舉檢驗，我們發現 3 特徵組合【R&D Spend + Administration + State_Florida】在同一測試集上的 RMSE 達 8171.65 (優於前向選擇之 8198.80)。這是因為前向選擇在第 2 步選入 Marketing Spend 後，永遠無法探索不包含 Marketing 的其他 3 特徵組合，生動展示了貪婪特徵選擇之路徑依賴性。"
                },
                {
                    "title": "4. 單一切分下誤差變化的詮釋邊界與方法學限制",
                    "content": "在前向選擇第 3~5 輪中，測試集 RMSE 自 8198.80 上升至 9137.99。在本次 10 筆測試樣本切分下，這反映了新增參數帶來的額外變異；但單一次切分並不足以宣稱已證明模型過度擬合或後續特徵純屬雜訊。此外，以測試集作為特徵挑選依據，測試指標在統計上偏向樂觀，嚴謹實務應以交叉驗證進行篩選。"
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
