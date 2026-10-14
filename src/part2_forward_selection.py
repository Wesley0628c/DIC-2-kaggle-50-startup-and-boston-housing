"""
模組名稱：part2_forward_selection.py
說明：依據講義規格自行實作 Forward Selection (前向特徵選擇) 演算法。
包含：
  - evaluate_performance(y_true, y_pred)
  - forward_selection(X_train, X_test, y_train, y_test, max_features=5)
  - 記錄每輪所有候選特徵的評估分數
  - 產生 1~5 個特徵的漸進比較表
  - 保存各輪特徵模型與逐筆預測
  - 繪製特徵數與指標變化趨勢圖
"""

import os
import json
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 設定中英相容字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang TC', 'Heiti TC', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "50_Startups.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def evaluate_performance(y_true, y_pred):
    """
    自訂評估函式：計算 MAE, MSE, RMSE, R²。
    回傳字典格式之指標。
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

def forward_selection(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame, 
                      y_train: np.ndarray, y_test: np.ndarray, 
                      max_features: int = 5):
    """
    依講義邏輯自行實作的前向特徵選擇 (Forward Selection)。
    選擇依據：以測試集 RMSE 最小者勝出。
    """
    all_candidate_features = sorted(list(X_train_df.columns))
    selected_features = []
    
    rounds_detailed_log = []     # 記錄每輪中所有候選特徵的詳細評分
    final_progression_summary = [] # 記錄 1~5 特徵最終各輪的指標與模型
    models_per_step = {}         # 保存每輪訓練好的模型與預測

    print(f"\n[前向選擇初始化] 所有候選特徵 ({len(all_candidate_features)} 個)：{all_candidate_features}")

    for step in range(1, max_features + 1):
        print(f"\n===== 第 {step} 輪前向選擇 (Round {step}) =====")
        remaining_candidates = [f for f in all_candidate_features if f not in selected_features]
        # 穩定排序以確保完全重現
        remaining_candidates.sort()

        best_candidate = None
        best_rmse = float('inf')
        best_r2 = -float('inf')
        best_model = None
        best_train_perf = None
        best_test_perf = None
        best_test_pred = None

        round_candidates_eval = []

        for candidate in remaining_candidates:
            current_combo = selected_features + [candidate]
            
            # 取出對應特徵子集
            X_tr_sub = X_train_df[current_combo].values
            X_te_sub = X_test_df[current_combo].values

            # 訓練 LinearRegression
            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_tr_sub, y_train)

            # 預測
            y_tr_pred = lr.predict(X_tr_sub)
            y_te_pred = lr.predict(X_te_sub)

            # 評估 (依照講義以 test RMSE 為選拔標準)
            perf_tr = evaluate_performance(y_train, y_tr_pred)
            perf_te = evaluate_performance(y_test, y_te_pred)

            eval_record = {
                "round": step,
                "candidate_feature": candidate,
                "current_combination": current_combo.copy(),
                "test_rmse": perf_te["RMSE"],
                "test_r2": perf_te["R2"],
                "train_rmse": perf_tr["RMSE"],
                "train_r2": perf_tr["R2"]
            }
            round_candidates_eval.append(eval_record)

            print(f"  測試候選: {candidate:<20} | Test RMSE: {perf_te['RMSE']:>10.4f} | Test R²: {perf_te['R2']:>8.4f}")

            # 比較選擇 RMSE 最小者
            if perf_te["RMSE"] < best_rmse:
                best_rmse = perf_te["RMSE"]
                best_r2 = perf_te["R2"]
                best_candidate = candidate
                best_model = lr
                best_train_perf = perf_tr
                best_test_perf = perf_te
                best_test_pred = y_te_pred

        # 標記本輪勝出者
        for rec in round_candidates_eval:
            rec["is_selected"] = (rec["candidate_feature"] == best_candidate)

        rounds_detailed_log.extend(round_candidates_eval)

        # 將最佳特徵加入已選清單
        selected_features.append(best_candidate)
        print(f"--> 第 {step} 輪勝出特徵：【{best_candidate}】 (Test RMSE: {best_rmse:.4f}, Test R²: {best_r2:.4f})")
        print(f"    目前已選清單：{selected_features}")

        # 儲存此輪模型與預測結果
        models_per_step[step] = {
            "num_features": step,
            "selected_features": selected_features.copy(),
            "model": best_model,
            "train_perf": best_train_perf,
            "test_perf": best_test_perf,
            "test_predictions": best_test_pred,
            "intercept": float(best_model.intercept_),
            "coefficients": dict(zip(selected_features.copy(), [float(c) for c in best_model.coef_]))
        }

        # 嚴謹計算訓練集 Adjusted R2 與 F 統計量 (n=40, p=step)
        from scipy import stats
        n_train = len(y_train)
        p = step
        df_resid = n_train - p - 1
        y_tr_pred = best_model.predict(X_train_df[selected_features].values)
        rss_tr = float(np.sum((y_train - y_tr_pred)**2))
        tss_tr = float(np.sum((y_train - np.mean(y_train))**2))
        train_adj_r2 = float(1.0 - (1.0 - best_train_perf["R2"]) * (n_train - 1) / df_resid) if df_resid > 0 else None
        if df_resid > 0 and rss_tr > 1e-12:
            f_stat = float(((tss_tr - rss_tr) / p) / (rss_tr / df_resid))
            f_pval = float(1.0 - stats.f.cdf(f_stat, p, df_resid))
        else:
            f_stat = None
            f_pval = None

        final_progression_summary.append({
            "step": step,
            "num_features": step,
            "added_feature": best_candidate,
            "selected_features": selected_features.copy(),
            "selected_features_str": ", ".join(selected_features),
            "test_rmse": best_test_perf["RMSE"],
            "test_r2": best_test_perf["R2"],
            "test_mae": best_test_perf["MAE"],
            "test_mse": best_test_perf["MSE"],
            "train_rmse": best_train_perf["RMSE"],
            "train_r2": best_train_perf["R2"],
            "train_adj_r2": train_adj_r2,
            "train_f_stat": f_stat,
            "train_f_pvalue": f_pval
        })

    return rounds_detailed_log, final_progression_summary, models_per_step

def run_part2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("【Part 2】講義 Forward Selection (前向特徵選擇) 演算法實作啟動")
    print("=" * 60)

    # 1. 讀取同一份資料
    df = pd.read_csv(DATA_PATH)
    X_raw = df.drop(columns=['Profit'])
    y = df['Profit'].values

    # 2. 講義方式建立前處理：pd.get_dummies(X, columns=['State'], drop_first=True)
    # drop_first=True 會自動以第一個類別作為基準州並剃除
    X_dummies = pd.get_dummies(X_raw, columns=['State'], drop_first=True, dtype=float)
    
    # 確保候選欄位為 5 個
    candidate_cols = list(X_dummies.columns)
    assert len(candidate_cols) == 5, f"候選特徵數非 5 個，而是 {len(candidate_cols)} 個：{candidate_cols}"
    assert 'Profit' not in candidate_cols, "候選特徵不應包含 Profit！"
    print(f"Part 2 經 pd.get_dummies 編碼後之 5 個候選特徵：{candidate_cols}")

    # 3. 與 Part 1 相同的 train_test_split (test_size=0.2, random_state=0)
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=0)
    train_idx = sorted(train_idx.tolist())
    test_idx = sorted(test_idx.tolist())

    # 驗證 train/test 列與 Part 1 嚴格一致
    expected_test_idx = [2, 4, 10, 11, 22, 27, 28, 31, 38, 41]
    assert test_idx == expected_test_idx, f"測試集索引不符合預期！得到 {test_idx}"
    print(f"測試集索引核對無誤：{test_idx}")

    X_train_df = X_dummies.loc[train_idx].copy()
    X_test_df = X_dummies.loc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    # 4. 執行 forward_selection
    rounds_detailed_log, final_progression, models_per_step = forward_selection(
        X_train_df, X_test_df, y_train, y_test, max_features=5
    )

    # 5. 輸出表格一：每輪所有候選特徵得分評估表
    all_candidates_df = pd.DataFrame(rounds_detailed_log)
    candidates_csv_path = os.path.join(OUTPUT_DIR, "part2_forward_selection_candidates.csv")
    all_candidates_df.to_csv(candidates_csv_path, index=False, encoding='utf-8')
    print(f"\n[輸出表格] 每輪候選特徵評估紀錄已輸出至：{candidates_csv_path}")

    # 6. 輸出表格二：1~5 特徵最終比較表
    summary_df = pd.DataFrame(final_progression)
    summary_csv_path = os.path.join(OUTPUT_DIR, "part2_forward_selection_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    print(f"[輸出表格] 1～5 個特徵最終比較表已輸出至：{summary_csv_path}")

    print("\n【1～5 個特徵最終比較表 (Final Progression Comparison)】：")
    print(f"{'特徵數':<6} | {'入選特徵清單 (Selected Features)':<45} | {'Test RMSE':<12} | {'Test R²':<10}")
    print("-" * 80)
    for row in final_progression:
        print(f"{row['num_features']:<6} | {row['selected_features_str']:<45} | {row['test_rmse']:<12.4f} | {row['test_r2']:<10.4f}")

    # 7. 保存每輪的逐筆預測供網頁即時切換
    step_predictions_records = {}
    for step in range(1, 6):
        step_info = models_per_step[step]
        y_pred = step_info["test_predictions"]
        records = []
        for i, orig_id in enumerate(test_idx):
            act_val = float(y_test[i])
            pred_val = float(y_pred[i])
            res = act_val - pred_val
            records.append({
                "original_id": int(orig_id),
                "actual_profit": act_val,
                "predicted_profit": pred_val,
                "residual": res,
                "abs_error": abs(res)
            })
        step_predictions_records[step] = records

    step_pred_json_path = os.path.join(OUTPUT_DIR, "part2_step_predictions.json")
    with open(step_pred_json_path, "w", encoding="utf-8") as f:
        json.dump(step_predictions_records, f, indent=2, ensure_ascii=False)
    print(f"[逐筆預測] 各輪特徵數測試集逐筆預測已保存至：{step_pred_json_path}")

    # 8. 儲存每輪模型為 pickle + gzip
    models_package = {
        "candidate_columns": candidate_cols,
        "test_indices": test_idx,
        "train_indices": train_idx,
        "models_by_step": {
            s: {
                "num_features": s,
                "features": m["selected_features"],
                "intercept": m["intercept"],
                "coefficients": m["coefficients"],
                "train_perf": m["train_perf"],
                "test_perf": m["test_perf"],
                "model": m["model"]
            } for s, m in models_per_step.items()
        }
    }
    part2_model_path = os.path.join(OUTPUT_DIR, "part2_forward_selection_models.pkl.gz")
    with gzip.open(part2_model_path, "wb") as f:
        pickle.dump(models_package, f)
    print(f"[模型儲存] Part 2 各輪模型套件已儲存至：{part2_model_path}")

    # 9. 繪製特徵數與指標變化趨勢圖 (RMSE 與 R²)
    steps = [row["num_features"] for row in final_progression]
    test_rmses = [row["test_rmse"] for row in final_progression]
    train_rmses = [row["train_rmse"] for row in final_progression]
    test_r2s = [row["test_r2"] for row in final_progression]
    train_r2s = [row["train_r2"] for row in final_progression]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # RMSE 趨勢
    ax1.plot(steps, test_rmses, 'o-', color='#dc2626', linewidth=2, markersize=8, label='測試集 RMSE (Test)')
    ax1.plot(steps, train_rmses, 's--', color='#2563eb', linewidth=1.5, markersize=6, label='訓練集 RMSE (Train)')
    ax1.set_title("特徵數 vs RMSE 變化趨勢", fontsize=12, fontweight='bold')
    ax1.set_xlabel("入選特徵數量 (Number of Features)", fontsize=11)
    ax1.set_ylabel("RMSE ($) [越低越佳]", fontsize=11)
    ax1.set_xticks(steps)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(frameon=True)

    # R² 趨勢
    ax2.plot(steps, test_r2s, 'o-', color='#10b981', linewidth=2, markersize=8, label='測試集 R² (Test)')
    ax2.plot(steps, train_r2s, 's--', color='#6366f1', linewidth=1.5, markersize=6, label='訓練集 R² (Train)')
    ax2.set_title("特徵數 vs R² 變化趨勢", fontsize=12, fontweight='bold')
    ax2.set_xlabel("入選特徵數量 (Number of Features)", fontsize=11)
    ax2.set_ylabel("R² (判定係數) [越接近 1 越佳]", fontsize=11)
    ax2.set_xticks(steps)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(frameon=True)

    plt.tight_layout()
    trend_chart_path = os.path.join(OUTPUT_DIR, "part2_features_trend.png")
    plt.savefig(trend_chart_path, dpi=300)
    plt.close()
    print(f"[圖表輸出] 前向選擇趨勢圖已儲存至：{trend_chart_path}")

    # 方法限制嚴正說明
    methodology_limitation = (
        "【重要方法限制說明】：\n"
        "此版本依講義教學步驟，以『測試集 (Test Set)』的 RMSE 作為特徵選擇之依據。\n"
        "在標準機器學習嚴謹規範中，當測試集參與了特徵選拔（Model Selection），"
        "測試集的分數即具有資料外洩 (Data Leakage) 性質，所得指標往往偏向樂觀，"
        "不能視為完全獨立的最終泛化評估 (Generalization Performance)。\n"
        "較嚴謹的做法應是在訓練集內部使用交叉驗證 (Cross-Validation) 或另切驗證集 (Validation Set) 進行特徵挑選，"
        "最終再由獨立之測試集做評估。"
    )
    print(f"\n{methodology_limitation}")

    return {
        "candidate_columns": candidate_cols,
        "rounds_detailed_log": rounds_detailed_log,
        "final_progression": final_progression,
        "step_predictions": step_predictions_records,
        "methodology_limitation": methodology_limitation
    }

if __name__ == "__main__":
    run_part2()
