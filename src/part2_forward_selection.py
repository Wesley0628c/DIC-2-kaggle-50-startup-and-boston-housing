"""
模組名稱：part2_forward_selection.py
說明：依據機器學習嚴謹規範實作 5-Fold Cross-Validation Forward Selection (前向特徵選擇) 演算法。
核心原則：
  1. 嚴格杜絕資料洩漏 (Data Leakage)：特徵選擇與超參數決策一律在訓練集 (40 筆) 內以 5-Fold 交叉驗證完成。
  2. 評估指標：每輪以 5 折平均 Validation RMSE 最低者納入特徵；最佳特徵數 k 由 CV 曲線之最小值決定。
  3. 獨立測試評估：在特徵選定後，才以完整訓練集擬合模型，並在未參與決策的獨立測試集 (10 筆) 進行最終泛化評估。
  4. 歷史講義對照：原先以測試集 RMSE 選特徵之方式，保留於「歷史探索示範區 (Historical Exploration Demo)」，清楚標示其評估偏差。
  5. 繪製圖表：
     - 特徵數 vs CV RMSE / Train RMSE 趨勢圖 (outputs/part2_features_trend.png)
     - 學習曲線 Learning Curve (outputs/learning_curve.png)
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
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
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
    計算回歸評估指標：MAE, MSE, RMSE, R²。
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def cv_forward_selection(X_train_df: pd.DataFrame, y_train: np.ndarray,
                         X_test_df: pd.DataFrame, y_test: np.ndarray,
                         n_splits: int = 5, cv_random_state: int = 42):
    """
    【正式規範】基於訓練集內部 5-Fold Cross-Validation 的前向特徵選擇。
    絕不使用測試集 RMSE 做特徵挑選！
    """
    all_candidate_features = sorted(list(X_train_df.columns))
    selected_features = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
    
    rounds_detailed_log = []
    progression_summary = []
    models_per_step = {}

    print(f"\n[CV 前向特徵選擇啟動] 候選特徵數：{len(all_candidate_features)} 個，使用 {n_splits}-Fold CV")

    for step in range(1, len(all_candidate_features) + 1):
        print(f"\n===== 第 {step} 輪前向選擇 (Round {step}) =====")
        remaining = [f for f in all_candidate_features if f not in selected_features]
        remaining.sort()

        best_cand = None
        best_cv_rmse = float('inf')
        best_cv_std = 0.0

        round_candidates_eval = []

        for cand in remaining:
            current_combo = selected_features + [cand]
            
            # 在 5 折交叉驗證中計算平均 Validation RMSE
            cv_scores = -cross_val_score(
                LinearRegression(fit_intercept=True),
                X_train_df[current_combo].values,
                y_train,
                cv=kf,
                scoring='neg_root_mean_squared_error'
            )
            mean_cv_rmse = float(np.mean(cv_scores))
            std_cv_rmse = float(np.std(cv_scores))

            round_candidates_eval.append({
                "round": step,
                "candidate_feature": cand,
                "current_combination": current_combo.copy(),
                "cv_rmse": mean_cv_rmse,
                "cv_rmse_std": std_cv_rmse
            })

            print(f"  測試候選: {cand:<20} | CV RMSE: {mean_cv_rmse:>10.2f} (+/- {std_cv_rmse:>8.2f})")

            if mean_cv_rmse < best_cv_rmse:
                best_cv_rmse = mean_cv_rmse
                best_cv_std = std_cv_rmse
                best_cand = cand

        # 標記勝出者
        for rec in round_candidates_eval:
            rec["is_selected"] = (rec["candidate_feature"] == best_cand)
        rounds_detailed_log.extend(round_candidates_eval)

        # 納入最佳特徵
        selected_features.append(best_cand)
        print(f"--> 第 {step} 輪勝出特徵：【{best_cand}】 (CV RMSE: {best_cv_rmse:.2f})")
        print(f"    目前特徵組合：{selected_features}")

        # 使用完整訓練集重新擬合 OLS 模型
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train_df[selected_features].values, y_train)

        # 評估訓練集與獨立測試集表現 (測試集僅作紀錄，不回流干預特徵決策)
        y_tr_pred = model.predict(X_train_df[selected_features].values)
        y_te_pred = model.predict(X_test_df[selected_features].values)

        train_perf = evaluate_performance(y_train, y_tr_pred)
        test_perf = evaluate_performance(y_test, y_te_pred)

        # 計算訓練集 Adjusted R² 與 F-statistic (n=40, p=step)
        n_train = len(y_train)
        p = step
        df_resid = n_train - p - 1
        rss_tr = float(np.sum((y_train - y_tr_pred) ** 2))
        tss_tr = float(np.sum((y_train - np.mean(y_train)) ** 2))
        train_adj_r2 = float(1.0 - (1.0 - train_perf["R2"]) * (n_train - 1) / df_resid) if df_resid > 0 else None
        if df_resid > 0 and rss_tr > 1e-12:
            f_stat = float(((tss_tr - rss_tr) / p) / (rss_tr / df_resid))
            f_pval = float(1.0 - stats.f.cdf(f_stat, p, df_resid))
        else:
            f_stat = None
            f_pval = None

        models_per_step[step] = {
            "num_features": step,
            "selected_features": selected_features.copy(),
            "model": model,
            "cv_rmse": best_cv_rmse,
            "cv_rmse_std": best_cv_std,
            "train_perf": train_perf,
            "test_perf": test_perf,
            "test_predictions": y_te_pred,
            "intercept": float(model.intercept_),
            "coefficients": dict(zip(selected_features.copy(), [float(c) for c in model.coef_]))
        }

        progression_summary.append({
            "step": step,
            "num_features": step,
            "added_feature": best_cand,
            "selected_features": selected_features.copy(),
            "selected_features_str": ", ".join(selected_features),
            "cv_rmse": best_cv_rmse,
            "cv_rmse_std": best_cv_std,
            "train_rmse": train_perf["RMSE"],
            "train_r2": train_perf["R2"],
            "train_adj_r2": train_adj_r2,
            "train_f_stat": f_stat,
            "train_f_pvalue": f_pval,
            "test_rmse": test_perf["RMSE"],
            "test_r2": test_perf["R2"],
            "test_mae": test_perf["MAE"],
            "test_mse": test_perf["MSE"]
        })

    # 依 CV RMSE 找出最佳特徵數 k
    best_step_by_cv = min(progression_summary, key=lambda x: x["cv_rmse"])["num_features"]

    return rounds_detailed_log, progression_summary, models_per_step, best_step_by_cv


def historical_test_sfs(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame,
                        y_train: np.ndarray, y_test: np.ndarray):
    """
    【歷史講義示範區 (教學對照組)】
    原先以「測試集 Test RMSE 最小化」進行特徵選拔之流程。
    此處明確標示其具有資料洩漏 (Data Leakage) 性質，僅供教學展示評估偏差，不作為正式模型選擇標準。
    """
    all_candidates = sorted(list(X_train_df.columns))
    selected = []
    history_progression = []

    for step in range(1, len(all_candidates) + 1):
        remaining = [c for c in all_candidates if c not in selected]
        remaining.sort()
        best_cand = None
        best_test_rmse = float('inf')

        for cand in remaining:
            combo = selected + [cand]
            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_train_df[combo].values, y_train)
            pred = lr.predict(X_test_df[combo].values)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            if rmse < best_test_rmse:
                best_test_rmse = rmse
                best_cand = cand

        selected.append(best_cand)
        lr = LinearRegression(fit_intercept=True).fit(X_train_df[selected].values, y_train)
        pred = lr.predict(X_test_df[selected].values)
        history_progression.append({
            "step": step,
            "num_features": step,
            "added_feature": best_cand,
            "selected_features_str": ", ".join(selected),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_r2": float(r2_score(y_test, pred))
        })

    return history_progression


def plot_feature_progression_and_learning_curves(progression_summary, X_train_df, y_train, best_features):
    """
    繪製：
      1. outputs/part2_features_trend.png: 特徵數 vs CV RMSE / Train RMSE (附帶 Test 診斷曲線)
      2. outputs/learning_curve.png: 最佳模型的學習曲線 (Learning Curve)
    """
    steps = [row["num_features"] for row in progression_summary]
    cv_rmses = [row["cv_rmse"] for row in progression_summary]
    train_rmses = [row["train_rmse"] for row in progression_summary]
    test_rmses = [row["test_rmse"] for row in progression_summary]

    # 1. 特徵數趨勢圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # CV RMSE vs Train RMSE (正式依據)
    ax1.plot(steps, cv_rmses, 'o-', color='#dc2626', linewidth=2.2, markersize=8, label='5-Fold CV RMSE (驗證選模依據)')
    ax1.plot(steps, train_rmses, 's--', color='#2563eb', linewidth=1.8, markersize=6, label='Train RMSE (訓練誤差)')
    ax1.set_title("特徵數 vs 交叉驗證誤差 (CV RMSE vs Train RMSE)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("入選特徵數量 k", fontsize=11)
    ax1.set_ylabel("RMSE ($) [越低越佳]", fontsize=11)
    ax1.set_xticks(steps)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(frameon=True)

    # 診斷比較：Train vs Test (固定方案後的事後檢驗，不能用於挑選 k)
    ax2.plot(steps, test_rmses, '^-', color='#059669', linewidth=2, markersize=7, label='獨立 Test RMSE (事後泛化診斷)')
    ax2.plot(steps, train_rmses, 's--', color='#2563eb', linewidth=1.5, markersize=6, label='Train RMSE (訓練誤差)')
    ax2.set_title("診斷性對比：Train vs Test RMSE (事後評估，非選模依據)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("入選特徵數量 k", fontsize=11)
    ax2.set_ylabel("RMSE ($)", fontsize=11)
    ax2.set_xticks(steps)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(frameon=True)

    plt.tight_layout()
    trend_path = os.path.join(OUTPUT_DIR, "part2_features_trend.png")
    plt.savefig(trend_path, dpi=300)
    plt.close()
    print(f"[圖表輸出] 前向選擇趨勢圖已儲存至：{trend_path}")

    # 2. 學習曲線 (Learning Curve) - 以 CV 最佳特徵子集繪製
    X_sub = X_train_df[best_features].values
    train_sizes, train_scores, val_scores = learning_curve(
        LinearRegression(fit_intercept=True),
        X_sub,
        y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        train_sizes=np.linspace(0.3, 1.0, 6),
        scoring='neg_root_mean_squared_error',
        random_state=42
    )

    train_rmse_mean = -np.mean(train_scores, axis=1)
    train_rmse_std = np.std(train_scores, axis=1)
    val_rmse_mean = -np.mean(val_scores, axis=1)
    val_rmse_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8.5, 5.5))
    plt.plot(train_sizes, train_rmse_mean, 'o-', color='#2563eb', label='Training Score (Train RMSE)')
    plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std, train_rmse_mean + train_rmse_std, alpha=0.15, color='#2563eb')
    plt.plot(train_sizes, val_rmse_mean, 's-', color='#dc2626', label='Cross-validation Score (Val RMSE)')
    plt.fill_between(train_sizes, val_rmse_mean - val_rmse_std, val_rmse_mean + val_rmse_std, alpha=0.15, color='#dc2626')

    plt.title(f"學習曲線 Learning Curve (特徵組合: {', '.join(best_features)})", fontsize=12, fontweight='bold')
    plt.xlabel("訓練樣本數量 (Training Examples)", fontsize=11)
    plt.ylabel("RMSE ($) [越低越佳]", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()

    lc_path = os.path.join(OUTPUT_DIR, "learning_curve.png")
    plt.savefig(lc_path, dpi=300)
    plt.close()
    print(f"[圖表輸出] 學習曲線圖已儲存至：{lc_path}")

    return {
        "train_sizes": train_sizes.tolist(),
        "train_rmse_mean": [float(x) for x in train_rmse_mean],
        "train_rmse_std": [float(x) for x in train_rmse_std],
        "val_rmse_mean": [float(x) for x in val_rmse_mean],
        "val_rmse_std": [float(x) for x in val_rmse_std]
    }


def run_part2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("【Part 2】5-Fold CV 前向特徵選擇 (Forward Selection) 嚴謹實作啟動")
    print("=" * 60)

    # 1. 讀取資料
    df = pd.read_csv(DATA_PATH)
    
    # 2. 切分 80% 訓練集與 20% 測試集 (固定種子 random_state=0)
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=0)
    train_idx = sorted(train_idx.tolist())
    test_idx = sorted(test_idx.tolist())

    # 3. 前處理：僅從訓練資料建立 One-Hot 基準類別與結構
    X_raw = df.drop(columns=['Profit'])
    y = df['Profit'].values

    # 在訓練集取得編碼結構
    X_train_raw = X_raw.loc[train_idx]
    X_test_raw = X_raw.loc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # 使用 pd.get_dummies(drop_first=True)
    X_dummies = pd.get_dummies(X_raw, columns=['State'], drop_first=True, dtype=float)
    candidate_cols = list(X_dummies.columns)

    X_train_df = X_dummies.loc[train_idx].copy()
    X_test_df = X_dummies.loc[test_idx].copy()

    # 4. 執行 5-Fold CV 前向特徵選擇
    rounds_log, progression, models_by_step, best_k_cv = cv_forward_selection(
        X_train_df, y_train, X_test_df, y_test, n_splits=5, cv_random_state=42
    )

    # 5. 執行歷史講義示範 (供對照教學使用，明確標示評估偏差)
    historical_progression = historical_test_sfs(X_train_df, X_test_df, y_train, y_test)

    # 6. 輸出表格一：每輪所有候選特徵得分評估表
    all_candidates_df = pd.DataFrame(rounds_log)
    candidates_csv_path = os.path.join(OUTPUT_DIR, "part2_forward_selection_candidates.csv")
    all_candidates_df.to_csv(candidates_csv_path, index=False, encoding='utf-8')
    print(f"\n[輸出表格] 每輪候選特徵評估紀錄已輸出至：{candidates_csv_path}")

    # 7. 輸出表格二：1~5 特徵最終比較表
    summary_df = pd.DataFrame(progression)
    summary_csv_path = os.path.join(OUTPUT_DIR, "part2_forward_selection_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    print(f"[輸出表格] 1～5 個特徵最終比較表已輸出至：{summary_csv_path}")

    # 輸出歷史探索表
    hist_df = pd.DataFrame(historical_progression)
    hist_csv_path = os.path.join(OUTPUT_DIR, "part2_historical_test_sfs_summary.csv")
    hist_df.to_csv(hist_csv_path, index=False, encoding='utf-8')
    print(f"[輸出表格] 歷史講義示範表已輸出至：{hist_csv_path}")

    # 8. 儲存每輪的測試集逐筆預測
    step_predictions_records = {}
    for step in range(1, len(candidate_cols) + 1):
        step_info = models_by_step[step]
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

    # 9. 儲存模型套件
    best_features = progression[best_k_cv - 1]["selected_features"]
    models_package = {
        "candidate_columns": candidate_cols,
        "test_indices": test_idx,
        "train_indices": train_idx,
        "best_k_cv": best_k_cv,
        "best_features_cv": best_features,
        "models_by_step": {
            s: {
                "num_features": s,
                "features": m["selected_features"],
                "intercept": m["intercept"],
                "coefficients": m["coefficients"],
                "cv_rmse": m["cv_rmse"],
                "cv_rmse_std": m["cv_rmse_std"],
                "train_perf": m["train_perf"],
                "test_perf": m["test_perf"],
                "model": m["model"]
            } for s, m in models_by_step.items()
        },
        "historical_progression": historical_progression
    }
    part2_model_path = os.path.join(OUTPUT_DIR, "part2_forward_selection_models.pkl.gz")
    with gzip.open(part2_model_path, "wb") as f:
        pickle.dump(models_package, f)
    print(f"[模型儲存] Part 2 各輪模型套件已儲存至：{part2_model_path}")

    # 10. 繪製圖表與學習曲線
    lc_data = plot_feature_progression_and_learning_curves(progression, X_train_df, y_train, best_features)

    # 11. 方法論完整說明
    methodology_statement = (
        "【機器學習模型選擇與資料洩漏防治說明】：\n"
        f"1. 正式規範流程：以 5-Fold 交叉驗證 (K-Fold CV) 最小化驗證誤差為目標。\n"
        f"   - 最佳特徵數為 k={best_k_cv} ({', '.join(best_features)})。\n"
        f"   - 5-Fold CV RMSE 為 {progression[best_k_cv-1]['cv_rmse']:.2f}。\n"
        f"   - 在獨立測試集最終評估結果：Test RMSE={progression[best_k_cv-1]['test_rmse']:.2f}, R²={progression[best_k_cv-1]['test_r2']:.4f}。\n"
        "2. 歷史講義對照：講義原版以最終測試集 RMSE 選特徵，選出 k=2 (Test RMSE=8198.80)。\n"
        "   - 本系統清楚將其標記為『歷史探索示範』，指明測試集參與選模所導致的評估樂觀偏差 (Optimism Bias)。\n"
        "   - 嚴謹的機器學習實踐必須以交叉驗證或獨立驗證集為唯一選模依據。"
    )
    print(f"\n{methodology_statement}")

    return {
        "candidate_columns": candidate_cols,
        "progression": progression,
        "best_k_cv": best_k_cv,
        "best_features": best_features,
        "historical_progression": historical_progression,
        "learning_curve_data": lc_data,
        "methodology_statement": methodology_statement
    }


if __name__ == "__main__":
    run_part2()
