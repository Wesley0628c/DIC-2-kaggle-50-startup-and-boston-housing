"""
模組名稱：optuna_tuner.py
說明：多元線性回歸課程之超參數自動優化模組 (使用 Optuna)。
包含：
  1. 基準模型 (Baseline LinearRegression)：無正則化之 OLS 模型作為對照基準。
  2. Optuna 調參模型：
     - 正則化回歸 (Ridge / Lasso) 搭配 StandardScaler 組裝為 Pipeline。
     - 嚴格杜絕資料洩漏：Objective 僅使用訓練集 (40 筆) 之 5-Fold 交叉驗證 RMSE。
     - 對數尺度 (Log Scale) 搜尋超參數 alpha (範圍 1e-4 ~ 1e4)。
     - 固定 TPESampler 種子 (seed=42)，預設 30 次試驗 (Trials)。
  3. 產出與驗證：
     - 保存 30 次試驗明細至 outputs/optuna_trials.json 與 CSV。
     - 繪製優化歷程圖 outputs/optuna_optimization_history.png。
     - 以最佳參數於完整訓練集重新擬合，並在獨立測試集 (10 筆) 進行最終評估。
     - 誠實比較基準模型與調參模型之 CV RMSE 與 Test RMSE。
     - 將最佳 Pipeline 模型以 gzip + pickle 儲存至 outputs/optuna_tuned_model.pkl.gz。
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
import optuna
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 設定繪圖字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang TC', 'Heiti TC', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "50_Startups.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def run_optuna_tuning(n_trials: int = 30, random_seed: int = 42):
    """
    執行 Optuna 超參數優化流程
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print(f"【Step 4】Optuna 超參數自動調校啟動 (Trials={n_trials}, Seed={random_seed})")
    print("=" * 60)

    # 1. 讀取資料與切分
    df = pd.read_csv(DATA_PATH)
    X_raw = df.drop(columns=['Profit'])
    y = df['Profit'].values

    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=0)
    train_idx = sorted(train_idx.tolist())
    test_idx = sorted(test_idx.tolist())

    X_train_df = X_raw.loc[train_idx].copy()
    X_test_df = X_raw.loc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    num_cols = ["R&D Spend", "Administration", "Marketing Spend"]
    cat_cols = ["State"]

    # 完整前處理 ColumnTransformer：數值以中位數填補並標準化；類別以 OneHotEncoder(drop='first') 編碼
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    # 2. 基準模型 (Baseline: OLS LinearRegression，包在 Pipeline 中)
    print("\n[1/4] 評估課程基準模型 (Baseline LinearRegression Pipeline)...")
    base_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression(fit_intercept=True))
    ])

    base_cv_scores = -cross_val_score(base_pipe, X_train_df, y_train, cv=kf, scoring='neg_root_mean_squared_error')
    base_cv_rmse = float(np.mean(base_cv_scores))
    base_cv_std = float(np.std(base_cv_scores))

    base_pipe.fit(X_train_df, y_train)
    base_te_pred = base_pipe.predict(X_test_df)
    base_te_rmse = float(np.sqrt(mean_squared_error(y_test, base_te_pred)))
    base_te_r2 = float(r2_score(y_test, base_te_pred))
    base_te_mae = float(mean_absolute_error(y_test, base_te_pred))

    print(f"  基準模型 5-Fold CV RMSE: {base_cv_rmse:.2f} (+/- {base_cv_std:.2f})")
    print(f"  基準模型 獨立 Test RMSE : {base_te_rmse:.2f} | Test R²: {base_te_r2:.4f}")

    # 3. Optuna 調參 (模型：Ridge 回歸，調校正則化強度 alpha)
    print(f"\n[2/4] 開始 Optuna 超參數搜尋 (Ridge 回歸，alpha 對數尺度 [1e-4, 1e4])...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trials_records = []

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=alpha, random_state=random_seed))
        ])

        scores = -cross_val_score(pipe, X_train_df, y_train, cv=kf, scoring='neg_root_mean_squared_error')
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        trial.set_user_attr("cv_std", std_score)

        trials_records.append({
            "trial_number": trial.number,
            "alpha": float(alpha),
            "cv_rmse": mean_score,
            "cv_std": std_score
        })

        return mean_score

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_trial_num = study.best_trial.number
    best_alpha = float(study.best_params["alpha"])
    best_cv_rmse = float(study.best_value)
    best_cv_std = float(study.best_trial.user_attrs.get("cv_std", 0.0))

    print(f"\n[3/4] Optuna 最優試驗結果：")
    print(f"  最佳 Trial: #{best_trial_num}")
    print(f"  最佳 alpha : {best_alpha:.4f}")
    print(f"  最佳 5-Fold CV RMSE: {best_cv_rmse:.2f} (+/- {best_cv_std:.2f})")

    # 4. 在完整訓練集上重新擬合最佳 Pipeline 模型
    best_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=best_alpha, random_state=random_seed))
    ])
    best_pipe.fit(X_train_df, y_train)

    # 取得轉換後的特徵欄位名稱
    ohe = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = [f"State_{cat}" for cat in ohe.categories_[0][1:]]
    feature_names = num_cols + cat_feature_names

    tuned_te_pred = best_pipe.predict(X_test_df)
    tuned_te_rmse = float(np.sqrt(mean_squared_error(y_test, tuned_te_pred)))
    tuned_te_r2 = float(r2_score(y_test, tuned_te_pred))
    tuned_te_mae = float(mean_absolute_error(y_test, tuned_te_pred))

    cv_improvement = ((base_cv_rmse - best_cv_rmse) / base_cv_rmse) * 100.0
    test_improvement = ((base_te_rmse - tuned_te_rmse) / base_te_rmse) * 100.0

    print(f"  調優模型 獨立 Test RMSE : {tuned_te_rmse:.2f} | Test R²: {tuned_te_r2:.4f}")
    print(f"  CV RMSE 改善幅度 : {cv_improvement:+.2f}%")
    print(f"  Test RMSE 改善幅度: {test_improvement:+.2f}%")

    # 5. 輸出試驗紀錄 CSV 與 JSON
    trials_df = pd.DataFrame(trials_records)
    trials_csv_path = os.path.join(OUTPUT_DIR, "optuna_trials.csv")
    trials_df.to_csv(trials_csv_path, index=False, encoding='utf-8')

    trials_json_path = os.path.join(OUTPUT_DIR, "optuna_trials.json")
    with open(trials_json_path, "w", encoding="utf-8") as f:
        json.dump(trials_records, f, indent=2, ensure_ascii=False)
    print(f"\n[輸出表格] Optuna {n_trials} 次試驗紀錄已輸出至：{trials_csv_path}")

    # 6. 繪製調參歷程圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Trial 歷程
    trial_nums = [t["trial_number"] for t in trials_records]
    cv_scores = [t["cv_rmse"] for t in trials_records]
    ax1.plot(trial_nums, cv_scores, 'o-', color='#3b82f6', markersize=5, alpha=0.7, label='Trial CV RMSE')
    ax1.axhline(base_cv_rmse, color='#dc2626', linestyle='--', linewidth=1.5, label=f'Baseline OLS CV RMSE ({base_cv_rmse:.1f})')
    ax1.axhline(best_cv_rmse, color='#10b981', linestyle='-', linewidth=1.5, label=f'Best Ridge CV RMSE ({best_cv_rmse:.1f})')
    ax1.scatter([best_trial_num], [best_cv_rmse], color='#f59e0b', s=120, zorder=5, label=f'Best Trial #{best_trial_num}')
    ax1.set_title("Optuna 試驗歷程 (Trial vs 5-Fold CV RMSE)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Trial 編號", fontsize=11)
    ax1.set_ylabel("5-Fold CV RMSE ($)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='best', frameon=True)

    # Alpha vs CV RMSE 散佈圖 (對數刻度)
    alphas = [t["alpha"] for t in trials_records]
    ax2.scatter(alphas, cv_scores, color='#8b5cf6', s=45, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax2.scatter([best_alpha], [best_cv_rmse], color='#ef4444', s=140, marker='*', zorder=5, label=f'Best alpha={best_alpha:.4f}')
    ax2.set_xscale('log')
    ax2.set_title("超參數對數尺度搜尋 (alpha vs CV RMSE)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("alpha (Log Scale)", fontsize=11)
    ax2.set_ylabel("5-Fold CV RMSE ($)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='best', frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "optuna_optimization_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[圖表輸出] Optuna 優化歷程圖已儲存至：{plot_path}")

    # 7. 序列化最佳模型 Pipeline (gzip + pickle)
    model_package = {
        "model_type": "Pipeline(StandardScaler + Ridge)",
        "pipeline": best_pipe,
        "feature_names": feature_names,
        "best_alpha": best_alpha,
        "best_trial_number": best_trial_num,
        "baseline_metrics": {
            "cv_rmse": base_cv_rmse,
            "cv_std": base_cv_std,
            "test_rmse": base_te_rmse,
            "test_r2": base_te_r2,
            "test_mae": base_te_mae
        },
        "tuned_metrics": {
            "cv_rmse": best_cv_rmse,
            "cv_std": best_cv_std,
            "test_rmse": tuned_te_rmse,
            "test_r2": tuned_te_r2,
            "test_mae": tuned_te_mae
        },
        "cv_improvement_pct": cv_improvement,
        "test_improvement_pct": test_improvement,
        "model": best_pipe.named_steps["regressor"],
        "preprocessor": best_pipe.named_steps["preprocessor"],
        "column_transformer": best_pipe.named_steps["preprocessor"],
        "coefficients": {f: float(c) for f, c in zip(feature_names, best_pipe.named_steps["regressor"].coef_)},
        "intercept": float(best_pipe.named_steps["regressor"].intercept_)
    }

    optuna_model_path = os.path.join(OUTPUT_DIR, "optuna_tuned_model.pkl.gz")
    with gzip.open(optuna_model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"[模型儲存] Optuna 調優模型已保存至：{optuna_model_path}")

    # 驗證重新載入推論一致性
    with gzip.open(optuna_model_path, "rb") as f:
        reloaded_pkg = pickle.load(f)
    reloaded_pipe = reloaded_pkg["pipeline"]
    reloaded_pred = reloaded_pipe.predict(X_test_df)
    max_diff = float(np.max(np.abs(tuned_te_pred - reloaded_pred)))
    assert max_diff < 1e-10, f"Optuna 模型重載推論不一致，最大差異：{max_diff}"
    print(f"[重載驗證] 重載推論數值完全一致 (Max Diff: {max_diff:.2e})")

    # 8. 儲存總結 JSON
    summary_path = os.path.join(OUTPUT_DIR, "optuna_tuning_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(model_package, f, indent=2, ensure_ascii=False, default=str)
    print(f"[輸出 JSON] 調優總結報告已儲存至：{summary_path}")

    return model_package


if __name__ == "__main__":
    run_optuna_tuning()
