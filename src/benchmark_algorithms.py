"""
模組名稱：benchmark_algorithms.py
說明：實作五大特徵選擇與正則化回歸演算法的完整真實運算、比較與輸出：
  1. Forward Selection (前向特徵選擇，以訓練集 5-Fold CV RMSE 逐步挑選)
  2. Lasso 正則化：
     - Lasso Direct (LassoCV L1 縮減直接預測)
     - Lasso + OLS (以 SelectFromModel 選取非零特徵後重擬合無偏 OLS)
  3. RFE (遞迴特徵消除法特徵排序 + 5-Fold CV 評估 + OLS 擬合)
  4. ElasticNet 彈性網絡：
     - ElasticNet Direct (L1+L2 縮減直接預測)
     - ElasticNet + OLS (特徵排序 + 5-Fold CV 評估 + OLS 擬合)
  5. Exhaustive Search (全子集窮舉，遍歷全部 31 組非空特徵組合之 5-Fold CV RMSE)

嚴格遵守無資料洩漏 (No Data Leakage) 原則：
所有演算法的特徵挑選、超參數與最佳特徵數 k，皆由訓練集內部 5-Fold CV 決定；
測試集僅作為模型確定後的獨立評估，不參與任何特徵選拔。
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, ElasticNetCV, ElasticNet
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "50_Startups.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def compute_model_metrics(model, X_train, y_train, X_test, y_test, feature_names, cv_rmse=None, cv_std=None):
    """
    計算回歸模型之完整訓練集、測試集指標與統計檢定量。
    """
    n_train = len(y_train)
    p = len(feature_names)
    df_resid = n_train - p - 1

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_mse = float(mean_squared_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(test_mse))
    test_r2 = float(r2_score(y_test, y_test_pred))

    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_mse = float(mean_squared_error(y_train, y_train_pred))
    train_rmse = float(np.sqrt(train_mse))
    train_r2 = float(r2_score(y_train, y_train_pred))

    # 訓練集 Adjusted R² 與 F-statistic
    rss_train = float(np.sum((y_train - y_train_pred) ** 2))
    tss_train = float(np.sum((y_train - np.mean(y_train)) ** 2))

    if df_resid > 0 and tss_train > 1e-12:
        train_adj_r2 = float(1.0 - (1.0 - train_r2) * (n_train - 1) / df_resid)
        denom = rss_train / df_resid
        if denom > 1e-12:
            f_stat = float(((tss_train - rss_train) / p) / denom)
            f_pvalue = float(1.0 - stats.f.cdf(f_stat, p, df_resid))
        else:
            f_stat = None
            f_pvalue = None
    else:
        train_adj_r2 = None
        f_stat = None
        f_pvalue = None

    intercept = float(model.intercept_)
    coefficients = {feat: float(coef) for feat, coef in zip(feature_names, model.coef_)}
    test_predictions = [float(pred) for pred in y_test_pred]

    return {
        "k": p,
        "features": list(feature_names),
        "cv_rmse": float(cv_rmse) if cv_rmse is not None else None,
        "cv_std": float(cv_std) if cv_std is not None else None,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "train_adj_r2": train_adj_r2,
        "train_f_stat": f_stat,
        "train_f_pvalue": f_pvalue,
        "intercept": intercept,
        "coefficients": coefficients,
        "test_predictions": test_predictions
    }


def eval_subset_cv(X_train_df, y_train, features, kf):
    """
    以 5-Fold CV 計算特定特徵子集之平均 Validation RMSE 與標準差。
    """
    scores = -cross_val_score(
        LinearRegression(fit_intercept=True),
        X_train_df[features].values,
        y_train,
        cv=kf,
        scoring='neg_root_mean_squared_error'
    )
    return float(np.mean(scores)), float(np.std(scores))


def run_cv_forward_sfs(X_train_df, X_test_df, y_train, y_test, kf):
    """
    1. Forward Selection (基於 5-Fold CV)
    每輪以 5 折交叉驗證 RMSE 最小者勝出。
    """
    all_candidates = sorted(list(X_train_df.columns))
    selected = []
    progression = []

    for step in range(1, len(all_candidates) + 1):
        remaining = [c for c in all_candidates if c not in selected]
        remaining.sort()

        best_cand = None
        best_cv_rmse = float('inf')
        best_cv_std = 0.0

        for cand in remaining:
            combo = selected + [cand]
            cv_mean, cv_std = eval_subset_cv(X_train_df, y_train, combo, kf)
            if cv_mean < best_cv_rmse:
                best_cv_rmse = cv_mean
                best_cv_std = cv_std
                best_cand = cand

        selected.append(best_cand)
        lr = LinearRegression(fit_intercept=True).fit(X_train_df[selected].values, y_train)
        res = compute_model_metrics(
            lr, X_train_df[selected].values, y_train,
            X_test_df[selected].values, y_test, selected,
            cv_rmse=best_cv_rmse, cv_std=best_cv_std
        )
        progression.append(res)

    return progression


def run_lasso_benchmark(X_train_df, X_test_df, y_train, y_test, kf):
    """
    2. Lasso 基準測試：
       (A) Lasso Direct (LassoCV 縮減係數直接預測)
       (B) Lasso + OLS (SelectFromModel 取非零特徵重擬合無偏 OLS)
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_df)
    X_te_s = scaler.transform(X_test_df)

    # (A) Lasso Direct
    lasso_cv = LassoCV(cv=kf, random_state=42, max_iter=10000)
    lasso_cv.fit(X_tr_s, y_train)

    best_alpha = float(lasso_cv.alpha_)
    # 交叉驗證 RMSE
    cv_scores = []
    for tr_f, val_f in kf.split(X_tr_s):
        m = Lasso(alpha=best_alpha, random_state=42, max_iter=10000).fit(X_tr_s[tr_f], y_train[tr_f])
        pred_val = m.predict(X_tr_s[val_f])
        cv_scores.append(np.sqrt(mean_squared_error(y_train[val_f], pred_val)))
    lasso_direct_cv_rmse = float(np.mean(cv_scores))
    lasso_direct_cv_std = float(np.std(cv_scores))

    y_tr_pred = lasso_cv.predict(X_tr_s)
    y_te_pred = lasso_cv.predict(X_te_s)

    lasso_direct_res = {
        "algorithm": "Lasso Direct",
        "description": "L1 正則化 (LassoCV) 直接預測，保留係數縮減懲罰",
        "best_alpha": best_alpha,
        "cv_rmse": lasso_direct_cv_rmse,
        "cv_std": lasso_direct_cv_std,
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_te_pred))),
        "test_r2": float(r2_score(y_test, y_te_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_te_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_tr_pred))),
        "train_r2": float(r2_score(y_train, y_tr_pred)),
        "intercept": float(lasso_cv.intercept_),
        "coefficients": {f: float(c) for f, c in zip(X_train_df.columns, lasso_cv.coef_)},
        "non_zero_features": [f for f, c in zip(X_train_df.columns, lasso_cv.coef_) if abs(c) > 1e-5],
        "test_predictions": [float(p) for p in y_te_pred]
    }

    # (B) Lasso + OLS (SelectFromModel)
    selector = SelectFromModel(lasso_cv, prefit=True)
    selected_mask = selector.get_support()
    selected_features = [f for f, s in zip(X_train_df.columns, selected_mask) if s]

    if len(selected_features) == 0:
        # 若全被剃除，則取係數絕對值最大者
        selected_features = [X_train_df.columns[np.argmax(np.abs(lasso_cv.coef_))]]

    cv_mean, cv_std = eval_subset_cv(X_train_df, y_train, selected_features, kf)
    lr = LinearRegression(fit_intercept=True).fit(X_train_df[selected_features].values, y_train)
    lasso_ols_res = compute_model_metrics(
        lr, X_train_df[selected_features].values, y_train,
        X_test_df[selected_features].values, y_test, selected_features,
        cv_rmse=cv_mean, cv_std=cv_std
    )
    lasso_ols_res["algorithm"] = "Lasso + OLS"
    lasso_ols_res["description"] = "以 Lasso 篩選非零特徵後，重新擬合無偏 OLS 模型"
    lasso_ols_res["best_alpha"] = best_alpha

    # 產生 k=1~5 階梯式排序比較
    coef_abs = np.abs(lasso_cv.coef_)
    ranking_idx = np.argsort(-coef_abs)
    ranked_features = [X_train_df.columns[i] for i in ranking_idx]
    progression = []
    for k in range(1, len(ranked_features) + 1):
        sub_feats = ranked_features[:k]
        c_mean, c_std = eval_subset_cv(X_train_df, y_train, sub_feats, kf)
        m = LinearRegression(fit_intercept=True).fit(X_train_df[sub_feats].values, y_train)
        progression.append(compute_model_metrics(
            m, X_train_df[sub_feats].values, y_train,
            X_test_df[sub_feats].values, y_test, sub_feats,
            cv_rmse=c_mean, cv_std=c_std
        ))

    return lasso_direct_res, lasso_ols_res, progression


def run_rfe_benchmark(X_train_df, X_test_df, y_train, y_test, kf):
    """
    3. RFE 基準測試 (Recursive Feature Elimination + 5-Fold CV + OLS)
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_df)

    rfe = RFE(estimator=LinearRegression(), n_features_to_select=1, step=1)
    rfe.fit(X_tr_s, y_train)

    ranked_pairs = sorted(zip(X_train_df.columns, rfe.ranking_), key=lambda x: x[1])
    ranked_features = [feat for feat, rank in ranked_pairs]

    progression = []
    for k in range(1, len(ranked_features) + 1):
        sub_feats = ranked_features[:k]
        cv_mean, cv_std = eval_subset_cv(X_train_df, y_train, sub_feats, kf)
        lr = LinearRegression(fit_intercept=True).fit(X_train_df[sub_feats].values, y_train)
        res = compute_model_metrics(
            lr, X_train_df[sub_feats].values, y_train,
            X_test_df[sub_feats].values, y_test, sub_feats,
            cv_rmse=cv_mean, cv_std=cv_std
        )
        progression.append(res)

    best_k = min(progression, key=lambda x: x["cv_rmse"])["k"]
    return progression, best_k


def run_elasticnet_benchmark(X_train_df, X_test_df, y_train, y_test, kf):
    """
    4. ElasticNet 基準測試 (L1+L2 Regularization + 5-Fold CV + OLS)
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_df)
    X_te_s = scaler.transform(X_test_df)

    enet_cv = ElasticNetCV(cv=kf, random_state=42, max_iter=10000)
    enet_cv.fit(X_tr_s, y_train)

    best_alpha = float(enet_cv.alpha_)
    best_l1_ratio = float(enet_cv.l1_ratio_)

    y_tr_pred = enet_cv.predict(X_tr_s)
    y_te_pred = enet_cv.predict(X_te_s)

    enet_direct_res = {
        "algorithm": "ElasticNet Direct",
        "best_alpha": best_alpha,
        "l1_ratio": best_l1_ratio,
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_te_pred))),
        "test_r2": float(r2_score(y_test, y_te_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_tr_pred))),
        "train_r2": float(r2_score(y_train, y_tr_pred)),
        "coefficients": {f: float(c) for f, c in zip(X_train_df.columns, enet_cv.coef_)},
        "test_predictions": [float(p) for p in y_te_pred]
    }

    # 排序重訓 OLS
    coef_abs = np.abs(enet_cv.coef_)
    ranking_idx = np.argsort(-coef_abs)
    ranked_features = [X_train_df.columns[i] for i in ranking_idx]

    progression = []
    for k in range(1, len(ranked_features) + 1):
        sub_feats = ranked_features[:k]
        cv_mean, cv_std = eval_subset_cv(X_train_df, y_train, sub_feats, kf)
        lr = LinearRegression(fit_intercept=True).fit(X_train_df[sub_feats].values, y_train)
        res = compute_model_metrics(
            lr, X_train_df[sub_feats].values, y_train,
            X_test_df[sub_feats].values, y_test, sub_feats,
            cv_rmse=cv_mean, cv_std=cv_std
        )
        progression.append(res)

    return enet_direct_res, progression


def run_exhaustive_search_benchmark(X_train_df, X_test_df, y_train, y_test, kf):
    """
    5. Exhaustive Search (全子集 31 組窮舉)
    嚴格遍歷全部 2^5 - 1 = 31 組非空特徵子集，計算 5-Fold CV RMSE 與獨立測試集表現。
    """
    all_features = list(X_train_df.columns)
    all_31_records = []
    best_per_k_records = []

    for k in range(1, len(all_features) + 1):
        best_k_res = None
        best_k_cv_rmse = float('inf')

        for combo in itertools.combinations(all_features, k):
            combo_list = list(combo)
            cv_mean, cv_std = eval_subset_cv(X_train_df, y_train, combo_list, kf)
            lr = LinearRegression(fit_intercept=True).fit(X_train_df[combo_list].values, y_train)
            res = compute_model_metrics(
                lr, X_train_df[combo_list].values, y_train,
                X_test_df[combo_list].values, y_test, combo_list,
                cv_rmse=cv_mean, cv_std=cv_std
            )
            all_31_records.append(res)

            if cv_mean < best_k_cv_rmse:
                best_k_cv_rmse = cv_mean
                best_k_res = res

        best_per_k_records.append(best_k_res)

    # 依 5-Fold CV RMSE 找出 31 組中之最佳組合
    overall_best_cv = min(all_31_records, key=lambda x: x["cv_rmse"])

    return all_31_records, best_per_k_records, overall_best_cv


def execute_benchmarks():
    """
    執行 5 大演算法基準測試並保存完整成果
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("【基準演算法比對】五大特徵選擇演算法 (5-Fold CV 規範) 啟動")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    X_raw = df.drop(columns=['Profit'])
    y = df['Profit'].values

    # 前處理 (相同切分: train_test_split test_size=0.2, random_state=0)
    X_dummies = pd.get_dummies(X_raw, columns=['State'], drop_first=True, dtype=float)
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=0)
    train_idx = sorted(train_idx.tolist())
    test_idx = sorted(test_idx.tolist())

    X_train_df = X_dummies.loc[train_idx].copy()
    X_test_df = X_dummies.loc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"訓練樣本：{len(train_idx)} 筆，測試樣本：{len(test_idx)} 筆")
    print(f"候選特徵欄位 ({len(X_train_df.columns)} 個)：{list(X_train_df.columns)}")

    # 1. Forward Selection
    print("\n[1/5] 執行 5-Fold CV Forward Selection (SFS)...")
    sfs_progression = run_cv_forward_sfs(X_train_df, X_test_df, y_train, y_test, kf)
    for r in sfs_progression:
        print(f"  k={r['k']} | 特徵: {r['features']} | CV RMSE: {r['cv_rmse']:.2f} | Test RMSE: {r['test_rmse']:.2f}")

    # 2. Lasso
    print("\n[2/5] 執行 Lasso (Lasso Direct & Lasso + OLS)...")
    lasso_direct, lasso_ols, lasso_progression = run_lasso_benchmark(X_train_df, X_test_df, y_train, y_test, kf)
    print(f"  Lasso Direct: alpha={lasso_direct['best_alpha']:.4f}, Test RMSE={lasso_direct['test_rmse']:.2f}, 非零特徵={lasso_direct['non_zero_features']}")
    print(f"  Lasso + OLS: 特徵={lasso_ols['features']}, CV RMSE={lasso_ols['cv_rmse']:.2f}, Test RMSE={lasso_ols['test_rmse']:.2f}")

    # 3. RFE
    print("\n[3/5] 執行 RFE (Recursive Feature Elimination + OLS)...")
    rfe_progression, rfe_best_k = run_rfe_benchmark(X_train_df, X_test_df, y_train, y_test, kf)
    for r in rfe_progression:
        print(f"  k={r['k']} | 特徵: {r['features']} | CV RMSE: {r['cv_rmse']:.2f} | Test RMSE: {r['test_rmse']:.2f}")

    # 4. ElasticNet
    print("\n[4/5] 執行 ElasticNet (Direct & OLS)...")
    enet_direct, enet_progression = run_elasticnet_benchmark(X_train_df, X_test_df, y_train, y_test, kf)
    for r in enet_progression:
        print(f"  k={r['k']} | 特徵: {r['features']} | CV RMSE: {r['cv_rmse']:.2f} | Test RMSE: {r['test_rmse']:.2f}")

    # 5. Exhaustive Search
    print("\n[5/5] 執行 Exhaustive Search (全子集 31 組窮舉)...")
    all_31, best_per_k, overall_best_cv = run_exhaustive_search_benchmark(X_train_df, X_test_df, y_train, y_test, kf)
    for r in best_per_k:
        print(f"  k={r['k']} 最優: {r['features']} | CV RMSE: {r['cv_rmse']:.2f} | Test RMSE: {r['test_rmse']:.2f}")
    print(f"\n>> 全子集 31 組中之最佳組合 (依 5-Fold CV)：{overall_best_cv['features']} (CV RMSE: {overall_best_cv['cv_rmse']:.2f}, Test RMSE: {overall_best_cv['test_rmse']:.2f})")

    # 儲存全子集 31 組 CSV
    all_31_df = pd.DataFrame([{
        "features": ", ".join(r["features"]),
        "k": r["k"],
        "cv_rmse": r["cv_rmse"],
        "cv_std": r["cv_std"],
        "test_rmse": r["test_rmse"],
        "test_r2": r["test_r2"],
        "train_rmse": r["train_rmse"],
        "train_r2": r["train_r2"]
    } for r in all_31])
    all_31_csv_path = os.path.join(OUTPUT_DIR, "exhaustive_search_31_subsets.csv")
    all_31_df.to_csv(all_31_csv_path, index=False, encoding='utf-8')
    print(f"\n[輸出表格] 全子集 31 組評估表已儲存至：{all_31_csv_path}")

    # 整合 5 大演算法成果為 JSON
    benchmark_payload = {
        "evaluation_standard": "5-Fold Cross-Validation on Training Set (n=40)",
        "train_indices": train_idx,
        "test_indices": test_idx,
        "algorithms": {
            "sfs": {
                "name": "Forward Selection (前向特徵選擇)",
                "progression": sfs_progression,
                "best_k": min(sfs_progression, key=lambda x: x["cv_rmse"])["k"]
            },
            "lasso_direct": lasso_direct,
            "lasso_ols": {
                "name": "Lasso + OLS (L1 篩選重訓)",
                "progression": lasso_progression,
                "selected_model": lasso_ols
            },
            "rfe": {
                "name": "RFE (遞迴消除法)",
                "progression": rfe_progression,
                "best_k": rfe_best_k
            },
            "elasticnet_direct": enet_direct,
            "elasticnet_ols": {
                "name": "ElasticNet + OLS (彈性網絡排序重訓)",
                "progression": enet_progression,
                "best_k": min(enet_progression, key=lambda x: x["cv_rmse"])["k"]
            },
            "exhaustive_search": {
                "name": "Exhaustive Search (全子集 31 組窮舉)",
                "best_per_k": best_per_k,
                "overall_best": overall_best_cv
            }
        },
        "algorithm_descriptions": {
            "sfs": "從空集合開始，每輪依 5-Fold CV RMSE 最小者加入一個候選特徵，直到全部納入。",
            "lasso_direct": "以 L1 懲罰項將不重要特徵權重收縮至 0，以縮減係數直接預測。",
            "lasso_ols": "利用 LassoCV 挑選出非零係數特徵，再使用標準 OLS 擬合無偏模型，兼顧稀疏性與估計不偏性。",
            "rfe": "以線性模型權重遞迴剔除最不重要特徵，依留存順序評估 top-k 表現。",
            "elasticnet_ols": "同時結合 L1 與 L2 正則化權重排序，再以 OLS 擬合最佳子集。",
            "exhaustive_search": "窮舉遍歷全部 31 種可能特徵組合，以 5-Fold CV 評估找出全域最優解。"
        }
    }

    benchmark_json_path = os.path.join(OUTPUT_DIR, "algorithms_benchmark_summary.json")
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2, ensure_ascii=False)
    print(f"[輸出 JSON] 五大演算法基準彙整已儲存至：{benchmark_json_path}")

    return benchmark_payload


if __name__ == "__main__":
    execute_benchmarks()
