"""
模組名稱：part1_crispdm.py
說明：實作 Part 1 CRISP-DM 完整規範回歸流程（徹底消除資料洩漏）：
  A. Business Understanding (商業理解)
  B. Data Understanding (資料探索 EDA、教學用隨機合成散佈圖與真實散佈圖)
  C. Data Preparation (處理無效目標值、先切分後前處理、僅從訓練集學習中位數與 One-Hot、Pipeline 整合)
  D. Modeling (LinearRegression 多元線性回歸、訓練集 OLS 統計推論)
  E. Evaluation (MAE, MSE, RMSE, R² 指標、Actual vs Pred、殘差圖、逐筆評估)
  F. Deployment (完整 Pipeline 序列化、原始 DataFrame 端到端重載驗證)
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 設定字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang TC', 'Heiti TC', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "50_Startups.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def run_part1():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("【Part 1】CRISP-DM 多元線性回歸流程 (嚴格無洩漏 Pipeline 規範) 啟動")
    print("=" * 65)

    # ---------------------------------------------------------
    # A. Business Understanding (商業理解)
    # ---------------------------------------------------------
    print("\n--- A. Business Understanding (商業理解) ---")
    business_context = {
        "problem": "創投基金與企業管理者希望依據 50 家初創企業在研發 (R&D Spend)、行政管理 (Administration)、市場行銷 (Marketing Spend) 之支出與所在州 (State)，建立多元線性回歸模型以預測企業獲利能力 (Profit)。",
        "variables": {
            "features_X": "R&D Spend, Administration, Marketing Spend, State",
            "target_y": "Profit (預測目標，單位：美元)"
        },
        "caution": "回歸模型展現的是變數間的統計相依性 (Statistical Association)，不能直接推論為因果關係 (Causality)。"
    }
    print(business_context["problem"])
    print(f"特徵 X：{business_context['variables']['features_X']}")
    print(f"目標 y：{business_context['variables']['target_y']}")
    print(f"統計提醒：{business_context['caution']}")

    # ---------------------------------------------------------
    # B. Data Understanding (資料理解與品質檢驗)
    # ---------------------------------------------------------
    print("\n--- B. Data Understanding (資料理解與探索) ---")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到資料集檔案：{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    total_rows, total_cols = df.shape
    print(f"原始資料規模：{total_rows} 筆列資料 (Rows)，{total_cols} 個欄位 (Columns)")
    print("\n[前五筆資料 Head(5)]：")
    print(df.head())

    # 缺失值與重複列檢查
    null_info = df.isnull().sum().to_dict()
    dup_count = int(df.duplicated().sum())
    state_dist = df['State'].value_counts().to_dict()
    print(f"\n缺失值檢測：{null_info}")
    print(f"重複列檢測：{dup_count} 筆")
    print(f"State 類別分佈：{state_dist}")

    # 1. 繪製【教學示範用隨機合成資料散佈圖】（Step 5 要求：固定種子，與真實資料明確區分）
    np.random.seed(42)
    synth_x = np.random.uniform(10, 100, 50)
    synth_noise = np.random.normal(0, 15, 50)
    synth_y = 2.5 * synth_x + 30 + synth_noise

    fig_demo, ax_demo = plt.subplots(figsize=(7, 4.5))
    ax_demo.scatter(synth_x, synth_y, color="#8b5cf6", edgecolors="k", alpha=0.8, s=60, label="隨機合成樣本 (Synthetic Samples)")
    ax_demo.plot(np.linspace(10, 100, 100), 2.5 * np.linspace(10, 100, 100) + 30, "r--", lw=2, label="理論母體趨勢線 (True Trend)")
    ax_demo.set_title("【教學示範】隨機合成資料散佈圖 (Synthetic Data Scatter Demonstration)", fontsize=11, fontweight="bold")
    ax_demo.set_xlabel("示範特徵變數 X (Arbitrary Feature)", fontsize=10)
    ax_demo.set_ylabel("示範目標變數 y (Synthetic Target)", fontsize=10)
    ax_demo.legend(fontsize=9)
    ax_demo.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    synth_chart_path = os.path.join(OUTPUT_DIR, "part1_synthetic_demo_scatter.png")
    fig_demo.savefig(synth_chart_path, dpi=300)
    plt.close(fig_demo)
    print(f"[圖表輸出] 教學示範用隨機散佈圖保存至：{synth_chart_path}")

    # 2. 繪製【真實資料特徵 vs. 利潤散佈圖】
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    spend_cols = [
        ("R&D Spend", "研發支出 (R&D Spend)"),
        ("Administration", "行政支出 (Administration)"),
        ("Marketing Spend", "行銷支出 (Marketing Spend)")
    ]
    colors = ['#2563eb', '#10b981', '#f59e0b']
    for idx, (col, col_title) in enumerate(spend_cols):
        ax = axes[idx]
        ax.scatter(df[col], df['Profit'], color=colors[idx], alpha=0.8, edgecolors='k', s=50)
        ax.set_title(f"{col_title} vs. 獲利 (Profit)", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel(f"{col_title} ($)", fontsize=10)
        ax.set_ylabel("企業獲利 Profit ($)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    scatter_chart_path = os.path.join(OUTPUT_DIR, "part1_scatter_plots.png")
    plt.savefig(scatter_chart_path, dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "real_data_scatters.png"), dpi=300)
    plt.close()
    print(f"[圖表輸出] 真實支出與獲利散點圖保存至：{scatter_chart_path}")

    # ---------------------------------------------------------
    # C. Data Preparation (資料準備與嚴格無洩漏切分)
    # ---------------------------------------------------------
    print("\n--- C. Data Preparation (資料準備：嚴格先切分後前處理) ---")

    # 1. 檢查並處理目標欄位 y 的無效值（若有缺失則移除列，不填補 y）
    valid_y_mask = df['Profit'].notnull() & np.isfinite(pd.to_numeric(df['Profit'], errors='coerce'))
    dropped_y_count = int((~valid_y_mask).sum())
    if dropped_y_count > 0:
        print(f"[警告] 移除目標值 y 缺失或無效之列數：{dropped_y_count} 筆")
        df_clean = df[valid_y_mask].copy()
    else:
        print("目標欄位 Profit 完整無缺失，無需移除無效列。")
        df_clean = df.copy()

    X_raw = df_clean.drop(columns=['Profit'])
    y_raw = df_clean['Profit'].values

    num_features = ['R&D Spend', 'Administration', 'Marketing Spend']
    cat_features = ['State']

    # 2. 嚴格切分：80% 訓練集 (40 筆)、20% 測試集 (10 筆)，鎖定 random_state=0
    # 確保原始索引 tracking
    train_idx, test_idx = train_test_split(df_clean.index, test_size=0.2, random_state=0)
    train_idx = sorted(train_idx.tolist())
    test_idx = sorted(test_idx.tolist())

    X_train_df = X_raw.loc[train_idx].copy()
    y_train = y_raw[train_idx]
    X_test_df = X_raw.loc[test_idx].copy()
    y_test = y_raw[test_idx]

    print(f"資料切分完成 (test_size=0.2, random_state=0)：")
    print(f"  訓練集 (Train)：{len(train_idx)} 筆 (原始索引: {train_idx[:5]}...)")
    print(f"  測試集 (Test) ：{len(test_idx)} 筆 (原始索引: {test_idx})")
    assert set(train_idx).isdisjoint(set(test_idx)), "訓練集與測試集索引不可重疊！"

    # 3. 建立前處理器（僅在訓練集學習參數，杜絕資料洩漏）
    # - 數值型：SimpleImputer(strategy='median')
    # - 類別型：OneHotEncoder(drop='first', handle_unknown='ignore') 避免 Dummy Variable Trap 且支援未知類別
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features)
        ],
        remainder='drop'
    )

    # 4. 建立端到端 Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression(fit_intercept=True))
    ])

    # ---------------------------------------------------------
    # D. Modeling (多元線性回歸建模)
    # ---------------------------------------------------------
    print("\n--- D. Modeling (端到端 Pipeline 擬合與訓練集統計推論) ---")
    # 僅使用訓練集擬合 Pipeline
    full_pipeline.fit(X_train_df, y_train)

    regressor = full_pipeline.named_steps['regressor']
    prep = full_pipeline.named_steps['preprocessor']

    # 取得編碼後特徵名稱
    cat_encoder = prep.named_transformers_['cat']
    ohe_cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    encoded_feature_names = num_features + ohe_cat_names
    print(f"模型特徵清單 ({len(encoded_feature_names)} 項)：{encoded_feature_names}")

    intercept = float(regressor.intercept_)
    coefficients = [float(c) for c in regressor.coef_]
    coef_dict = dict(zip(encoded_feature_names, coefficients))

    print(f"模型截距 (Intercept, b0)：{intercept:.4f}")
    print("回歸係數 (Coefficients)：")
    for feat, coef in coef_dict.items():
        print(f"  - {feat:16s}: {coef:12.4f}")

    # 訓練集 OLS 統計推論量計算
    X_train_transformed = prep.transform(X_train_df)
    y_train_pred = regressor.predict(X_train_transformed)

    n_train = len(y_train)
    p_feat = len(encoded_feature_names)
    df_resid = n_train - p_feat - 1

    rss_train = float(np.sum((y_train - y_train_pred) ** 2))
    tss_train = float(np.sum((y_train - np.mean(y_train)) ** 2))
    sigma2 = rss_train / df_resid

    train_r2 = float(1.0 - rss_train / tss_train) if tss_train > 1e-12 else 0.0
    train_adj_r2 = float(1.0 - (1.0 - train_r2) * (n_train - 1) / df_resid) if df_resid > 0 else train_r2
    f_stat = float(((tss_train - rss_train) / p_feat) / (rss_train / df_resid)) if df_resid > 0 and rss_train > 0 else 0.0
    f_pvalue = float(1.0 - stats.f.cdf(f_stat, p_feat, df_resid)) if df_resid > 0 else 1.0

    X_design = np.column_stack([np.ones(n_train), X_train_transformed])
    cov_beta = sigma2 * np.linalg.pinv(X_design.T @ X_design)
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 1e-12))

    beta_vec = np.concatenate([[intercept], regressor.coef_])
    t_stats = beta_vec / se_beta
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=df_resid))

    param_names = ["Intercept"] + encoded_feature_names
    param_stats = {}
    for name, b, s, t, pv in zip(param_names, beta_vec, se_beta, t_stats, p_values):
        param_stats[name] = {
            "coef": float(b),
            "stderr": float(s),
            "tstat": float(t),
            "pvalue": float(pv)
        }

    # 動態產生預測方程式字串
    formula_terms = [f"{intercept:.2f}"]
    for feat, coef in coef_dict.items():
        sign = "+" if coef >= 0 else "-"
        formula_terms.append(f"{sign} {abs(coef):.4f} * ({feat})")
    regression_formula = "Profit = " + " ".join(formula_terms)
    print(f"\n【實測回歸方程式】：\n{regression_formula}")

    # ---------------------------------------------------------
    # E. Evaluation (獨立測試集評估與視覺化)
    # ---------------------------------------------------------
    print("\n--- E. Evaluation (獨立測試集評估) ---")
    y_test_pred = full_pipeline.predict(X_test_df)

    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_mse = float(mean_squared_error(y_train, y_train_pred))
    train_rmse = float(np.sqrt(train_mse))

    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_mse = float(mean_squared_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(test_mse))
    test_r2 = float(r2_score(y_test, y_test_pred))

    n_test = len(y_test)
    df_test_resid = n_test - p_feat - 1
    test_adj_r2 = float(1.0 - (1.0 - test_r2) * (n_test - 1) / df_test_resid) if df_test_resid > 0 else test_r2

    print(f"訓練集 (Train): RMSE = {train_rmse:.4f}, R2 = {train_r2:.6f}, Adj R2 = {train_adj_r2:.6f}")
    print(f"測試集 (Test) : RMSE = {test_rmse:.4f}, R2 = {test_r2:.6f}, MAE = {test_mae:.4f}")

    # 輸出逐筆預測
    test_preds_list = []
    for i, orig_idx in enumerate(test_idx):
        row_raw = df_clean.loc[orig_idx]
        act = float(y_test[i])
        prd = float(y_test_pred[i])
        res = float(act - prd)
        test_preds_list.append({
            "original_id": int(orig_idx),
            "state": str(row_raw['State']),
            "rd_spend": float(row_raw['R&D Spend']),
            "admin_spend": float(row_raw['Administration']),
            "marketing_spend": float(row_raw['Marketing Spend']),
            "actual_profit": act,
            "predicted_profit": prd,
            "residual": res,
            "abs_error": abs(res)
        })

    test_preds_df = pd.DataFrame(test_preds_list)
    preds_csv_path = os.path.join(OUTPUT_DIR, "part1_test_predictions.csv")
    test_preds_df.to_csv(preds_csv_path, index=False)
    print(f"[逐筆預測 CSV] 已輸出至：{preds_csv_path}")

    # 繪製 Actual vs Predicted 與 Residuals 圖表
    fig_eval, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 子圖 1: 實際 vs. 預測
    ax1.scatter(y_test, y_test_pred, color='#0284c7', edgecolors='k', s=80, alpha=0.9, label='測試觀測值 (Test)')
    lims = [min(min(y_test), min(y_test_pred)) - 5000, max(max(y_test), max(y_test_pred)) + 5000]
    ax1.plot(lims, lims, 'r--', lw=2, label='理想完美擬合線 (y = x)')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel('實際獲利 Actual Profit ($)', fontsize=11)
    ax1.set_ylabel('預測獲利 Predicted Profit ($)', fontsize=11)
    ax1.set_title('Part 1: 實際獲利 vs. 預測獲利 (Actual vs. Predicted)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 子圖 2: 殘差圖
    residuals = y_test - y_test_pred
    ax2.scatter(y_test_pred, residuals, color='#10b981', edgecolors='k', s=80, alpha=0.9)
    ax2.axhline(0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('預測獲利 Fitted Value ($)', fontsize=11)
    ax2.set_ylabel('殘差 Residual ($)', fontsize=11)
    ax2.set_title('Part 1: 殘差分佈圖 (Residuals Plot)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    eval_chart_path = os.path.join(OUTPUT_DIR, "part1_evaluation_plots.png")
    fig_eval.savefig(eval_chart_path, dpi=300)
    plt.close(fig_eval)
    print(f"[圖表輸出] 評估圖表已保存至：{eval_chart_path}")

    # 個別保存單獨圖表
    fig_avp, ax_avp = plt.subplots(figsize=(7, 5.5))
    ax_avp.scatter(y_test, y_test_pred, color='#0284c7', edgecolors='k', s=80, alpha=0.9, label='測試觀測值 (Test)')
    ax_avp.plot(lims, lims, 'r--', lw=2, label='理想完美擬合線 (y = x)')
    ax_avp.set_xlim(lims)
    ax_avp.set_ylim(lims)
    ax_avp.set_xlabel('實際獲利 Actual Profit ($)', fontsize=11)
    ax_avp.set_ylabel('預測獲利 Predicted Profit ($)', fontsize=11)
    ax_avp.set_title('實際獲利 vs. 預測獲利 (Actual vs. Predicted)', fontsize=12, fontweight='bold')
    ax_avp.legend(fontsize=10)
    ax_avp.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig_avp.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=300)
    plt.close(fig_avp)

    fig_res, ax_res = plt.subplots(figsize=(7, 5.5))
    ax_res.scatter(y_test_pred, residuals, color='#10b981', edgecolors='k', s=80, alpha=0.9)
    ax_res.axhline(0, color='r', linestyle='--', lw=2)
    ax_res.set_xlabel('預測獲利 Fitted Value ($)', fontsize=11)
    ax_res.set_ylabel('殘差 Residual ($)', fontsize=11)
    ax_res.set_title('殘差分佈圖 (Residuals Plot)', fontsize=12, fontweight='bold')
    ax_res.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig_res.savefig(os.path.join(OUTPUT_DIR, "residuals_plot.png"), dpi=300)
    plt.close(fig_res)

    # ---------------------------------------------------------
    # F. Deployment (完整 Pipeline 保存與端到端推論驗證)
    # ---------------------------------------------------------
    print("\n--- F. Deployment (Pipeline 序列化與重載推論驗證) ---")
    deployment_package = {
        "pipeline": full_pipeline,
        "model": full_pipeline.named_steps["regressor"],
        "preprocessor": full_pipeline.named_steps["preprocessor"],
        "column_transformer": full_pipeline.named_steps["preprocessor"],
        "raw_features_order": list(X_raw.columns),
        "encoded_features_order": encoded_feature_names,
        "intercept": intercept,
        "coefficients": coef_dict,
        "experiment_config": {
            "test_size": 0.2,
            "random_state": 0,
            "train_sample_count": len(train_idx),
            "test_sample_count": len(test_idx),
            "total_sample_count": total_rows
        }
    }

    model_path = os.path.join(OUTPUT_DIR, "part1_linear_regression_model.pkl.gz")
    with gzip.open(model_path, "wb") as f:
        pickle.dump(deployment_package, f)
    print(f"[模型儲存] 完整 Pipeline 套件已保存至：{model_path}")

    # 重新載入驗證：直接使用原始 DataFrame 輸入
    with gzip.open(model_path, "rb") as f:
        loaded_pkg = pickle.load(f)
    loaded_pipe = loaded_pkg["pipeline"]

    reloaded_preds = loaded_pipe.predict(X_test_df)
    max_diff = float(np.max(np.abs(reloaded_preds - y_test_pred)))
    assert max_diff < 1e-9, f"原始 DataFrame 輸入重載推論不一致！Max diff: {max_diff}"
    print(f"[驗證通過] 重載 Pipeline 接受原始欄位推論，誤差為 {max_diff:.2e} (完全一致)！")

    # 未知類別容錯驗證 (handle_unknown='ignore')
    unknown_df = pd.DataFrame([{
        "R&D Spend": 100000,
        "Administration": 120000,
        "Marketing Spend": 200000,
        "State": "Texas"  # 未曾出現的類別
    }])
    try:
        unk_pred = loaded_pipe.predict(unknown_df)
        print(f"[未知類別容錯驗證通過] 出現新類別 'Texas' 時自動編碼為基準，預測利潤為：${unk_pred[0]:,.2f}")
    except Exception as e:
        raise AssertionError(f"未知類別導致崩潰：{e}")

    part1_summary = {
        "business_context": business_context,
        "data_summary": {
            "total_rows": total_rows,
            "dropped_y_rows": dropped_y_count,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "null_info": null_info,
            "duplicate_count": dup_count
        },
        "model_parameters": {
            "intercept": intercept,
            "coefficients": coef_dict,
            "formula": regression_formula
        },
        "train_metrics": {
            "MAE": train_mae,
            "MSE": train_mse,
            "RMSE": train_rmse,
            "R2": train_r2,
            "Adjusted_R2": train_adj_r2,
            "F_statistic": f_stat,
            "F_pvalue": f_pvalue
        },
        "test_metrics": {
            "MAE": test_mae,
            "MSE": test_mse,
            "RMSE": test_rmse,
            "R2": test_r2,
            "Adjusted_R2": test_adj_r2
        },
        "param_stats": param_stats,
        "test_predictions": test_preds_list
    }

    return part1_summary


if __name__ == "__main__":
    run_part1()
