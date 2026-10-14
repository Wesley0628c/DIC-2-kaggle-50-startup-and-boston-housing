"""
模組名稱：part1_crispdm.py
說明：實作 Part 1 CRISP-DM 完整流程：
  A. Business Understanding (商業理解)
  B. Data Understanding (資料理解與探索，散點圖)
  C. Data Preparation (State 類別編碼，刪除基準州虛擬欄位，train_test_split test_size=0.2, random_state=0)
  D. Modeling (LinearRegression 多元線性回歸)
  E. Evaluation (MAE, MSE, RMSE, R² 評估指標、實際值 vs 預測值圖、殘差圖、逐筆評估)
  F. Deployment (pickle + gzip 序列化與相容性驗證)
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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 設定中英相容字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang TC', 'Heiti TC', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "50_Startups.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def run_part1():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("【Part 1】CRISP-DM 多元線性回歸 (Multiple Linear Regression) 啟動")
    print("=" * 60)

    # ---------------------------------------------------------
    # A. Business Understanding (商業理解)
    # ---------------------------------------------------------
    print("\n--- A. Business Understanding (商業理解) ---")
    business_context = {
        "problem": "創投基金與企業管理者希望依據 50 家初創企業在研發 (R&D Spend)、行政管理 (Administration)、市場行銷 (Marketing Spend) 之支出與所在州 (State)，建立多元線性回歸模型以預測企業獲利能力 (Profit)。",
        "variables": {
            "features_X": "R&D Spend, Administration, Marketing Spend, State",
            "target_y": "Profit (預測目標)"
        },
        "caution": "回歸模型展現的是變數間的統計相關性 (Statistical Association)，並不能直接推論為因果關係 (Causality)。"
    }
    print(business_context["problem"])
    print(f"特徵 X：{business_context['variables']['features_X']}")
    print(f"目標 y：{business_context['variables']['target_y']}")
    print(f"統計提醒：{business_context['caution']}")

    # ---------------------------------------------------------
    # B. Data Understanding (資料理解)
    # ---------------------------------------------------------
    print("\n--- B. Data Understanding (資料理解) ---")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到資料集檔案：{DATA_PATH}，請先執行 src/download_data.py")
    
    df = pd.read_csv(DATA_PATH)
    total_rows, total_cols = df.shape
    print(f"資料筆數 (Rows)：{total_rows}，原始欄位數 (Columns)：{total_cols}")
    print("\n[前五筆資料 Head(5)]：")
    print(df.head())
    
    desc_stats = df.describe().to_dict()
    print("\n[數值欄位描述統計]：")
    print(df.describe())

    null_info = df.isnull().sum().to_dict()
    duplicate_count = int(df.duplicated().sum())
    state_distribution = df['State'].value_counts().to_dict()
    print(f"缺失值狀態：{null_info}")
    print(f"重複列數：{duplicate_count}")
    print(f"State 類別分佈：{state_distribution}")

    # 繪製資料探索散點圖 (支出 vs 利潤)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    spend_cols = [("R&D Spend", "研發支出 (R&D Spend)"), 
                  ("Administration", "行政支出 (Administration)"), 
                  ("Marketing Spend", "行銷支出 (Marketing Spend)")]
    
    colors = ['#2563eb', '#10b981', '#f59e0b']
    for idx, (col, col_title) in enumerate(spend_cols):
        ax = axes[idx]
        ax.scatter(df[col], df['Profit'], color=colors[idx], alpha=0.75, edgecolors='none', s=45)
        ax.set_title(f"{col_title} vs 利潤 (Profit)", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(f"{col_title} ($)", fontsize=10)
        ax.set_ylabel("獲利 Profit ($)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    scatter_chart_path = os.path.join(OUTPUT_DIR, "part1_scatter_plots.png")
    plt.savefig(scatter_chart_path, dpi=300)
    plt.close()
    print(f"[圖表輸出] 支出與獲利散點圖保存至：{scatter_chart_path}")

    # ---------------------------------------------------------
    # C. Data Preparation (資料準備)
    # ---------------------------------------------------------
    print("\n--- C. Data Preparation (資料準備) ---")
    # 依照講義標準重現：
    # 原始特徵矩陣 X（4 個欄位），目標向量 y
    X_raw = df.drop(columns=['Profit'])
    y_raw = df['Profit'].values

    # 使用 ColumnTransformer 與 OneHotEncoder 編碼 State
    # 講義說明：若 State 有 3 個類別 (California, Florida, New York)，
    # 為避免虛擬變數陷阱 (Dummy Variable Trap) 引起的多重共線性，需移除第一個虛擬欄位 (drop='first')。
    # 基準州 (Reference State) 即為被刪除的第一個類別。
    
    # 取得 State 的排序類別，確認基準州
    sorted_states = sorted(df['State'].unique())
    reference_state = sorted_states[0] # 字母排序首位：California
    print(f"State 所有類別：{sorted_states}，基準州 (Reference/Base State)：{reference_state}")

    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(drop='first', sparse_output=False), ['State'])],
        remainder='passthrough'
    )
    X_transformed = ct.fit_transform(X_raw)
    
    # 取得編碼後特徵名稱
    ohe_feature_names = ct.named_transformers_['encoder'].get_feature_names_out(['State']).tolist()
    # remainder 包含原數值欄位：['R&D Spend', 'Administration', 'Marketing Spend']
    remainder_cols = [c for c in X_raw.columns if c != 'State']
    encoded_feature_names = ohe_feature_names + remainder_cols
    print(f"編碼後 5 個特徵欄位名稱：{encoded_feature_names}")
    print(f"轉換後特徵矩陣維度：{X_transformed.shape}")

    # 建立 DataFrame 方便追蹤原始列 ID (Original Row Indices)
    X_encoded_df = pd.DataFrame(X_transformed, columns=encoded_feature_names, index=df.index)

    # 執行切分：test_size=0.2, random_state=0
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=0)
    train_idx = sorted(train_idx.tolist())
    test_idx = sorted(test_idx.tolist())

    X_train = X_encoded_df.loc[train_idx].values
    X_test = X_encoded_df.loc[test_idx].values
    y_train = y_raw[train_idx]
    y_test = y_raw[test_idx]

    print(f"訓練集樣本數 (Train)：{len(train_idx)} 筆，測試集樣本數 (Test)：{len(test_idx)} 筆")
    print(f"測試集原始列索引 (Test Indices, 共 {len(test_idx)} 筆)：{test_idx}")
    print(f"訓練集原始列索引 (Train Indices, 共 {len(train_idx)} 筆)：{train_idx}")

    # ---------------------------------------------------------
    # D. Modeling (建模)
    # ---------------------------------------------------------
    print("\n--- D. Modeling (線性回歸建模) ---")
    regressor = LinearRegression(fit_intercept=True)
    regressor.fit(X_train, y_train)

    intercept = float(regressor.intercept_)
    coefficients = [float(c) for c in regressor.coef_]
    coef_dict = dict(zip(encoded_feature_names, coefficients))

    print(f"模型截距 (Intercept, b0)：{intercept:.4f}")
    print("模型迴歸係數 (Coefficients, b1 ~ b5)：")
    for feat, coef in coef_dict.items():
        print(f"  - {feat}: {coef:.4f}")

    # 動態產生預測方程式字串
    formula_terms = [f"{intercept:.2f}"]
    for feat, coef in coef_dict.items():
        sign = "+" if coef >= 0 else "-"
        formula_terms.append(f"{sign} {abs(coef):.4f} * ({feat})")
    regression_formula = "Profit = " + " ".join(formula_terms)
    print(f"\n【實測回歸預測公式】：\n{regression_formula}")

    # ---------------------------------------------------------
    # E. Evaluation (模型評估)
    # ---------------------------------------------------------
    print("\n--- E. Evaluation (模型評估與檢驗) ---")
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_mse = float(mean_squared_error(y_train, y_train_pred))
    train_rmse = float(np.sqrt(train_mse))
    train_r2 = float(r2_score(y_train, y_train_pred))

    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_mse = float(mean_squared_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(test_mse))
    test_r2 = float(r2_score(y_test, y_test_pred))

    # 嚴謹計算訓練集 OLS 統計推論量 (樣本數 n=40, 特徵數 p=5, 自由度 df=34)
    from scipy import stats
    n_train = len(y_train)
    p_feat = X_train.shape[1]
    df_resid = n_train - p_feat - 1
    rss_train = float(np.sum((y_train - y_train_pred) ** 2))
    tss_train = float(np.sum((y_train - np.mean(y_train)) ** 2))
    sigma2 = rss_train / df_resid

    train_adj_r2 = float(1.0 - (1.0 - train_r2) * (n_train - 1) / df_resid)
    f_stat = float(((tss_train - rss_train) / p_feat) / (rss_train / df_resid))
    f_pvalue = float(1.0 - stats.f.cdf(f_stat, p_feat, df_resid))

    # 設計矩陣含有截距項
    X_design = np.column_stack([np.ones(n_train), X_train])
    cov_beta = sigma2 * np.linalg.inv(X_design.T @ X_design)
    se_beta = np.sqrt(np.diag(cov_beta))

    beta_vec = np.concatenate([[intercept], regressor.coef_])
    t_stats = beta_vec / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_resid))

    param_names = ["Intercept"] + encoded_feature_names
    param_stats = {}
    for name, b, s, t, pv in zip(param_names, beta_vec, se_beta, t_stats, p_values):
        param_stats[name] = {
            "coef": float(b),
            "stderr": float(s),
            "tstat": float(t),
            "pvalue": float(pv)
        }

    metrics = {
        "train": {
            "MAE": train_mae,
            "MSE": train_mse,
            "RMSE": train_rmse,
            "R2": train_r2,
            "Adjusted_R2": train_adj_r2,
            "F_statistic": f_stat,
            "F_pvalue": f_pvalue,
            "degrees_of_freedom": {"df_model": p_feat, "df_residuals": df_resid}
        },
        "test": {
            "MAE": test_mae,
            "MSE": test_mse,
            "RMSE": test_rmse,
            "R2": test_r2
        }
    }

    print("\n[評估指標比較表 (Metrics Comparison)]：")
    print(f"{'指標 Metric':<16} | {'訓練集 (Train)':<22} | {'測試集 (Test)':<22}")
    print("-" * 62)
    print(f"{'MAE':<16} | {train_mae:<22.4f} | {test_mae:<22.4f}")
    print(f"{'MSE':<16} | {train_mse:<22.4f} | {test_mse:<22.4f}")
    print(f"{'RMSE':<16} | {train_rmse:<22.4f} | {test_rmse:<22.4f}")
    print(f"{'R² (判定係數)':<14} | {train_r2:<22.4f} | {test_r2:<22.4f}")
    print(f"{'Adjusted R²':<16} | {train_adj_r2:<22.4f} | {'— (非OLS樣本統計量)':<22}")
    print(f"{'F-statistic':<16} | {f'{f_stat:.2f} (p={f_pvalue:.2e})':<22} | {'—':<22}")

    # 逐筆測試集明細 (Row ID, 實際值, 預測值, 殘差, 絕對誤差)
    test_residuals = y_test - y_test_pred
    test_abs_errors = np.abs(test_residuals)

    test_predictions_records = []
    for i, orig_id in enumerate(test_idx):
        test_predictions_records.append({
            "original_id": int(orig_id),
            "actual_profit": float(y_test[i]),
            "predicted_profit": float(y_test_pred[i]),
            "residual": float(test_residuals[i]),
            "abs_error": float(test_abs_errors[i]),
            "state": df.loc[orig_id, "State"],
            "rd_spend": float(df.loc[orig_id, "R&D Spend"]),
            "admin_spend": float(df.loc[orig_id, "Administration"]),
            "marketing_spend": float(df.loc[orig_id, "Marketing Spend"])
        })

    test_pred_df = pd.DataFrame(test_predictions_records)
    test_pred_csv_path = os.path.join(OUTPUT_DIR, "part1_test_predictions.csv")
    test_pred_df.to_csv(test_pred_csv_path, index=False, encoding='utf-8')
    print(f"\n[逐筆測試結果] 已保存至：{test_pred_csv_path}")

    # 繪製 Actual vs Predicted 圖與殘差圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Actual vs Predicted 圖
    min_val = min(y_test.min(), y_test_pred.min()) * 0.9
    max_val = max(y_test.max(), y_test_pred.max()) * 1.1
    ax1.scatter(y_test, y_test_pred, color='#2563eb', s=60, edgecolors='black', linewidth=0.8, alpha=0.85, label='測試樣本 (Test Data)')
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='理想線 (Ideal: y = x)')
    ax1.set_title("Part 1: 實際值 vs. 預測值 (Actual vs Predicted)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("實際利潤 Actual Profit ($)", fontsize=11)
    ax1.set_ylabel("模型預測利潤 Predicted Profit ($)", fontsize=11)
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', frameon=True)

    # 2. 殘差圖 (Residual plot: y_test - y_pred vs y_pred)
    ax2.scatter(y_test_pred, test_residuals, color='#dc2626', s=60, edgecolors='black', linewidth=0.8, alpha=0.85)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.2)
    ax2.set_title("Part 1: 殘差分佈圖 (Residual Plot: Actual - Predicted)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("預測利潤 Predicted Profit ($)", fontsize=11)
    ax2.set_ylabel("殘差 Residuals ($)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    eval_chart_path = os.path.join(OUTPUT_DIR, "part1_evaluation_plots.png")
    plt.savefig(eval_chart_path, dpi=300)
    plt.close()
    print(f"[圖表輸出] 評估圖表已保存至：{eval_chart_path}")

    # ---------------------------------------------------------
    # F. Deployment (模型保存與重載驗證)
    # ---------------------------------------------------------
    print("\n--- F. Deployment (模型打包與驗證) ---")
    deployment_package = {
        "column_transformer": ct,
        "raw_features_order": list(X_raw.columns),
        "encoded_features_order": encoded_feature_names,
        "reference_state": reference_state,
        "model": regressor,
        "intercept": intercept,
        "coefficients": coef_dict,
        "experiment_config": {
            "test_size": 0.2,
            "random_state": 0,
            "train_sample_count": len(train_idx),
            "test_sample_count": len(test_idx),
            "total_sample_count": total_rows
        },
        "valid_states": sorted_states
    }

    model_path = os.path.join(OUTPUT_DIR, "part1_linear_regression_model.pkl.gz")
    with gzip.open(model_path, "wb") as f:
        pickle.dump(deployment_package, f)
    print(f"[模型儲存] 完整預測套件已使用 pickle + gzip 保存至：{model_path}")

    # 重新載入驗證
    with gzip.open(model_path, "rb") as f:
        loaded_package = pickle.load(f)
    
    loaded_model = loaded_package["model"]
    loaded_ct = loaded_package["column_transformer"]
    test_reloaded_pred = loaded_model.predict(X_test)
    
    max_diff = np.max(np.abs(test_reloaded_pred - y_test_pred))
    assert max_diff < 1e-9, f"模型重載後預測不一致！最大誤差：{max_diff}"
    print(f"[驗證通過] 模型重新載入後，預測結果與原模型完全一致 (Max Diff = {max_diff:.2e})。")

    # 未知州別防呆驗證
    def predict_new_startup(raw_data_df: pd.DataFrame):
        for s in raw_data_df['State']:
            if s not in loaded_package["valid_states"]:
                raise ValueError(f"不支援的未知州別：'{s}'！有效州別為：{loaded_package['valid_states']}")
        x_enc = loaded_ct.transform(raw_data_df[loaded_package["raw_features_order"]])
        return loaded_model.predict(x_enc)

    try:
        sample_unknown = pd.DataFrame([{
            "R&D Spend": 100000,
            "Administration": 120000,
            "Marketing Spend": 200000,
            "State": "Texas"  # 未知州別
        }])
        predict_new_startup(sample_unknown)
        print("[異常警告] 未知州別竟然未引發錯誤！")
    except ValueError as e:
        print(f"[未知州別防呆驗證通過] 正確捕捉例外：{e}")

    # 回傳供整體 pipeline 或 JSON 匯出使用的字典結構
    part1_summary = {
        "business_context": business_context,
        "data_summary": {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "raw_features": list(X_raw.columns),
            "encoded_features": encoded_feature_names,
            "reference_state": reference_state,
            "state_distribution": state_distribution,
            "null_info": null_info,
            "duplicate_count": duplicate_count,
            "describe": desc_stats
        },
        "split_info": {
            "test_size": 0.2,
            "random_state": 0,
            "train_indices": train_idx,
            "test_indices": test_idx
        },
        "model_parameters": {
            "intercept": intercept,
            "coefficients": coef_dict,
            "parameter_statistics": param_stats,
            "formula": regression_formula
        },
        "metrics": metrics,
        "test_predictions": test_predictions_records
    }

    return part1_summary

if __name__ == "__main__":
    run_part1()
