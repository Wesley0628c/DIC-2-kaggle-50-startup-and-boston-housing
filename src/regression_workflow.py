"""
模組名稱：regression_workflow.py
用途：可重用的多元線性回歸 (Multiple Linear Regression) 核心工作流程
支援任意數值回歸資料集（如 50 Startups, Boston Housing, 自訂資料集）
流程遵循 CRISP-DM 規範與嚴謹機器學習標準：
  1. Data Acquisition & Validation: 資料讀取與品質檢驗，檢測並剔除無效目標值 y
  2. Strict Train-Test Split First: 嚴格先切分 80% 訓練集與 20% 測試集，前處理參數純粹從訓練集學習
  3. Preprocessing Pipeline: 使用 ColumnTransformer + SimpleImputer(median) + OneHotEncoder(drop='first', handle_unknown='ignore')
  4. End-to-End Pipeline Modeling: 串接 LinearRegression，訓練集 5-Fold 交叉驗證與統計推論檢定
  5. Comprehensive Evaluation: MAE, MSE, RMSE, R², 殘差檢驗, 驗證 RMSE = sqrt(MSE)
  6. Deployment & Parity Verification: gzip+pickle 序列化完整 Pipeline，驗證重新載入後接受原始 DataFrame 推論完全一致
  7. Production Maintenance & Monitoring Plan: 提供資料漂移、未知類別率與真實指標監控藍圖
"""

import os
import json
import gzip
import pickle
import hashlib
import sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 設定圖表通用字型與清晰度
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang TC', 'Heiti TC', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RegressionWorkflow:
    def __init__(self, dataset_name, data_source, target_column, feature_columns=None, 
                 categorical_columns=None, test_size=0.2, random_state=0, output_dir=None):
        self.dataset_name = dataset_name
        self.data_source = data_source
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        self.raw_df = None
        self.train_df = None
        self.test_df = None
        self.X_train_df = None
        self.X_test_df = None
        self.y_train = None
        self.y_test = None
        self.train_indices = None
        self.test_indices = None
        
        self.numeric_cols = []
        self.cat_cols = []
        self.encoded_feature_names = []
        
        self.full_pipeline = None
        self.results = {}
        self.data_hash = None

    def step1_load_and_validate(self):
        """Step 1: 讀取資料、計算檔案雜湊並進行品質檢核"""
        if isinstance(self.data_source, pd.DataFrame):
            self.raw_df = self.data_source.copy()
            self.data_hash = hashlib.sha256(pd.util.hash_pandas_object(self.raw_df).values).hexdigest()[:16]
        elif isinstance(self.data_source, str):
            if not os.path.exists(self.data_source):
                raise FileNotFoundError(f"找不到資料集檔案：{self.data_source}")
            with open(self.data_source, "rb") as f:
                self.data_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            self.raw_df = pd.read_csv(self.data_source)
        else:
            raise TypeError("data_source 必須為檔案路徑或 pandas.DataFrame")

        if self.target_column not in self.raw_df.columns:
            raise ValueError(f"目標欄位 '{self.target_column}' 不在資料集欄位中！可用欄位：{list(self.raw_df.columns)}")

        # 檢驗目標值 y 之有效性，剔除缺失列
        valid_y_mask = self.raw_df[self.target_column].notnull() & np.isfinite(pd.to_numeric(self.raw_df[self.target_column], errors='coerce'))
        dropped_y_count = int((~valid_y_mask).sum())
        if dropped_y_count > 0:
            print(f"[{self.dataset_name}] 警告：目標變數 y 有 {dropped_y_count} 筆無效/缺失，已予以剔除（不填補目標值）。")
            df_clean = self.raw_df[valid_y_mask].copy()
        else:
            df_clean = self.raw_df.copy()

        total_rows, total_cols = df_clean.shape
        missing_count = int(df_clean.isnull().sum().sum())
        duplicate_count = int(df_clean.duplicated().sum())

        quality_report = {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "dropped_invalid_y_rows": dropped_y_count,
            "missing_values": missing_count,
            "duplicates": duplicate_count,
            "columns": list(df_clean.columns),
            "dtypes": {col: str(dtype) for col, dtype in df_clean.dtypes.items()},
            "data_sha256_prefix": self.data_hash
        }
        self.results["quality_report"] = quality_report
        self.cleaned_df = df_clean
        return df_clean

    def step2_prepare_data(self):
        """Step 2: 嚴格先切分資料集，再構建以訓練集為基準的預處理 Pipeline"""
        df = self.cleaned_df.copy()

        # 確定特徵欄位清單
        if self.feature_columns is None:
            features = [c for c in df.columns if c != self.target_column]
        else:
            features = [c for c in self.feature_columns if c in df.columns and c != self.target_column]

        # 切分訓練集與測試集 (使用 index 切分以鎖定原始索引)
        train_idx, test_idx = train_test_split(
            df.index, test_size=self.test_size, random_state=self.random_state
        )
        self.train_indices = sorted(train_idx.tolist())
        self.test_indices = sorted(test_idx.tolist())

        self.train_df = df.loc[self.train_indices].copy()
        self.test_df = df.loc[self.test_indices].copy()

        self.X_train_df = self.train_df[features].copy()
        self.X_test_df = self.test_df[features].copy()
        self.y_train = self.train_df[self.target_column].values.astype(float)
        self.y_test = self.test_df[self.target_column].values.astype(float)

        # 從訓練集判定數值與類別特徵
        if self.categorical_columns is None:
            self.cat_cols = [c for c in features if self.X_train_df[c].dtype == 'object' or self.X_train_df[c].dtype.name == 'category']
        else:
            self.cat_cols = [c for c in self.categorical_columns if c in features]
        self.numeric_cols = [c for c in features if c not in self.cat_cols]

        # 構建前處理器：數值採中位數填補，類別採 drop='first' + handle_unknown='ignore'
        transformers = []
        if len(self.numeric_cols) > 0:
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ])
            transformers.append(('num', num_pipe, self.numeric_cols))

        if len(self.cat_cols) > 0:
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_pipe, self.cat_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers)

        self.results["split_info"] = {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "train_rows": len(self.train_indices),
            "test_rows": len(self.test_indices),
            "raw_feature_count": len(features),
            "raw_feature_names": features,
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.cat_cols,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices
        }
        return self.X_train_df, self.X_test_df, self.y_train, self.y_test

    def step3_train_and_inference(self):
        """Step 3: 擬合端到端 Pipeline 並進行訓練集統計檢定與 5-Fold CV"""
        self.full_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', LinearRegression(fit_intercept=True))
        ])

        # 擬合 Pipeline (內部預處理器只接觸 X_train_df)
        self.full_pipeline.fit(self.X_train_df, self.y_train)

        # 提取編碼後特徵名稱
        prep = self.full_pipeline.named_steps['preprocessor']
        encoded_names = []
        if len(self.numeric_cols) > 0:
            encoded_names.extend(self.numeric_cols)
        if len(self.cat_cols) > 0:
            cat_encoder = prep.named_transformers_['cat'].named_steps['onehot']
            cat_names = cat_encoder.get_feature_names_out(self.cat_cols).tolist()
            encoded_names.extend(cat_names)
        self.encoded_feature_names = encoded_names

        reg = self.full_pipeline.named_steps['regressor']
        intercept = float(reg.intercept_)
        coefficients = {name: float(coef) for name, coef in zip(self.encoded_feature_names, reg.coef_)}

        # 訓練集預測
        y_train_pred = self.full_pipeline.predict(self.X_train_df)
        n = len(self.y_train)
        p = len(self.encoded_feature_names)
        df_residuals = n - p - 1

        residuals = self.y_train - y_train_pred
        sse = float(np.sum(residuals ** 2))
        sst = float(np.sum((self.y_train - np.mean(self.y_train)) ** 2))
        train_r2 = float(1.0 - sse / sst) if sst > 1e-12 else 0.0

        # 訓練集 5-Fold 交叉驗證
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = -cross_val_score(
            self.full_pipeline, self.X_train_df, self.y_train, cv=kf, scoring='neg_root_mean_squared_error'
        )
        cv_rmse_mean = float(np.mean(cv_scores))
        cv_rmse_std = float(np.std(cv_scores))

        # 參數標準誤與 t 檢定
        param_stats = {}
        if df_residuals > 0:
            sigma_sq = sse / df_residuals
            X_trans = prep.transform(self.X_train_df)
            X_design = np.column_stack([np.ones(n), X_trans])
            try:
                cov_matrix = sigma_sq * np.linalg.pinv(X_design.T @ X_design)
                std_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))

                t_b0 = intercept / std_errors[0]
                p_b0 = 2.0 * (1.0 - stats.t.cdf(abs(t_b0), df=df_residuals))
                param_stats["Intercept"] = {
                    "coef": round(intercept, 4),
                    "stderr": round(float(std_errors[0]), 4),
                    "tstat": round(float(t_b0), 4),
                    "pvalue": float(np.format_float_scientific(p_b0, precision=4))
                }

                for j, feat in enumerate(self.encoded_feature_names, start=1):
                    coef_val = float(reg.coef_[j - 1])
                    se_val = float(std_errors[j])
                    t_val = coef_val / se_val if se_val > 1e-12 else 0.0
                    pval = 2.0 * (1.0 - stats.t.cdf(abs(t_val), df=df_residuals))
                    param_stats[feat] = {
                        "coef": round(coef_val, 4),
                        "stderr": round(se_val, 4),
                        "tstat": round(t_val, 4),
                        "pvalue": float(np.format_float_scientific(pval, precision=4))
                    }
            except Exception:
                pass

        # 全局 F 檢定
        f_stat = 0.0
        f_pvalue = 1.0
        if df_residuals > 0 and p > 0 and (1.0 - train_r2) > 1e-12:
            f_stat = float((train_r2 / p) / ((1.0 - train_r2) / df_residuals))
            f_pvalue = float(1.0 - stats.f.cdf(f_stat, p, df_residuals))

        adj_r2 = float(1.0 - ((1.0 - train_r2) * (n - 1) / df_residuals)) if df_residuals > 0 else train_r2

        # 構建方程式字串
        terms = [f"{intercept:.4f}"]
        for name, coef in coefficients.items():
            sign = "+" if coef >= 0 else "-"
            terms.append(f"{sign} {abs(coef):.4f} * ({name})")
        formula_str = f"{self.target_column} = " + " ".join(terms)

        self.results["model_params"] = {
            "intercept": intercept,
            "coefficients": coefficients,
            "formula": formula_str,
            "encoded_feature_names": self.encoded_feature_names,
            "parameter_statistics": param_stats,
            "cv_metrics": {
                "cv_rmse_mean": cv_rmse_mean,
                "cv_rmse_std": cv_rmse_std
            },
            "train_statistics": {
                "r2": train_r2,
                "adj_r2": adj_r2,
                "f_stat": round(f_stat, 2),
                "f_pvalue": float(np.format_float_scientific(f_pvalue, precision=4)),
                "df_model": p,
                "df_residuals": df_residuals
            }
        }
        return y_train_pred

    def step4_evaluate(self):
        """Step 4: 在未參與訓練的獨立測試集進行評估與逐筆預測"""
        y_train_pred = self.full_pipeline.predict(self.X_train_df)
        y_test_pred = self.full_pipeline.predict(self.X_test_df)

        train_mae = float(mean_absolute_error(self.y_train, y_train_pred))
        train_mse = float(mean_squared_error(self.y_train, y_train_pred))
        train_rmse = float(np.sqrt(train_mse))
        train_r2 = float(r2_score(self.y_train, y_train_pred))

        test_mae = float(mean_absolute_error(self.y_test, y_test_pred))
        test_mse = float(mean_squared_error(self.y_test, y_test_pred))
        test_rmse = float(np.sqrt(test_mse))
        test_r2 = float(r2_score(self.y_test, y_test_pred))

        test_predictions = []
        for i, (orig_id, actual, pred) in enumerate(zip(self.test_indices, self.y_test, y_test_pred)):
            residual = float(actual - pred)
            test_predictions.append({
                "original_id": int(orig_id),
                "actual": float(actual),
                "predicted": float(pred),
                "residual": residual,
                "abs_error": abs(residual)
            })

        self.results["metrics"] = {
            "train": {
                "mae": train_mae,
                "mse": train_mse,
                "rmse": train_rmse,
                "r2": train_r2,
                "adj_r2": self.results["model_params"]["train_statistics"]["adj_r2"]
            },
            "test": {
                "mae": test_mae,
                "mse": test_mse,
                "rmse": test_rmse,
                "r2": test_r2
            },
            "cv": self.results["model_params"]["cv_metrics"]
        }
        self.results["test_predictions"] = test_predictions
        return self.results["metrics"]

    def step5_plot_and_export(self):
        """Step 5: 繪製評估圖表、序列化 Pipeline 模型並驗證原始資料推論完全一致"""
        prefix = self.dataset_name.lower().replace(" ", "_")

        # 1. 繪製 Actual vs Predicted 與 Residuals 圖表
        y_test_pred = self.full_pipeline.predict(self.X_test_df)
        residuals = self.y_test - y_test_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].scatter(self.y_test, y_test_pred, color="#0284c7", s=70, alpha=0.85, edgecolors="k", label="Test Observations")
        lims = [min(min(self.y_test), min(y_test_pred)) * 0.95, max(max(self.y_test), max(y_test_pred)) * 1.05]
        axes[0].plot(lims, lims, "r--", lw=2, label="Ideal Fit (y = x)")
        axes[0].set_xlim(lims)
        axes[0].set_ylim(lims)
        axes[0].set_title(f"Actual vs. Predicted: {self.dataset_name}", fontsize=12, fontweight="bold")
        axes[0].set_xlabel(f"Actual {self.target_column}", fontsize=11)
        axes[0].set_ylabel(f"Predicted {self.target_column}", fontsize=11)
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].scatter(y_test_pred, residuals, color="#ef4444", s=70, alpha=0.85, edgecolors="k")
        axes[1].axhline(0, color="k", linestyle="--", lw=1.5)
        axes[1].set_title(f"Residual Plot: {self.dataset_name}", fontsize=12, fontweight="bold")
        axes[1].set_xlabel(f"Predicted {self.target_column}", fontsize=11)
        axes[1].set_ylabel("Residual (Actual - Pred)", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{prefix}_evaluation_plots.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        self.results["plot_path"] = plot_path

        # 2. 序列化完整 Pipeline (gzip + pickle)
        model_pkg = {
            "dataset_name": self.dataset_name,
            "target_column": self.target_column,
            "feature_names": self.encoded_feature_names,
            "raw_feature_names": self.results["split_info"]["raw_feature_names"],
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.cat_cols,
            "pipeline": self.full_pipeline,
            "model": self.full_pipeline.named_steps["regressor"],
            "preprocessor": self.full_pipeline.named_steps["preprocessor"],
            "column_transformer": self.full_pipeline.named_steps["preprocessor"],
            "model_metadata": {
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "data_sha256": self.data_hash,
                "random_state": self.random_state,
                "test_size": self.test_size
            },
            "split_info": self.results["split_info"],
            "model_params": self.results["model_params"],
            "metrics": self.results["metrics"],
            "production_maintenance_plan": {
                "implemented_checks": [
                    "Input schema verification (missing columns detection)",
                    "Automatic median imputation for numerical gaps",
                    "One-Hot encoding with reference category dropped and unknown category ignore fallback"
                ],
                "future_production_monitoring": [
                    "Feature distribution drift (Evidently AI / KS-test)",
                    "Target drift tracking when ground truth labels arrive",
                    "Prediction latency and memory footprint monitoring"
                ]
            }
        }
        model_path = os.path.join(self.output_dir, f"{prefix}_linear_regression_model.pkl.gz")
        with gzip.open(model_path, "wb") as f:
            pickle.dump(model_pkg, f)
        self.results["model_path"] = model_path

        # 3. 測試重載推論完全一致性 (以原始 DataFrame 測試)
        with gzip.open(model_path, "rb") as f:
            reloaded_pkg = pickle.load(f)
        reloaded_pipe = reloaded_pkg["pipeline"]
        reloaded_pred = reloaded_pipe.predict(self.X_test_df)
        max_diff = float(np.max(np.abs(y_test_pred - reloaded_pred)))
        assert max_diff < 1e-10, f"模型重載推論不一致！最大誤差：{max_diff}"
        print(f"[{self.dataset_name}] Pipeline 重載推論驗證成功 (Max Diff: {max_diff:.2e})！")

        # 4. 匯出逐筆預測 CSV
        pred_df = pd.DataFrame(self.results["test_predictions"])
        csv_path = os.path.join(self.output_dir, f"{prefix}_test_predictions.csv")
        pred_df.to_csv(csv_path, index=False)
        self.results["predictions_csv_path"] = csv_path

        return self.results

    def run_full_pipeline(self):
        """一鍵執行完整 CRISP-DM 回歸工作流"""
        self.step1_load_and_validate()
        self.step2_prepare_data()
        self.step3_train_and_inference()
        self.step4_evaluate()
        self.step5_plot_and_export()
        return self.results


def run_boston_housing_workflow():
    """執行 Boston Housing 第二資料集標準流程"""
    boston_csv = os.path.join(BASE_DIR, "data", "BostonHousing.csv")
    if not os.path.exists(boston_csv):
        print(f"找不到 Boston Housing 資料集：{boston_csv}")
        return None

    print("\n" + "=" * 60)
    print("【Step 7】通用回歸工作流：Boston Housing 資料集分析啟動")
    print("=" * 60)

    wf = RegressionWorkflow(
        dataset_name="Boston_Housing",
        data_source=boston_csv,
        target_column="medv",
        test_size=0.2,
        random_state=0
    )
    res = wf.run_full_pipeline()
    
    print("\n【Boston Housing 最終實測指標】：")
    print(f"  5-Fold CV RMSE: {res['metrics']['cv']['cv_rmse_mean']:.4f} (+/- {res['metrics']['cv']['cv_rmse_std']:.4f})")
    print(f"  Train RMSE    : {res['metrics']['train']['rmse']:.4f} | R²: {res['metrics']['train']['r2']:.4f}")
    print(f"  Test RMSE     : {res['metrics']['test']['rmse']:.4f} | R²: {res['metrics']['test']['r2']:.4f}")
    print(f"  回歸方程式    : {res['model_params']['formula']}")

    return res


if __name__ == "__main__":
    run_boston_housing_workflow()
