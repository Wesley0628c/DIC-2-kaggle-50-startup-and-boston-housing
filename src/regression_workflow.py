"""
模組名稱：regression_workflow.py
用途：可重用的多元線性回歸 (Multiple Linear Regression) 核心工作流程
支援任意數值回歸資料集（如 50 Startups, Boston Housing, 自訂資料集）
流程遵循 CRISP-DM 規範：
  1. Data Acquisition & Validation (資料讀取與品質檢驗)
  2. Data Preparation (自動識別數值/類別特徵、One-Hot 編碼、避免虛擬變數陷阱、資料切分)
  3. Modeling (Ordinary Least Squares 多元線性回歸擬合、統計推論 t/F 統計量計算)
  4. Evaluation (MAE, MSE, RMSE, R², 殘差檢驗, 驗證 RMSE = sqrt(MSE))
  5. Deployment & Artifacts Export (模型 gzip+pickle 序列化、圖表繪製、結構化 JSON 與 CSV 匯出)
"""

import os
import json
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 設定圖表通用字型與清晰度
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class RegressionWorkflow:
    def __init__(self, dataset_name, data_source, target_column, feature_columns=None, 
                 categorical_columns=None, test_size=0.2, random_state=0, output_dir=None):
        """
        初始化回歸分析工作流
        :param dataset_name: 資料集名稱 (如 '50_startups', 'boston_housing')
        :param data_source: 本機路徑 (str) 或 pandas DataFrame
        :param target_column: 目標變數欄位名稱 (y)
        :param feature_columns: 特徵欄位清單 (X)，若為 None 則使用除了 target 以外的所有欄位
        :param categorical_columns: 需進行 One-Hot 編碼之類別欄位清單 (若為 None 則自動偵測 object/category 型別)
        :param test_size: 測試集比例
        :param random_state: 隨機種子
        :param output_dir: 成品輸出目錄
        """
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
        self.processed_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_indices = None
        self.test_indices = None
        self.model = None
        self.encoded_feature_names = []
        self.results = {}

    def step1_load_and_validate(self):
        """Step 1: 讀取資料並進行品質檢核"""
        if isinstance(self.data_source, pd.DataFrame):
            self.raw_df = self.data_source.copy()
        elif isinstance(self.data_source, str):
            if not os.path.exists(self.data_source):
                raise FileNotFoundError(f"找不到資料集檔案：{self.data_source}")
            self.raw_df = pd.read_csv(self.data_source)
        else:
            raise TypeError("data_source 必須為檔案路徑或 pandas.DataFrame")

        if self.target_column not in self.raw_df.columns:
            raise ValueError(f"目標欄位 '{self.target_column}' 不在資料集欄位中！可用欄位：{list(self.raw_df.columns)}")

        # 品質檢查
        total_rows = len(self.raw_df)
        total_cols = len(self.raw_df.columns)
        missing_count = int(self.raw_df.isnull().sum().sum())
        duplicate_count = int(self.raw_df.duplicated().sum())

        quality_report = {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "missing_values": missing_count,
            "duplicates": duplicate_count,
            "columns": list(self.raw_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.raw_df.dtypes.items()}
        }
        self.results["quality_report"] = quality_report
        return self.raw_df

    def step2_prepare_data(self):
        """Step 2: 特徵編碼、規避虛擬變數陷阱、切分資料"""
        df = self.raw_df.copy()

        # 確定特徵欄位
        if self.feature_columns is None:
            features = [c for c in df.columns if c != self.target_column]
        else:
            features = [c for c in self.feature_columns if c in df.columns and c != self.target_column]

        # 偵測或設定類別欄位
        if self.categorical_columns is None:
            cat_cols = [c for c in features if df[c].dtype == 'object' or df[c].dtype.name == 'category']
        else:
            cat_cols = [c for c in self.categorical_columns if c in features]

        # 類別欄位 One-Hot 編碼 (drop_first=True 避免 Dummy Variable Trap)
        if len(cat_cols) > 0:
            df_encoded = pd.get_dummies(df[features], columns=cat_cols, drop_first=True)
        else:
            df_encoded = df[features].copy()

        # 處理可能的缺失值 (以中位數填補)
        for c in df_encoded.columns:
            if df_encoded[c].isnull().sum() > 0:
                df_encoded[c] = df_encoded[c].fillna(df_encoded[c].median())

        self.raw_features = features
        self.cat_cols = cat_cols
        self.encoded_feature_names = list(df_encoded.columns)

        # 取得特徵與目標
        X = df_encoded.values.astype(float)
        y = df[self.target_column].values.astype(float)

        # 切分資料
        indices = np.arange(len(df))
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, indices, test_size=self.test_size, random_state=self.random_state
        )

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.train_indices = idx_train.tolist()
        self.test_indices = idx_test.tolist()

        self.results["split_info"] = {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "num_features": len(self.encoded_feature_names),
            "feature_names": self.encoded_feature_names,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices
        }
        return X_train, X_test, y_train, y_test

    def step3_train_and_inference(self):
        """Step 3: 訓練 OLS 模型並推導統計推論參數"""
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(self.X_train, self.y_train)

        intercept = float(self.model.intercept_)
        coefficients = {name: float(coef) for name, coef in zip(self.encoded_feature_names, self.model.coef_)}

        # 訓練集預測與統計檢定
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        n = len(self.y_train)
        p = len(self.encoded_feature_names)
        df_residuals = n - p - 1

        residuals = self.y_train - y_train_pred
        sse = float(np.sum(residuals ** 2))
        sst = float(np.sum((self.y_train - np.mean(self.y_train)) ** 2))
        train_r2 = float(1.0 - sse / sst) if sst > 1e-12 else 0.0

        # 計算參數標準誤與 t 檢定
        param_stats = {}
        if df_residuals > 0:
            sigma_sq = sse / df_residuals
            X_design = np.column_stack([np.ones(n), self.X_train])
            try:
                cov_matrix = sigma_sq * np.linalg.pinv(X_design.T @ X_design)
                std_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))

                # 截距檢定
                t_b0 = intercept / std_errors[0]
                p_b0 = 2.0 * (1.0 - stats.t.cdf(abs(t_b0), df=df_residuals))
                param_stats["Intercept"] = {
                    "coef": round(intercept, 4),
                    "stderr": round(float(std_errors[0]), 4),
                    "tstat": round(float(t_b0), 4),
                    "pvalue": float(np.format_float_scientific(p_b0, precision=4))
                }

                # 各變數檢定
                for j, feat in enumerate(self.encoded_feature_names, start=1):
                    coef_val = float(self.model.coef_[j - 1])
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

        # 計算整體 F 統計量
        f_stat = 0.0
        f_pvalue = 1.0
        if df_residuals > 0 and p > 0 and (1.0 - train_r2) > 1e-12:
            f_stat = float((train_r2 / p) / ((1.0 - train_r2) / df_residuals))
            f_pvalue = float(1.0 - stats.f.cdf(f_stat, p, df_residuals))

        adj_r2 = float(1.0 - ((1.0 - train_r2) * (n - 1) / df_residuals)) if df_residuals > 0 else train_r2

        # 構建回歸方程式字串
        terms = [f"{intercept:.4f}"]
        for name, coef in coefficients.items():
            sign = "+" if coef >= 0 else "-"
            terms.append(f"{sign} {abs(coef):.4f} * ({name})")
        formula_str = f"{self.target_column} = " + " ".join(terms)

        self.results["model_params"] = {
            "intercept": intercept,
            "coefficients": coefficients,
            "formula": formula_str,
            "parameter_statistics": param_stats,
            "train_statistics": {
                "r2": train_r2,
                "adj_r2": adj_r2,
                "f_stat": round(f_stat, 2),
                "f_pvalue": float(np.format_float_scientific(f_pvalue, precision=4)),
                "df_model": p,
                "df_residuals": df_residuals
            }
        }
        return y_train_pred, y_test_pred

    def step4_evaluate(self):
        """Step 4: 模型評估指標與逐筆預測"""
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # 訓練集指標
        train_mae = float(mean_absolute_error(self.y_train, y_train_pred))
        train_mse = float(mean_squared_error(self.y_train, y_train_pred))
        train_rmse = float(np.sqrt(train_mse))
        train_r2 = float(r2_score(self.y_train, y_train_pred))

        # 測試集指標
        test_mae = float(mean_absolute_error(self.y_test, y_test_pred))
        test_mse = float(mean_squared_error(self.y_test, y_test_pred))
        test_rmse = float(np.sqrt(test_mse))
        test_r2 = float(r2_score(self.y_test, y_test_pred))

        # 逐筆測試預測
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
                "r2": train_r2
            },
            "test": {
                "mae": test_mae,
                "mse": test_mse,
                "rmse": test_rmse,
                "r2": test_r2
            }
        }
        self.results["test_predictions"] = test_predictions
        return self.results["metrics"]

    def step5_plot_and_export(self):
        """Step 5: 繪製散佈圖、殘差圖、序列化儲存並輸出報告"""
        prefix = self.dataset_name.lower().replace(" ", "_")

        # 1. 繪製 Actual vs Predicted 與 Residuals 圖表
        y_test_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_test_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # 子圖 1: 實際 vs 預測
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

        # 子圖 2: 殘差分佈
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

        # 2. 序列化模型 (gzip + pickle)
        model_pkg = {
            "dataset_name": self.dataset_name,
            "target_column": self.target_column,
            "feature_names": self.encoded_feature_names,
            "raw_features": self.raw_features,
            "categorical_columns": self.cat_cols,
            "model": self.model,
            "preprocessor": {
                "encoded_features": self.encoded_feature_names,
                "categorical_columns": self.cat_cols,
                "raw_features": self.raw_features
            },
            "split_info": self.results["split_info"],
            "metrics": self.results["metrics"]
        }
        model_path = os.path.join(self.output_dir, f"{prefix}_linear_regression_model.pkl.gz")
        with gzip.open(model_path, "wb") as f:
            pickle.dump(model_pkg, f)
        self.results["model_path"] = model_path

        # 3. 匯出逐筆預測 CSV
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


if __name__ == "__main__":
    # 測試執行 Boston Housing 工作流
    boston_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "BostonHousing.csv")
    if os.path.exists(boston_csv):
        print("=== 測試執行通用回歸工作流：Boston Housing ===")
        wf = RegressionWorkflow(
            dataset_name="Boston_Housing",
            data_source=boston_csv,
            target_column="medv",
            test_size=0.2,
            random_state=0
        )
        res = wf.run_full_pipeline()
        print(f"Boston Housing 訓練完成！Test RMSE: {res['metrics']['test']['rmse']:.4f}, Test R2: {res['metrics']['test']['r2']:.4f}")
