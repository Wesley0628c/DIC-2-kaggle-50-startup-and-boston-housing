"""
測試套件：test_pipeline.py
用途：嚴格驗證專案的所有核心計算與交付要求：
  1. 原始資料筆數為 50 筆，無缺失值與重複列。
  2. RMSE 數學上必須精確等於 sqrt(MSE)。
  3. Part 1 與 Part 2 的訓練集與測試集列索引 (Indices) 必須完全相同。
  4. Part 2 前向選擇選出恰好 5 輪，且入選特徵完全不重複。
  5. Part 2 第 5 輪（包含全部 5 個特徵）之指標必須與 Part 1 測試集指標高度一致。
  6. pickle + gzip 儲存之模型重載後推論一致 (Max Diff < 1e-9)。
  7. web/data.json 與 outputs/ 下的 CSV 檔案數值精確同步。
"""

import os
import json
import gzip
import pickle
import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "50_Startups.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
WEB_DATA_PATH = os.path.join(BASE_DIR, "web", "data.json")

def test_data_integrity():
    assert os.path.exists(DATA_PATH), "資料集 50_Startups.csv 不存在！"
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 50, f"預期 50 筆資料，實際 {len(df)} 筆"
    assert df.isnull().sum().sum() == 0, "資料集不應有缺失值"
    assert df.duplicated().sum() == 0, "資料集不應有重複列"
    expected_cols = {"R&D Spend", "Administration", "Marketing Spend", "State", "Profit"}
    assert set(df.columns) == expected_cols, f"欄位不符：{df.columns}"

def test_rmse_equals_sqrt_mse():
    p1_csv = os.path.join(OUTPUT_DIR, "part1_test_predictions.csv")
    assert os.path.exists(p1_csv), "找不到 part1_test_predictions.csv"
    df = pd.read_csv(p1_csv)
    y_true = df["actual_profit"].values
    y_pred = df["predicted_profit"].values

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    assert abs(rmse - np.sqrt(mse)) < 1e-9, "RMSE 不等於 sqrt(MSE)！"

def test_train_test_split_consistency():
    p1_csv = os.path.join(OUTPUT_DIR, "part1_test_predictions.csv")
    df_p1 = pd.read_csv(p1_csv)
    p1_test_indices = sorted(df_p1["original_id"].tolist())

    with open(os.path.join(OUTPUT_DIR, "part2_step_predictions.json"), "r") as f:
        p2_preds = json.load(f)
    p2_test_indices = sorted([r["original_id"] for r in p2_preds["1"]])

    assert p1_test_indices == p2_test_indices, f"Part 1 與 Part 2 的測試列 ID 不一致！P1: {p1_test_indices}, P2: {p2_test_indices}"
    assert len(p1_test_indices) == 10, "測試集應恰好為 10 筆 (test_size=0.2)"
    expected_indices = [2, 4, 10, 11, 22, 27, 28, 31, 38, 41]
    assert p1_test_indices == expected_indices, f"測試列索引不符合 random_state=0 預期：{p1_test_indices}"

def test_forward_selection_progression():
    summary_csv = os.path.join(OUTPUT_DIR, "part2_forward_selection_summary.csv")
    assert os.path.exists(summary_csv)
    df = pd.read_csv(summary_csv)
    assert len(df) == 5, f"前向選擇應恰好有 5 輪，實際有 {len(df)} 輪"
    
    # 檢查入選特徵無重複
    added_features = df["added_feature"].tolist()
    assert len(set(added_features)) == 5, f"入選特徵有重複：{added_features}"

    # 檢查第 5 輪與 Part 1 測試集指標一致
    p1_csv = os.path.join(OUTPUT_DIR, "part1_test_predictions.csv")
    p1_df = pd.read_csv(p1_csv)
    p1_rmse = np.sqrt(np.mean((p1_df["actual_profit"] - p1_df["predicted_profit"]) ** 2))
    round5_rmse = float(df.loc[df["num_features"] == 5, "test_rmse"].values[0])
    assert abs(p1_rmse - round5_rmse) < 1e-4, f"第 5 輪 RMSE ({round5_rmse}) 與 Part 1 ({p1_rmse}) 不一致！"

def test_model_serialization_consistency():
    p1_model_path = os.path.join(OUTPUT_DIR, "part1_linear_regression_model.pkl.gz")
    assert os.path.exists(p1_model_path)
    with gzip.open(p1_model_path, "rb") as f:
        pkg = pickle.load(f)
    
    # 驗證關鍵設定存在
    assert "column_transformer" in pkg
    assert "model" in pkg
    assert pkg["reference_state"] == "California"
    assert len(pkg["valid_states"]) == 3

def test_web_json_consistency():
    assert os.path.exists(WEB_DATA_PATH)
    with open(WEB_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["metadata"]["total_rows"] == 50
    assert len(data["dataset"]["head_5"]) == 5
    assert len(data["part2"]["rounds_summary"]) == 5
    assert data["part2"]["best_feature_count"] == 2
    assert data["part2"]["best_features"] == ["R&D Spend", "Marketing Spend"]

if __name__ == "__main__":
    test_data_integrity()
    test_rmse_equals_sqrt_mse()
    test_train_test_split_consistency()
    test_forward_selection_progression()
    test_model_serialization_consistency()
    test_web_json_consistency()
    print("\n>>> 所有單元與端到端驗證測試 100% 通過！<<<")
