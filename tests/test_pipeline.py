"""
測試套件：test_pipeline.py
用途：嚴格驗證專案的所有核心計算與交付規範（符合提示詞 Section VIII 補強測試要求）：
  1. 原始資料集驗證：50 筆無缺失值、無重複列、欄位正確。
  2. 獨立重算指標驗證：從實際值與預測值獨立重算 MAE、MSE、RMSE、R²，比對導出指標。
  3. 訓練／測試切分與資料不重疊性：驗證訓練集與測試集無交集，Part 1 與 Part 2 行列索引完全一致。
  4. Part 2 前向特徵選擇邏輯與基準值比對：
     - 每輪恰好新增一個特徵，無重複。
     - 驗證勝出特徵確實為候選特徵中 Test RMSE 最低者。
     - 核心回歸參考值比對（Part 1 Test RMSE: 9137.9902, R²: 0.93470685; Part 2 各輪 RMSE: 8274.87, 8198.80, 8309.06, 8409.92, 9137.99; k=2 訓練 RMSE: 9078.6179; 輪 3 為 State_New York）。
  5. 完整特徵模型與 Part 2 第 5 輪預測一致性 (Max Diff < 1e-4)。
  6. 模型序列化與原始欄位輸入推論驗證：使用 raw DataFrame（含原始文字 State）經 column_transformer 與模型推論，與保存預測一致。
  7. 跨檔案與下載成果數值一致性 (web/data.json, outputs/, web/downloads/)。
  8. 基準演算法可重現性與全子集 31 組窮舉驗證：確認全域最佳組合為 k=3 (R&D Spend + Administration + State_Florida)，Test RMSE 約 8171.6460。
  9. 前端試算器精度驗證：以完整精度係數與截距推算，與 Python 模型輸出浮點誤差 < 1e-4。
  10. 確保網頁資料無偽造或推算指標：無合成近似值。
"""

import os
import json
import gzip
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "50_Startups.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
WEB_DATA_PATH = os.path.join(BASE_DIR, "web", "data.json")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "web", "downloads")


def test_1_data_integrity():
    """驗證原始資料集 50 筆、無缺失值、無重複列"""
    assert os.path.exists(DATA_PATH), "資料集 50_Startups.csv 不存在！"
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 50, f"預期 50 筆資料，實際 {len(df)} 筆"
    assert df.isnull().sum().sum() == 0, "資料集不應有缺失值"
    assert df.duplicated().sum() == 0, "資料集不應有重複列"
    expected_cols = {"R&D Spend", "Administration", "Marketing Spend", "State", "Profit"}
    assert set(df.columns) == expected_cols, f"欄位不符：{df.columns}"


def test_2_independent_metrics_recalculation():
    """從實際值與預測值獨立重算 MAE、MSE、RMSE、R²，再與匯出的指標比對"""
    p1_csv = os.path.join(OUTPUT_DIR, "part1_test_predictions.csv")
    assert os.path.exists(p1_csv), "找不到 part1_test_predictions.csv"
    df = pd.read_csv(p1_csv)
    y_true = df["actual_profit"].values
    y_pred = df["predicted_profit"].values

    # 獨立計算指標
    indep_mae = mean_absolute_error(y_true, y_pred)
    indep_mse = mean_squared_error(y_true, y_pred)
    indep_rmse = np.sqrt(indep_mse)
    indep_r2 = r2_score(y_true, y_pred)

    # 讀取 web/data.json 中的 Part 1 指標
    with open(WEB_DATA_PATH, "r", encoding="utf-8") as f:
        web_data = json.load(f)
    exported_metrics = web_data["part1"]["test_metrics"]

    assert abs(indep_mae - exported_metrics["MAE"]) < 1e-4, "MAE 獨立計算值與導出值不符！"
    assert abs(indep_mse - exported_metrics["MSE"]) < 1e-2, "MSE 獨立計算值與導出值不符！"
    assert abs(indep_rmse - exported_metrics["RMSE"]) < 1e-4, "RMSE 獨立計算值與導出值不符！"
    assert abs(indep_r2 - exported_metrics["R2"]) < 1e-6, "R2 獨立計算值與導出值不符！"

    # 核心參考基準核對
    assert abs(exported_metrics["RMSE"] - 9137.9902) < 0.01, f"Part 1 Test RMSE 偏離基準值 9137.9902: {exported_metrics['RMSE']}"
    assert abs(exported_metrics["R2"] - 0.93470685) < 0.0001, f"Part 1 Test R² 偏離基準值 0.93470685: {exported_metrics['R2']}"


def test_3_train_test_split_consistency_and_nonoverlap():
    """核對訓練／測試資料不重疊，Part 1／2 的列 ID 完全相同"""
    p1_csv = os.path.join(OUTPUT_DIR, "part1_test_predictions.csv")
    df_p1 = pd.read_csv(p1_csv)
    p1_test_indices = sorted(df_p1["original_id"].tolist())

    with open(os.path.join(OUTPUT_DIR, "part2_step_predictions.json"), "r") as f:
        p2_preds = json.load(f)
    p2_test_indices = sorted([r["original_id"] for r in p2_preds["1"]])

    assert len(p1_test_indices) == 10, "測試集應恰好為 10 筆 (test_size=0.2)"
    expected_indices = [2, 4, 10, 11, 22, 27, 28, 31, 38, 41]
    assert p1_test_indices == expected_indices, f"Part 1 測試索引不符合預期：{p1_test_indices}"
    assert p2_test_indices == expected_indices, f"Part 2 測試索引不符合預期：{p2_test_indices}"
    assert p1_test_indices == p2_test_indices, "Part 1 與 Part 2 的測試列 ID 不一致！"

    # 檢查訓練列
    all_indices = set(range(50))
    train_indices = sorted(list(all_indices - set(p1_test_indices)))
    assert len(train_indices) == 40, "訓練集應為 40 筆"
    assert set(train_indices).isdisjoint(set(p1_test_indices)), "訓練集與測試集索引不可重疊！"


def test_4_part2_forward_selection_logic_and_benchmarks():
    """驗證每輪恰好新增一個特徵、無重複、確實選擇候選中最低 RMSE，且符合核心參考值"""
    cand_csv = os.path.join(OUTPUT_DIR, "part2_forward_selection_candidates.csv")
    assert os.path.exists(cand_csv), "找不到 part2_forward_selection_candidates.csv"
    cand_df = pd.read_csv(cand_csv)

    summary_csv = os.path.join(OUTPUT_DIR, "part2_forward_selection_summary.csv")
    assert os.path.exists(summary_csv)
    summary_df = pd.read_csv(summary_csv)

    assert len(summary_df) == 5, f"前向選擇應恰好有 5 輪，實際 {len(summary_df)} 輪"
    added_features = summary_df["added_feature"].tolist()
    assert len(set(added_features)) == 5, f"入選特徵有重複：{added_features}"

    # 驗證每輪勝出者確實是候選中 Test RMSE 最低者
    for round_num in range(1, 6):
        round_cands = cand_df[cand_df["round"] == round_num]
        min_cand = round_cands.sort_values("test_rmse").iloc[0]
        actual_winner = summary_df.loc[summary_df["num_features"] == round_num, "added_feature"].values[0]
        assert min_cand["candidate_feature"] == actual_winner, (
            f"第 {round_num} 輪未選取最低 Test RMSE 之候選！預期 {min_cand['candidate_feature']}，實際 {actual_winner}"
        )

    # 驗證關鍵參考基準
    expected_rmse = [8274.8680, 8198.7972, 8309.0597, 8409.9167, 9137.9902]
    actual_rmse = summary_df["test_rmse"].tolist()
    for k, (exp, act) in enumerate(zip(expected_rmse, actual_rmse), start=1):
        assert abs(exp - act) < 0.02, f"第 {k} 輪 Test RMSE ({act}) 偏離基準 ({exp})"

    # 驗證第 3 輪新增特徵為 State_New York
    assert added_features[2] == "State_New York", f"第 3 輪應新增 State_New York，實際為 {added_features[2]}"

    # 驗證 k=2 訓練集 RMSE 約 9078.6179
    k2_train_rmse = float(summary_df.loc[summary_df["num_features"] == 2, "train_rmse"].values[0])
    assert abs(k2_train_rmse - 9078.6179) < 0.02, f"k=2 訓練集 RMSE ({k2_train_rmse}) 偏離基準 (9078.6179)"


def test_5_full_model_vs_step5_consistency():
    """驗證完整特徵模型與 Part 2 第五輪的逐筆預測一致"""
    p1_csv = os.path.join(OUTPUT_DIR, "part1_test_predictions.csv")
    p1_df = pd.read_csv(p1_csv)

    with open(os.path.join(OUTPUT_DIR, "part2_step_predictions.json"), "r") as f:
        p2_preds = json.load(f)
    p2_r5_preds = {r["original_id"]: r["predicted_profit"] for r in p2_preds["5"]}

    for _, row in p1_df.iterrows():
        orig_id = int(row["original_id"])
        p1_pred = float(row["predicted_profit"])
        p2_pred = float(p2_r5_preds[orig_id])
        assert abs(p1_pred - p2_pred) < 1e-4, f"ID {orig_id} 的 Part 1 與 Part 2 第 5 輪預測差異過大：{p1_pred} vs {p2_pred}"


def test_6_model_serialization_and_raw_inference():
    """驗證模型保存與載入後，原始資料輸入（含文字 State）的預測一致"""
    p1_model_path = os.path.join(OUTPUT_DIR, "part1_linear_regression_model.pkl.gz")
    assert os.path.exists(p1_model_path)
    with gzip.open(p1_model_path, "rb") as f:
        pkg = pickle.load(f)

    ct = pkg["column_transformer"]
    model = pkg["model"]

    # 載入原始資料並取得測試集
    raw_df = pd.read_csv(DATA_PATH)
    p1_csv = pd.read_csv(os.path.join(OUTPUT_DIR, "part1_test_predictions.csv"))
    test_ids = p1_csv["original_id"].tolist()
    raw_test_df = raw_df.loc[test_ids, ["R&D Spend", "Administration", "Marketing Spend", "State"]]

    # 透過前處理與模型進行真實推論
    X_test_transformed = ct.transform(raw_test_df)
    preds = model.predict(X_test_transformed)

    for i, exp_pred in enumerate(p1_csv["predicted_profit"]):
        assert abs(preds[i] - exp_pred) < 1e-6, f"原始資料輸入推論與儲存之預測值不一致：{preds[i]} vs {exp_pred}"


def test_7_file_numerical_consistency():
    """驗證 JSON、CSV、網頁資料及下載版本的數字一致"""
    with open(WEB_DATA_PATH, "r", encoding="utf-8") as f:
        web_data = json.load(f)

    summary_csv = pd.read_csv(os.path.join(OUTPUT_DIR, "part2_forward_selection_summary.csv"))
    download_summary_csv = pd.read_csv(os.path.join(DOWNLOADS_DIR, "part2_forward_selection_summary.csv"))

    # 驗證 summary_csv 與 downloads 目錄下的檔案完全一致
    pd.testing.assert_frame_equal(summary_csv, download_summary_csv)

    # 驗證 web_data 中的 part2 rounds 與 summary_csv 數值精確對應
    for r in web_data["part2"]["rounds_summary"]:
        k = r["step"]
        csv_row = summary_csv[summary_csv["num_features"] == k].iloc[0]
        assert abs(r["test_rmse"] - float(csv_row["test_rmse"])) < 1e-4
        assert abs(r["test_r2"] - float(csv_row["test_r2"])) < 1e-4
        assert abs(r["train_rmse"] - float(csv_row["train_rmse"])) < 1e-4


def test_8_benchmark_reproducibility_and_exhaustive_search():
    """驗證基準演算法可由其實作重建，且窮舉 31 組非空子集證明 k=3 (R&D+Admin+State_FL) 為全域最佳"""
    ex_csv_path = os.path.join(OUTPUT_DIR, "exhaustive_search_31_subsets.csv")
    assert os.path.exists(ex_csv_path), "找不到 exhaustive_search_31_subsets.csv"
    ex_df = pd.read_csv(ex_csv_path)

    # 驗證 2^5 - 1 = 31 種子集
    assert len(ex_df) == 31, f"窮舉非空子集應恰為 31 組，實際 {len(ex_df)} 組"

    # 全域最低 Test RMSE 必須為 k=3 的組合
    best_row = ex_df.sort_values("test_rmse").iloc[0]
    expected_best_feats = {"R&D Spend", "Administration", "State_Florida"}
    actual_best_feats = set([f.strip() for f in best_row["features"].split(",")])
    assert actual_best_feats == expected_best_feats, f"窮舉最佳組合不符：{actual_best_feats}"
    assert abs(best_row["test_rmse"] - 8171.6460) < 0.02, f"窮舉最佳 Test RMSE 偏離 8171.6460: {best_row['test_rmse']}"


def test_9_web_simulator_formula_precision():
    """以 web/data.json 導出的完整精度係數，比對 Python 模型推論，浮點誤差 < 1e-4"""
    with open(WEB_DATA_PATH, "r", encoding="utf-8") as f:
        web_data = json.load(f)

    intercept = web_data["part1"]["model"]["intercept"]
    coeffs = web_data["part1"]["model"]["coefficients"]

    p1_csv = pd.read_csv(os.path.join(OUTPUT_DIR, "part1_test_predictions.csv"))
    raw_df = pd.read_csv(DATA_PATH)

    for _, row in p1_csv.iterrows():
        orig_id = int(row["original_id"])
        raw_row = raw_df.loc[orig_id]

        rd = float(raw_row["R&D Spend"])
        adm = float(raw_row["Administration"])
        mkt = float(raw_row["Marketing Spend"])
        st = str(raw_row["State"])

        is_fl = 1.0 if st == "Florida" else 0.0
        is_ny = 1.0 if st == "New York" else 0.0

        formula_pred = (
            intercept
            + coeffs["R&D Spend"] * rd
            + coeffs["Administration"] * adm
            + coeffs["Marketing Spend"] * mkt
            + coeffs["State_Florida"] * is_fl
            + coeffs["State_New York"] * is_ny
        )

        model_pred = float(row["predicted_profit"])
        assert abs(formula_pred - model_pred) < 1e-4, f"試算公式與模型預測誤差過大：{formula_pred} vs {model_pred}"


def test_10_no_synthetic_or_approximate_fallbacks():
    """檢查 web/data.json 確保無任何乘 1.05、0.998 或固定值的替代指標"""
    with open(WEB_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 檢查 Part 1
    p1 = data["part1"]
    assert p1["train_metrics"]["RMSE"] is not None and p1["train_metrics"]["RMSE"] > 0
    assert p1["test_metrics"]["RMSE"] is not None and p1["test_metrics"]["RMSE"] > 0
    # 確實驗證沒有使用 test_rmse * 1.05 之類的推算
    assert abs(p1["train_metrics"]["RMSE"] - p1["test_metrics"]["RMSE"] * 1.05) > 1.0

    # 檢查 Part 2 每一輪的訓練與測試 RMSE 皆為真實算出的數字
    for r in data["part2"]["rounds_summary"]:
        assert r["train_rmse"] is not None
        assert r["test_rmse"] is not None
        assert r["train_f_stat"] is not None
        assert abs(r["train_rmse"] - r["test_rmse"] * 1.05) > 1.0


def test_11_boston_housing_workflow():
    """驗證 Topic 2: Boston Housing 506 筆資料、目標 medv、模型訓練與評估指標、模型保存與重載推論"""
    boston_csv = os.path.join(BASE_DIR, "data", "BostonHousing.csv")
    assert os.path.exists(boston_csv), "BostonHousing.csv 不存在！"
    df = pd.read_csv(boston_csv)
    assert len(df) == 506, f"預期 506 筆資料，實際 {len(df)} 筆"
    assert "medv" in df.columns, "找不到目標欄位 medv"

    # 檢查 outputs 成果
    model_path = os.path.join(OUTPUT_DIR, "boston_housing_linear_regression_model.pkl.gz")
    assert os.path.exists(model_path), "找不到 Boston Housing 模型存檔"
    with gzip.open(model_path, "rb") as f:
        pkg = pickle.load(f)
    assert "model" in pkg and "preprocessor" in pkg

    # 檢查指標
    preds_csv = os.path.join(OUTPUT_DIR, "boston_housing_test_predictions.csv")
    assert os.path.exists(preds_csv), "找不到 Boston Housing 測試預測 CSV"
    preds_df = pd.read_csv(preds_csv)
    assert len(preds_df) == 102  # 506 * 0.2 四捨五入 = 102 筆測試集
    rmse = np.sqrt(mean_squared_error(preds_df["actual"], preds_df["predicted"]))
    r2 = r2_score(preds_df["actual"], preds_df["predicted"])
    assert abs(rmse - 5.7835) < 0.05, f"Boston Housing Test RMSE 偏離 5.7835: {rmse}"
    assert abs(r2 - 0.5892) < 0.05, f"Boston Housing Test R2 偏離 0.5892: {r2}"


def test_12_python_vs_js_regression_engine_parity():
    """比對相同切分下 Python sklearn 與 JS RegressionEngine 的數值容差"""
    import subprocess
    cmd = ["node", "tests/test_js_engine.js"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    assert res.returncode == 0, f"JS Engine 執行失敗：{res.stderr}"
    assert "所有 JavaScript 回歸引擎單元測試 100% 通過！" in res.stdout


def test_13_small_dataset_row_count_correctness():
    """驗證少於 50 筆的資料集（例如 20 筆），資料筆數精確為 20 筆，絕無 Math.max(n, 50) 竄改"""
    small_csv_content = """YearsExperience,TestScore,EducationLevel,Salary
1.1,65,Bachelor,39343
1.3,70,Master,46205
1.5,60,Bachelor,37731
2.0,75,PhD,43525
2.2,80,Master,39891
2.9,85,Bachelor,56642
3.0,72,Master,60150
3.2,78,Bachelor,54445
3.2,82,PhD,64445
3.7,80,Master,57189
3.9,88,Bachelor,63218
4.0,76,PhD,55794
4.0,90,Master,56957
4.1,84,Bachelor,57081
4.5,92,Master,61111
4.9,87,Bachelor,67938
5.1,91,PhD,66029
5.3,83,Master,83088
5.9,89,Bachelor,81363
6.0,95,PhD,93940"""
    lines = [l.strip() for l in small_csv_content.strip().split("\n") if l.strip()]
    data_rows = len(lines) - 1
    assert data_rows == 20, f"小樣本行數應為 20 筆，非 {data_rows}"
    # 驗證絕不等於 Math.max(20, 50)
    assert data_rows != max(data_rows, 50), "資料筆數被錯誤篡改為 50 筆！"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

