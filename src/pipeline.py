"""
模組名稱：pipeline.py
說明：一鍵執行 50 Startups 多元線性回歸專案之端到端分析工作流程：
  1. 下載資料與品質檢查 (download_data)
  2. Part 1 CRISP-DM 多元線性回歸 (part1_crispdm)
  3. Part 2 前向特徵選擇演算法 (part2_forward_selection, 5-Fold CV 規範)
  4. 五大多元回歸演算法基準比對與 31 組全子集窮舉 (benchmark_algorithms)
  5. Optuna 超參數自動調校 (optuna_tuner)
  6. Topic 2: Boston Housing 通用回歸工作流 (regression_workflow)
  7. 整合並匯出網頁 JSON 數據 (export_web_data)
  8. 同步產出物至 web/downloads/ 與 docs/
"""

import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_data import download_dataset
from part1_crispdm import run_part1
from part2_forward_selection import run_part2
from benchmark_algorithms import execute_benchmarks
from optuna_tuner import run_optuna_tuning
from regression_workflow import run_boston_housing_workflow
from export_web_data import generate_web_json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
WEB_DIR = os.path.join(BASE_DIR, "web")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DOWNLOADS_DIR = os.path.join(WEB_DIR, "downloads")


def sync_artifacts():
    """
    確保 outputs/、web/downloads/ 與 docs/ 的資料與圖表完全同步，
    杜絕網頁更新而離線資料或下載檔仍是舊版本的情況。
    """
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(os.path.join(DOCS_DIR, "downloads"), exist_ok=True)
    os.makedirs(os.path.join(DOCS_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(WEB_DIR, "images"), exist_ok=True)

    # 1. 複製圖表到 web/images 與 docs/images
    img_list = [
        "part1_synthetic_demo_scatter.png",
        "part1_scatter_plots.png",
        "real_data_scatters.png",
        "part1_evaluation_plots.png",
        "actual_vs_predicted.png",
        "residuals_plot.png",
        "part2_features_trend.png",
        "learning_curve.png",
        "optuna_optimization_history.png",
        "boston_housing_evaluation_plots.png"
    ]
    for img_name in img_list:
        src_path = os.path.join(OUTPUT_DIR, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(WEB_DIR, "images", img_name))
            shutil.copy(src_path, os.path.join(DOCS_DIR, "images", img_name))

    # 2. 複製下載檔案到 web/downloads 與 docs/downloads
    download_files = [
        "part1_linear_regression_model.pkl.gz",
        "part2_forward_selection_models.pkl.gz",
        "optuna_tuned_model.pkl.gz",
        "part1_test_predictions.csv",
        "part2_forward_selection_summary.csv",
        "part2_forward_selection_candidates.csv",
        "part2_historical_test_sfs_summary.csv",
        "exhaustive_search_31_subsets.csv",
        "optuna_trials.csv",
        "boston_housing_linear_regression_model.pkl.gz",
        "boston_housing_test_predictions.csv"
    ]
    for fn in download_files:
        src_path = os.path.join(OUTPUT_DIR, fn)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(DOWNLOADS_DIR, fn))
            shutil.copy(src_path, os.path.join(DOCS_DIR, "downloads", fn))

    # 複製 Jupyter Notebook
    nb_name = "50_Startup_Multiple_Linear_Regression_CRISP_DM.ipynb"
    nb_src = os.path.join(BASE_DIR, nb_name)
    if os.path.exists(nb_src):
        shutil.copy(nb_src, os.path.join(DOWNLOADS_DIR, nb_name))
        shutil.copy(nb_src, os.path.join(DOCS_DIR, "downloads", nb_name))

    # 複製 results.md 與 README.md
    for doc in ["results.md", "README.md"]:
        doc_src = os.path.join(BASE_DIR, doc)
        if os.path.exists(doc_src):
            shutil.copy(doc_src, os.path.join(DOWNLOADS_DIR, doc))
            shutil.copy(doc_src, os.path.join(DOCS_DIR, "downloads", doc))

    # 複製 data.json
    data_json_src = os.path.join(WEB_DIR, "data.json")
    if os.path.exists(data_json_src):
        shutil.copy(data_json_src, os.path.join(DOWNLOADS_DIR, "data.json"))
        shutil.copy(data_json_src, os.path.join(DOCS_DIR, "downloads", "data.json"))

    # 3. 同步前端站點 HTML/CSS/JS 到 docs/ 以及專案根目錄
    for web_file in ["index.html", "style.css", "app.js", "regression_engine.js", "chart.umd.min.js", "data.json", "default_data.js"]:
        src_path = os.path.join(WEB_DIR, web_file)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(DOCS_DIR, web_file))
            shutil.copy(src_path, os.path.join(BASE_DIR, web_file))

    # 4. 確保 .nojekyll 存在於根目錄、docs/ 與 web/
    for nojekyll_dir in [BASE_DIR, DOCS_DIR, WEB_DIR]:
        nojekyll_file = os.path.join(nojekyll_dir, ".nojekyll")
        with open(nojekyll_file, "a") as f:
            pass

    print("[同步完成] 所有產出物、模型、圖表與數據已 100% 同步至 web/、web/downloads/、docs/ 與專案根目錄 (含 .nojekyll)")


def main():
    print("=============================================================")
    print(" 🚀 50 Startups 多元線性回歸 (MLR) 完整工作流 (Workflow) 啟動")
    print("=============================================================")
    
    # 步驟 1: 下載與品質驗證
    print("\n[Stage 1/7] 下載資料集並執行欄位與遺漏值檢查...")
    df = download_dataset()

    # 步驟 2: Part 1 CRISP-DM
    print("\n[Stage 2/7] 執行 Part 1: CRISP-DM 多元線性回歸...")
    p1_res = run_part1()

    # 步驟 3: Part 2 Forward Selection (5-Fold CV)
    print("\n[Stage 3/7] 執行 Part 2: Forward Selection 前向特徵選擇 (5-Fold CV)...")
    p2_res = run_part2()

    # 步驟 4: 五大演算法基準評估與窮舉
    print("\n[Stage 4/7] 執行五大特徵選擇演算法真實基準評估 (5-Fold CV) 與 31 組全子集窮舉...")
    benchmarks_res = execute_benchmarks()

    # 步驟 5: Optuna 超參數自動調校
    print("\n[Stage 5/7] 執行 Step 4: Optuna 超參數自動調校 (Ridge)...")
    optuna_res = run_optuna_tuning()

    # 步驟 6: 執行 Topic 2: Boston Housing 通用回歸工作流
    print("\n[Stage 6/7] 執行 Topic 2: Boston Housing 多元回歸實證 (通用 Workflow)...")
    boston_res = run_boston_housing_workflow()

    # 步驟 7: 匯出網頁數據
    print("\n[Stage 7/7] 匯出結構化網頁 data.json 與報告...")
    web_data = generate_web_json()

    # 同步檔案
    sync_artifacts()

    print("\n=============================================================")
    print(" ✅ 完整工作流執行完畢！所有輸出與圖表均已保存至 outputs/、web/ 與 docs/")
    print("=============================================================")


if __name__ == "__main__":
    main()
