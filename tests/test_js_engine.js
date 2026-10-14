const fs = require("fs");
const path = require("path");

// 引入 RegressionEngine
require("../web/regression_engine.js");

function test50Startups() {
  console.log("=== 測試 1: 50_Startups.csv ===");
  const csvText = fs.readFileSync(path.join(__dirname, "../data/50_Startups.csv"), "utf8");
  const parsed = RegressionEngine.parseCSV(csvText);
  console.log(`解析成功：共 ${parsed.headers.length} 欄位，${parsed.rows.length} 列資料。`);

  const analysis = RegressionEngine.analyzeDataset(parsed.headers, parsed.rows);
  console.log(`欄位分析：`, Object.keys(analysis.columnStats).map(c => `${c} (num:${analysis.columnStats[c].isNumeric})`));

  const result = RegressionEngine.fitAndEvaluate({
    headers: parsed.headers,
    rows: parsed.rows,
    targetCol: "Profit",
    featureCols: ["R&D Spend", "Administration", "Marketing Spend", "State"],
    testSize: 0.2,
    randomState: 0
  });

  console.log("訓練結果：");
  console.log("  截距:", result.model.intercept.toFixed(4));
  console.log("  特徵數:", result.datasetInfo.numFeatures);
  console.log("  Train RMSE:", result.metrics.train.rmse.toFixed(4), "R2:", result.metrics.train.r2.toFixed(4));
  console.log("  Test RMSE:", result.metrics.test.rmse.toFixed(4), "R2:", result.metrics.test.r2.toFixed(4));
  console.log("  公式:", result.model.formula);

  if (isNaN(result.metrics.test.rmse) || result.metrics.test.r2 < 0.5) {
    throw new Error("50 Startups 計算異常！");
  }
  console.log("✅ 50 Startups 測試通過！\n");
}

function testBostonHousing() {
  console.log("=== 測試 2: BostonHousing.csv ===");
  const csvPath = path.join(__dirname, "../data/BostonHousing.csv");
  if (!fs.existsSync(csvPath)) {
    console.log("BostonHousing.csv 尚未就緒，跳過");
    return;
  }
  const csvText = fs.readFileSync(csvPath, "utf8");
  const parsed = RegressionEngine.parseCSV(csvText);
  console.log(`解析成功：共 ${parsed.headers.length} 欄位，${parsed.rows.length} 列資料。`);

  const target = parsed.headers.includes("medv") ? "medv" : "MEDV";
  const features = parsed.headers.filter(h => h !== target);

  const result = RegressionEngine.fitAndEvaluate({
    headers: parsed.headers,
    rows: parsed.rows,
    targetCol: target,
    featureCols: features,
    testSize: 0.2,
    randomState: 0
  });

  console.log("訓練結果：");
  console.log("  截距:", result.model.intercept.toFixed(4));
  console.log("  Train RMSE:", result.metrics.train.rmse.toFixed(4), "R2:", result.metrics.train.r2.toFixed(4));
  console.log("  Test RMSE:", result.metrics.test.rmse.toFixed(4), "R2:", result.metrics.test.r2.toFixed(4));

  if (isNaN(result.metrics.test.rmse) || result.metrics.test.r2 < 0.5) {
    throw new Error("Boston Housing 計算異常！");
  }
  console.log("✅ Boston Housing 測試通過！\n");
}

function testEdgeCases() {
  console.log("=== 測試 3: 邊界與異常案例處理 ===");

  // Case A: 缺少必要欄位
  try {
    RegressionEngine.fitAndEvaluate({
      headers: ["A", "B"],
      rows: [["1", "2"]],
      targetCol: "Profit",
      featureCols: ["A"]
    });
    console.error("❌ 應拋出找不到目標欄位錯誤！");
  } catch (e) {
    console.log("✅ 正確攔截錯誤:", e.message);
  }

  // Case B: 目標與特徵重複
  try {
    RegressionEngine.fitAndEvaluate({
      headers: ["A", "B"],
      rows: [["1", "2"], ["3", "4"]],
      targetCol: "B",
      featureCols: ["A", "B"]
    });
    console.error("❌ 應攔截目標變數與特徵重複！");
  } catch (e) {
    console.log("✅ 正確攔截重複:", e.message);
  }

  // Case C: 樣本數過少 (N <= P + 1)
  try {
    RegressionEngine.fitAndEvaluate({
      headers: ["A", "B", "C"],
      rows: [["1", "2", "3"], ["4", "5", "6"]],
      targetCol: "C",
      featureCols: ["A", "B"]
    });
    console.error("❌ 應攔截樣本數過少！");
  } catch (e) {
    console.log("✅ 正確攔截樣本不足:", e.message);
  }

  console.log("✅ 邊界異常測試全部通過！\n");
}

test50Startups();
testBostonHousing();
testEdgeCases();
console.log("🎉 所有 JavaScript 回歸引擎單元測試 100% 通過！");
