/**
 * Aether Quant Lab - Client-side Pure JavaScript Linear Regression Engine
 * 專為 GitHub Pages 靜態環境設計之瀏覽器端真實多元線性回歸引擎
 * 特色：
 *   1. RFC 4180 標準 CSV 解析器 (支援引號、跨行、逗號、UTF-8 BOM)
 *   2. 自動型別偵測 (數值 vs 類別) 與缺失值統計
 *   3. One-Hot Encoding (自動移除基準類別規避 Dummy Variable Trap)
 *   4. 可重現隨機種子之 train_test_split
 *   5. 數值穩定 OLS 求解器 (全主元高斯消去法 + 奇異矩陣正則防護)
 *   6. 完整指標計算 (MAE, MSE, RMSE, R², Adjusted R²)
 */

(function (window) {
  "use strict";

  const RegressionEngine = {};

  // 1. RFC 4180 CSV 解析器
  RegressionEngine.parseCSV = function (text) {
    if (!text || typeof text !== "string") {
      throw new Error("輸入的 CSV 內容為空或非字串！");
    }

    // 移除 UTF-8 BOM
    if (text.charCodeAt(0) === 0xFEFF) {
      text = text.slice(1);
    }

    const rows = [];
    let currentRow = [];
    let currentCell = "";
    let insideQuote = false;
    let i = 0;
    const len = text.length;

    while (i < len) {
      const c = text[i];
      const nextC = i + 1 < len ? text[i + 1] : "";

      if (insideQuote) {
        if (c === '"') {
          if (nextC === '"') {
            currentCell += '"';
            i += 2;
            continue;
          } else {
            insideQuote = false;
            i++;
            continue;
          }
        } else {
          currentCell += c;
          i++;
          continue;
        }
      } else {
        if (c === '"') {
          insideQuote = true;
          i++;
          continue;
        } else if (c === ',') {
          currentRow.push(currentCell.trim());
          currentCell = "";
          i++;
          continue;
        } else if (c === '\r') {
          if (nextC === '\n') {
            i++;
          }
          currentRow.push(currentCell.trim());
          if (currentRow.some(c => c.length > 0)) {
            rows.push(currentRow);
          }
          currentRow = [];
          currentCell = "";
          i++;
          continue;
        } else if (c === '\n') {
          currentRow.push(currentCell.trim());
          if (currentRow.some(c => c.length > 0)) {
            rows.push(currentRow);
          }
          currentRow = [];
          currentCell = "";
          i++;
          continue;
        } else {
          currentCell += c;
          i++;
          continue;
        }
      }
    }

    if (currentCell.length > 0 || currentRow.length > 0) {
      currentRow.push(currentCell.trim());
      if (currentRow.some(c => c.length > 0)) {
        rows.push(currentRow);
      }
    }

    if (rows.length < 2) {
      throw new Error(`CSV 資料筆數過少 (僅 ${rows.length} 列)，至少需要標題列與 1 列資料！`);
    }

    const headers = rows[0].map((h, idx) => h.length > 0 ? h : `Column_${idx + 1}`);
    const dataRows = rows.slice(1);

    return { headers, rows: dataRows };
  };

  // 2. 資料集品質分析與型別推斷
  RegressionEngine.analyzeDataset = function (headers, rows) {
    const totalRows = rows.length;
    const totalCols = headers.length;
    const columnStats = {};

    headers.forEach((col, cIdx) => {
      let missingCount = 0;
      let numericCount = 0;
      let values = [];

      rows.forEach(r => {
        const val = r[cIdx] !== undefined ? r[cIdx] : "";
        if (val === "" || val === null || val === undefined || val.toLowerCase() === "nan" || val.toLowerCase() === "null") {
          missingCount++;
        } else {
          const num = Number(val);
          if (!isNaN(num) && isFinite(num)) {
            numericCount++;
          }
          values.push(val);
        }
      });

      const validCount = values.length;
      const isNumeric = validCount > 0 && numericCount === validCount;

      columnStats[col] = {
        index: cIdx,
        isNumeric,
        missingCount,
        validCount,
        sampleValues: values.slice(0, 5)
      };
    });

    return { totalRows, totalCols, columnStats };
  };

  // 3. 偽隨機生成器 (Mulberry32，鎖定 random_state)
  RegressionEngine.mulberry32 = function (seed) {
    return function () {
      let t = (seed += 0x6D2B79F5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  // 4. 高斯消去法求解線性聯立方程組 (A * x = b)
  RegressionEngine.solveLinearSystem = function (A, b) {
    const n = b.length;
    // 建立增廣矩陣 [A | b]
    const M = [];
    for (let i = 0; i < n; i++) {
      M[i] = new Float64Array(n + 1);
      for (let j = 0; j < n; j++) {
        M[i][j] = A[i][j];
      }
      M[i][n] = b[i];
    }

    // 主元消去
    for (let p = 0; p < n; p++) {
      let maxRow = p;
      let maxVal = Math.abs(M[p][p]);
      for (let r = p + 1; r < n; r++) {
        const val = Math.abs(M[r][p]);
        if (val > maxVal) {
          maxVal = val;
          maxRow = r;
        }
      }

      if (maxVal < 1e-12) {
        // 近似奇異主元，加入極小數值防護
        M[p][p] += 1e-7;
      }

      if (maxRow !== p) {
        const temp = M[p];
        M[p] = M[maxRow];
        M[maxRow] = temp;
      }

      const pivot = M[p][p];
      for (let j = p; j <= n; j++) {
        M[p][j] /= pivot;
      }

      for (let r = 0; r < n; r++) {
        if (r !== p) {
          const factor = M[r][p];
          if (Math.abs(factor) > 1e-15) {
            for (let j = p; j <= n; j++) {
              M[r][j] -= factor * M[p][j];
            }
          }
        }
      }
    }

    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = M[i][n];
    }
    return Array.from(x);
  };

  // 5. 核心訓練、預測與評估
  RegressionEngine.fitAndEvaluate = function (config) {
    const {
      headers,
      rows,
      targetCol,
      featureCols,
      testSize = 0.2,
      randomState = 0
    } = config;

    if (!targetCol || !featureCols || featureCols.length === 0) {
      throw new Error("必須指定目標欄位 (y) 與至少一個特徵欄位 (X)！");
    }

    if (featureCols.includes(targetCol)) {
      throw new Error("目標欄位 (y) 不得同時被選為特徵變數 (X)！");
    }

    const targetIdx = headers.indexOf(targetCol);
    if (targetIdx === -1) {
      throw new Error(`找不到目標欄位：${targetCol}`);
    }

    // 1. 篩選出目標值合法的樣本列
    const validRows = [];
    const originalRowIds = [];
    let droppedCount = 0;

    rows.forEach((r, idx) => {
      const yRaw = r[targetIdx];
      const yNum = Number(yRaw);
      if (yRaw !== "" && yRaw !== null && !isNaN(yNum) && isFinite(yNum)) {
        validRows.push(r);
        originalRowIds.push(idx);
      } else {
        droppedCount++;
      }
    });

    if (validRows.length < 5) {
      throw new Error(`有效數值目標樣本數僅 ${validRows.length} 筆，樣本不足無法訓練！`);
    }

    // 2. 特徵預處理與 One-Hot Encoding
    const featureIndices = featureCols.map(c => headers.indexOf(c));
    const isCatCol = [];
    const catUniqueValues = [];
    const numMedians = [];

    featureIndices.forEach((fIdx, colSeq) => {
      const vals = validRows.map(r => r[fIdx]);
      const isNum = vals.every(v => v === "" || (!isNaN(Number(v)) && isFinite(Number(v))));
      isCatCol.push(!isNum);

      if (!isNum) {
        // 類別型：收集不重複值並排序
        const uniques = Array.from(new Set(vals.filter(v => v !== "" && v !== null))).sort();
        catUniqueValues.push(uniques);
        numMedians.push(0);
      } else {
        // 數值型：計算中位數填補
        catUniqueValues.push([]);
        const nums = vals.map(v => Number(v)).filter(v => !isNaN(v) && isFinite(v)).sort((a, b) => a - b);
        const med = nums.length > 0 ? (nums[Math.floor(nums.length / 2)]) : 0;
        numMedians.push(med);
      }
    });

    // 建立編碼後特徵名稱清單
    const encodedNames = [];
    featureIndices.forEach((fIdx, colSeq) => {
      const colName = headers[fIdx];
      if (isCatCol[colSeq]) {
        // One-Hot: drop_first 避免 Dummy Variable Trap
        const cats = catUniqueValues[colSeq];
        for (let k = 1; k < cats.length; k++) {
          encodedNames.push(`${colName}_${cats[k]}`);
        }
      } else {
        encodedNames.push(colName);
      }
    });

    if (encodedNames.length === 0) {
      throw new Error("經前處理編碼後未產生任何有效特徵！");
    }

    // 轉換每筆資料為數值特徵向量
    const X_all = [];
    const y_all = [];

    validRows.forEach((r, rIdx) => {
      const rowVec = [];
      featureIndices.forEach((fIdx, colSeq) => {
        const val = r[fIdx];
        if (isCatCol[colSeq]) {
          const cats = catUniqueValues[colSeq];
          for (let k = 1; k < cats.length; k++) {
            rowVec.push(val === cats[k] ? 1.0 : 0.0);
          }
        } else {
          const num = Number(val);
          rowVec.push(!isNaN(num) && isFinite(num) ? num : numMedians[colSeq]);
        }
      });
      X_all.push(rowVec);
      y_all.push(Number(r[targetIdx]));
    });

    const N = X_all.length;
    const P = encodedNames.length;

    if (N <= P + 1) {
      throw new Error(`樣本數 (N=${N}) 小於等於特徵參數個數 (P+1=${P + 1})，無法進行多元回歸擬合！請減少特徵或增加資料筆數。`);
    }

    // 3. 隨機種子切分 Train / Test
    const rng = RegressionEngine.mulberry32(randomState);
    const shuffledIndices = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = shuffledIndices[i];
      shuffledIndices[i] = shuffledIndices[j];
      shuffledIndices[j] = tmp;
    }

    const testCount = Math.max(1, Math.floor(N * testSize));
    const trainCount = N - testCount;

    const trainIdxList = shuffledIndices.slice(0, trainCount);
    const testIdxList = shuffledIndices.slice(trainCount);

    const X_train = trainIdxList.map(i => X_all[i]);
    const y_train = trainIdxList.map(i => y_all[i]);
    const X_test = testIdxList.map(i => X_all[i]);
    const y_test = testIdxList.map(i => y_all[i]);
    const testOrigIds = testIdxList.map(i => originalRowIds[i]);

    // 4. 構建正規方程 (X_design^T * X_design + lambda*I) * beta = X_design^T * y
    // 設計矩陣加入第 0 欄全為 1.0 (截距項)
    const dim = P + 1;
    const XtX = Array.from({ length: dim }, () => new Float64Array(dim));
    const Xty = new Float64Array(dim);

    for (let i = 0; i < trainCount; i++) {
      const xRow = [1.0, ...X_train[i]];
      const yi = y_train[i];
      for (let r = 0; r < dim; r++) {
        Xty[r] += xRow[r] * yi;
        for (let c = 0; c < dim; c++) {
          XtX[r][c] += xRow[r] * xRow[c];
        }
      }
    }

    // 加入微小 Ridge 正規化 (lambda = 1e-9) 保障數值穩定性
    for (let d = 1; d < dim; d++) {
      XtX[d][d] += 1e-9;
    }

    // 求解係數 beta
    const beta = RegressionEngine.solveLinearSystem(XtX, Xty);
    const intercept = beta[0];
    const coefficients = {};
    encodedNames.forEach((name, idx) => {
      coefficients[name] = beta[idx + 1];
    });

    // 5. 預測函式
    const predictVector = (xVec) => {
      let pred = intercept;
      for (let j = 0; j < P; j++) {
        pred += coefficients[encodedNames[j]] * xVec[j];
      }
      return pred;
    };

    const y_train_pred = X_train.map(predictVector);
    const y_test_pred = X_test.map(predictVector);

    // 6. 計算評估指標
    const calcMetrics = (actuals, preds, nSamples, nParams) => {
      let sumAbsErr = 0;
      let sumSqErr = 0;
      let sumActual = 0;
      const count = actuals.length;

      for (let i = 0; i < count; i++) {
        const diff = actuals[i] - preds[i];
        sumAbsErr += Math.abs(diff);
        sumSqErr += diff * diff;
        sumActual += actuals[i];
      }

      const meanActual = sumActual / count;
      let sst = 0;
      for (let i = 0; i < count; i++) {
        const d = actuals[i] - meanActual;
        sst += d * d;
      }

      const mae = sumAbsErr / count;
      const mse = sumSqErr / count;
      const rmse = Math.sqrt(mse);
      const r2 = sst > 1e-12 ? (1.0 - sumSqErr / sst) : 0.0;
      const dfRes = nSamples - nParams - 1;
      const adjR2 = dfRes > 0 ? (1.0 - ((1.0 - r2) * (nSamples - 1) / dfRes)) : r2;

      return { mae, mse, rmse, r2, adjR2 };
    };

    const trainMetrics = calcMetrics(y_train, y_train_pred, trainCount, P);
    const testMetrics = calcMetrics(y_test, y_test_pred, testCount, P);

    // 7. 逐筆測試預測紀錄
    const testPredictions = y_test.map((act, i) => {
      const pred = y_test_pred[i];
      const residual = act - pred;
      return {
        originalId: testOrigIds[i],
        actual: act,
        predicted: pred,
        residual: residual,
        absError: Math.abs(residual)
      };
    });

    // 8. 產生公式文字
    const formulaParts = [`${intercept.toFixed(4)}`];
    encodedNames.forEach(name => {
      const coef = coefficients[name];
      const sign = coef >= 0 ? "+" : "-";
      formulaParts.push(`${sign} ${Math.abs(coef).toFixed(4)} * (${name})`);
    });
    const formulaString = `${targetCol} = ${formulaParts.join(" ")}`;

    return {
      success: true,
      datasetInfo: {
        totalRows: N,
        trainRows: trainCount,
        testRows: testCount,
        droppedRows: droppedCount,
        numFeatures: P,
        encodedFeatureNames: encodedNames,
        testSize,
        randomState
      },
      model: {
        intercept,
        coefficients,
        formula: formulaString
      },
      metrics: {
        train: trainMetrics,
        test: testMetrics
      },
      testPredictions
    };
  };

  window.RegressionEngine = RegressionEngine;

})(typeof window !== "undefined" ? window : global);
