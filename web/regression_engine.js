/**
 * Aether Quant Lab - Client-side Pure JavaScript Linear Regression Engine
 * 專為 GitHub Pages 靜態環境設計之瀏覽器端嚴謹多元線性回歸引擎
 * 特色：
 *   1. RFC 4180 標準 CSV 解析器 (支援引號、跨行、逗號、UTF-8 BOM、重複欄名檢測)
 *   2. 嚴格缺失值判定 (明確區分空字串、空白、null、undefined、NaN 與合法數值 0，杜絕 Number("") = 0 陷阱)
 *   3. 嚴格防止資料外洩：先切分 Train / Test，前處理參數 (中位數、One-Hot 類別) 100% 僅從訓練集學習
 *   4. 偶數筆中位數平滑插值 (中間兩數均值)
 *   5. 未知類別容錯防護 (未知類別編碼為全 0 基準類別，避免崩潰)
 *   6. 自由度充足性與模型可辨識性驗證 (訓練樣本數 > 特徵參數數 + 1)
 *   7. 數值穩定 OLS 求解器 (全主元高斯消去法)
 *   8. 嚴謹統計指標與「不適用」診斷 (SST <= 0 或 df <= 0 時回傳「不適用」)
 *   9. 匯出完整推論模型套件 (含欄位順序、填補中位數、One-Hot 字典、係數與截距)
 */

(function (window) {
  "use strict";

  const RegressionEngine = {};

  // 1. 缺失值嚴格判定輔助函式
  RegressionEngine.isMissing = function (val) {
    if (val === null || val === undefined) return true;
    const s = String(val).trim();
    if (s === "") return true;
    const lower = s.toLowerCase();
    return lower === "nan" || lower === "null" || lower === "none" || lower === "na" || lower === "n/a" || lower === "nil";
  };

  // 2. 數值嚴格解析函式 (合法 0 視為數值，空字串回傳 null)
  RegressionEngine.parseNumeric = function (val) {
    if (RegressionEngine.isMissing(val)) return null;
    const s = String(val).trim();
    const num = Number(s);
    if (!isNaN(num) && isFinite(num)) {
      return num;
    }
    return null;
  };

  // 3. 中位數計算 (偶數筆資料為中間兩數平均)
  RegressionEngine.computeMedian = function (numArr) {
    if (!numArr || numArr.length === 0) return null;
    const sorted = numArr.slice().sort((a, b) => a - b);
    const half = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
      return (sorted[half - 1] + sorted[half]) / 2.0;
    } else {
      return sorted[half];
    }
  };

  // 4. RFC 4180 CSV 解析器
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

    // 重複欄名檢測
    const seenHeaders = new Set();
    headers.forEach(h => {
      if (seenHeaders.has(h)) {
        throw new Error(`CSV 標題列包含重複的欄位名稱：「${h}」，請確保所有欄位名稱唯一！`);
      }
      seenHeaders.add(h);
    });

    const dataRows = rows.slice(1);
    return { headers, rows: dataRows };
  };

  // 5. 資料集品質分析與型別推斷
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
        if (RegressionEngine.isMissing(val)) {
          missingCount++;
        } else {
          const num = RegressionEngine.parseNumeric(val);
          if (num !== null) {
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

  // 6. 偽隨機生成器 (Mulberry32)
  RegressionEngine.mulberry32 = function (seed) {
    return function () {
      let t = (seed += 0x6D2B79F5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  // 7. 高斯消去法求解線性聯立方程組 (A * x = b)
  RegressionEngine.solveLinearSystem = function (A, b) {
    const n = b.length;
    const M = [];
    for (let i = 0; i < n; i++) {
      M[i] = new Float64Array(n + 1);
      for (let j = 0; j < n; j++) {
        M[i][j] = A[i][j];
      }
      M[i][n] = b[i];
    }

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

  // 8. 核心訓練、預測與評估 (嚴格先切分、杜絕資料外洩)
  RegressionEngine.fitAndEvaluate = function (config) {
    const {
      headers,
      rows,
      targetCol,
      featureCols,
      testSize = 0.2,
      randomState = 0
    } = config;

    // 基本規格驗證
    if (!targetCol) {
      throw new Error("必須指定目標欄位 (y)！");
    }
    if (!featureCols || featureCols.length === 0) {
      throw new Error("至少需要選取一個特徵變數 (X)！");
    }
    if (featureCols.includes(targetCol)) {
      throw new Error("目標欄位 (y) 不得同時被選為特徵變數 (X)！");
    }

    const targetIdx = headers.indexOf(targetCol);
    if (targetIdx === -1) {
      throw new Error(`找不到目標欄位：${targetCol}`);
    }

    const featureIndices = featureCols.map(c => {
      const idx = headers.indexOf(c);
      if (idx === -1) throw new Error(`找不到特徵欄位：${c}`);
      return idx;
    });

    if (typeof testSize !== "number" || isNaN(testSize) || testSize <= 0 || testSize >= 1) {
      throw new Error(`無效的測試集比例 (test_size=${testSize})，必須介於 0 與 1 之間 (例如 0.2)！`);
    }

    // 1. 處理無效目標值 y (若缺失或無法轉為數值，則移除該列並統計筆數，不填補目標值)
    const validRows = [];
    const validRowOriginalIds = [];
    let droppedRowCount = 0;

    rows.forEach((r, idx) => {
      const yVal = r[targetIdx];
      const yNum = RegressionEngine.parseNumeric(yVal);
      if (yNum !== null) {
        validRows.push(r);
        validRowOriginalIds.push(idx);
      } else {
        droppedRowCount++;
      }
    });

    if (validRows.length < 5) {
      throw new Error(`有效目標樣本數僅 ${validRows.length} 筆 (移除了 ${droppedRowCount} 筆缺失目標值)，樣本嚴重不足無法建模！`);
    }

    // 2. 嚴格切分訓練集與測試集 (Train-Test Split First)
    // 說明：此處採用 Mulberry32 PRNG 進行隨機抽樣，不同語言亂數演算法不同，相同 seed 不代表切分結果與 Python 完全相同。
    const N = validRows.length;
    const rng = RegressionEngine.mulberry32(randomState);
    const shuffledIndices = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = shuffledIndices[i];
      shuffledIndices[i] = shuffledIndices[j];
      shuffledIndices[j] = tmp;
    }

    let testCount = Math.floor(N * testSize);
    if (testCount < 1) testCount = 1;
    const trainCount = N - testCount;

    if (trainCount < 2) {
      throw new Error(`切分後訓練集僅有 ${trainCount} 筆樣本，無法擬合回歸模型！請調低 test_size 或增加資料。`);
    }

    const trainRowIndices = shuffledIndices.slice(0, trainCount);
    const testRowIndices = shuffledIndices.slice(trainCount);

    const trainRows = trainRowIndices.map(i => validRows[i]);
    const testRows = testRowIndices.map(i => validRows[i]);
    const testOriginalIds = testRowIndices.map(i => validRowOriginalIds[i]);

    // 3. 前處理規則純粹從「訓練集」學習
    const featureMetadata = [];
    const encodedNames = [];

    featureIndices.forEach((fIdx, colOrder) => {
      const colName = headers[fIdx];
      const trainColValues = trainRows.map(r => r[fIdx]);

      // 檢查是否為全部缺失欄位
      const nonMissingTrainVals = trainColValues.filter(v => !RegressionEngine.isMissing(v));
      if (nonMissingTrainVals.length === 0) {
        throw new Error(`欄位「${colName}」在訓練集中全部皆為缺失值，無法計算填補參數！`);
      }

      // 判斷是否為數值型欄位
      const isNum = nonMissingTrainVals.every(v => RegressionEngine.parseNumeric(v) !== null);

      if (isNum) {
        const validNums = nonMissingTrainVals.map(v => RegressionEngine.parseNumeric(v));
        const median = RegressionEngine.computeMedian(validNums);
        featureMetadata.push({
          type: "numeric",
          colName,
          fIdx,
          median: median !== null ? median : 0.0
        });
        encodedNames.push(colName);
      } else {
        // 類別型：收集訓練集不重複類別並排序
        const uniqueCategories = Array.from(new Set(nonMissingTrainVals)).sort();
        // 規避 Dummy Variable Trap: drop_first (第一個類別作為基準類別)
        const dummyCategories = uniqueCategories.slice(1);
        featureMetadata.push({
          type: "categorical",
          colName,
          fIdx,
          baseCategory: uniqueCategories[0] || "",
          dummyCategories: dummyCategories
        });
        dummyCategories.forEach(cat => {
          encodedNames.push(`${colName}_${cat}`);
        });
      }
    });

    const P = encodedNames.length;
    if (P === 0) {
      throw new Error("經前處理後未產生任何有效特徵變數！");
    }

    // 4. 模型自由度與可辨識性嚴格檢查 (Degrees of Freedom Check)
    if (trainCount <= P + 1) {
      throw new Error(`訓練樣本數不足無法辨識模型！切分後訓練樣本數 N_train=${trainCount}，但需估計參數個數 P+1=${P + 1}。請減少特徵數、降低 test_size 或提供更多樣本。`);
    }

    // 5. 向量化轉換 (訓練集與測試集依已學習的 metadata 轉換)
    const transformRow = (row) => {
      const vec = [];
      featureMetadata.forEach(meta => {
        const val = row[meta.fIdx];
        if (meta.type === "numeric") {
          const num = RegressionEngine.parseNumeric(val);
          vec.push(num !== null ? num : meta.median);
        } else {
          // 類別型：未知類別全設為 0 (未知類別容錯 fallback)
          meta.dummyCategories.forEach(cat => {
            vec.push(val === cat ? 1.0 : 0.0);
          });
        }
      });
      return vec;
    };

    const X_train = trainRows.map(transformRow);
    const y_train = trainRows.map(r => RegressionEngine.parseNumeric(r[targetIdx]));
    const X_test = testRows.map(transformRow);
    const y_test = testRows.map(r => RegressionEngine.parseNumeric(r[targetIdx]));

    // 6. 構建正規方程式求解 OLS
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

    // 微小數值防護 (保障數值計算不溢位)
    for (let d = 1; d < dim; d++) {
      XtX[d][d] += 1e-9;
    }

    const beta = RegressionEngine.solveLinearSystem(XtX, Xty);
    const intercept = beta[0];
    const coefficients = {};
    encodedNames.forEach((name, idx) => {
      coefficients[name] = beta[idx + 1];
    });

    // 7. 預測
    const predictVector = (xVec) => {
      let pred = intercept;
      for (let j = 0; j < P; j++) {
        pred += coefficients[encodedNames[j]] * xVec[j];
      }
      return pred;
    };

    const y_train_pred = X_train.map(predictVector);
    const y_test_pred = X_test.map(predictVector);

    // 8. 評估指標計算 (包含不適用診斷)
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
      
      let r2 = null;
      let r2Display = "不適用";
      if (sst > 1e-12) {
        r2 = 1.0 - sumSqErr / sst;
        r2Display = r2.toFixed(4);
      }

      const dfRes = nSamples - nParams - 1;
      let adjR2 = null;
      let adjR2Display = "不適用";
      if (r2 !== null && dfRes > 0) {
        adjR2 = 1.0 - ((1.0 - r2) * (nSamples - 1) / dfRes);
        adjR2Display = adjR2.toFixed(4);
      }

      return { mae, mse, rmse, r2, r2Display, adjR2, adjR2Display };
    };

    const trainMetrics = calcMetrics(y_train, y_train_pred, trainCount, P);
    const testMetrics = calcMetrics(y_test, y_test_pred, testCount, P);

    // 9. 逐筆測試預測紀錄
    const testPredictions = y_test.map((act, i) => {
      const pred = y_test_pred[i];
      const residual = act - pred;
      return {
        originalId: testOriginalIds[i],
        actual: act,
        predicted: pred,
        residual: residual,
        absError: Math.abs(residual)
      };
    });

    // 10. 公式字串
    const formulaParts = [`${intercept.toFixed(4)}`];
    encodedNames.forEach(name => {
      const coef = coefficients[name];
      const sign = coef >= 0 ? "+" : "-";
      formulaParts.push(`${sign} ${Math.abs(coef).toFixed(4)} * (${name})`);
    });
    const formulaString = `${targetCol} = ${formulaParts.join(" ")}`;

    // 11. 匯出完整推論套件 (Deployment Pipeline Package)
    const exportableModel = {
      model_type: "PureJS_LinearRegression",
      target_column: targetCol,
      raw_feature_columns: featureCols,
      encoded_feature_names: encodedNames,
      feature_metadata: featureMetadata,
      intercept: intercept,
      coefficients: coefficients,
      training_summary: {
        total_rows_input: rows.length,
        dropped_invalid_y_rows: droppedRowCount,
        valid_rows: N,
        train_rows: trainCount,
        test_rows: testCount,
        test_size: testSize,
        random_state: randomState,
        random_algorithm_note: "JavaScript 使用 Mulberry32 PRNG，與 Python MT19937 之隨機數列不同，相同 seed 不代表相同切分。"
      },
      metrics: {
        train: trainMetrics,
        test: testMetrics
      }
    };

    return {
      success: true,
      datasetInfo: {
        totalRows: rows.length,
        droppedRows: droppedRowCount,
        validRows: N,
        trainRows: trainCount,
        testRows: testCount,
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
      testPredictions,
      exportableModel
    };
  };

  window.RegressionEngine = RegressionEngine;

})(typeof window !== "undefined" ? window : global);
