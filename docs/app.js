/**
 * Aether Quant Lab - Core Application Controller (app.js)
 * Supports dynamic 5-algorithm benchmarks, Occam's Razor Valley at k=2 highlights,
 * problem 1 / problem 2 seamless display switching, custom dataset runner,
 * real-time OLS inference simulator, and responsive Chart.js vector rendering.
 */

// Global Application State
let APP_DATA = null;
let currentAlgo = "Forward SFS (Lecture Stepwise)";
let currentK = 2; // Default to k=2 (Optimal Valley)
let currentP1Chart = "actual_vs_pred";
let currentBostonChart = "actual_vs_pred";
let currentCustomChart = "actual_vs_pred";
let currentProblemView = "all"; // "part1", "part2", "boston", "custom", "all"
let chartRmseInst = null;
let chartR2Inst = null;
let chartP1Inst = null;
let chartBostonInst = null;
let chartCustomInst = null;

let currentCustomParsed = null;
let currentCustomResult = null;

// Currency & Number Formatters
const formatCurrency = (val) => {
  if (val === undefined || val === null || isNaN(val)) return "$0.00";
  return "$" + Number(val).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

const formatNumber = (val, decimals = 4) => {
  if (val === undefined || val === null || isNaN(val)) return "0";
  return Number(val).toFixed(decimals);
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  fetchData();
  setupEventListeners();
  initSimulator();
  initBostonHousing();
  initCustomDatasetEngine();
});

// Theme Toggle (Dark / Light)
function initTheme() {
  const savedTheme = localStorage.getItem("aether-theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeIcon(savedTheme);

  const toggleBtn = document.getElementById("theme-toggle-btn");
  if (toggleBtn) {
    toggleBtn.onclick = () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("aether-theme", next);
      updateThemeIcon(next);
      
      // Re-render charts with updated theme colors
      if (APP_DATA) {
        renderPart2Charts();
        drawPart1Chart(currentP1Chart);
      }
    };
  }
}

function updateThemeIcon(theme) {
  const icon = document.querySelector("#theme-toggle-btn i");
  if (icon) {
    icon.className = theme === "dark" ? "fa-solid fa-sun" : "fa-solid fa-moon";
  }
}

// Fetch JSON Data (with fallback to embedded data)
async function fetchData() {
  let data = null;

  try {
    const res = await fetch("data.json");
    if (res.ok) {
      data = await res.json();
    }
  } catch (err) {
    console.log("Fetch failed, falling back to window.EMBEDDED_DATA...");
  }

  if (!data && typeof window.EMBEDDED_DATA !== "undefined") {
    data = window.EMBEDDED_DATA;
    console.log("Using window.EMBEDDED_DATA successfully!");
  }

  if (data) {
    APP_DATA = data;
    renderAll(data);
  } else {
    alert("無法載入實驗數據，請確保已啟動 HTTP 伺服器 (python3 -m http.server 8000 --directory web)");
  }
}

// Master Render Pipeline
function renderAll(data) {
  renderCrispDM(data.crisp_dm_stages);
  renderPart1(data.part1);
  renderPart2(data.part2);
  if (data.boston_housing) {
    renderBostonHousing(data.boston_housing);
  }
  renderDiscussion(data.discussion);
}

// 1. Render CRISP-DM Steps
function renderCrispDM(stages) {
  const container = document.getElementById("crisp-container");
  if (!container || !stages) return;

  container.innerHTML = stages.map((st, idx) => `
    <div class="crisp-step-card ${idx === 0 ? 'active' : ''}" data-step="${idx + 1}" style="cursor: pointer;">
      <div class="crisp-num">PHASE 0${idx + 1}</div>
      <div class="crisp-title">${st.phase_name_zh}</div>
      <div class="crisp-sub">${st.goal}</div>
      <div class="crisp-expanded-detail" style="display: ${idx === 0 ? 'block' : 'none'}; margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--border-subtle); font-size: 0.76rem; color: var(--cyan-primary);">
        <i class="fa-solid fa-lightbulb"></i> 實作要點：${st.key_takeaway || '遵循 CRISP-DM 標準流程'}
      </div>
    </div>
  `).join("");

  // Click card to toggle detail
  container.querySelectorAll(".crisp-step-card").forEach(card => {
    card.onclick = () => {
      container.querySelectorAll(".crisp-step-card").forEach(c => {
        c.classList.remove("active");
        const detail = c.querySelector(".crisp-expanded-detail");
        if (detail) detail.style.display = "none";
      });
      card.classList.add("active");
      const myDetail = card.querySelector(".crisp-expanded-detail");
      if (myDetail) myDetail.style.display = "block";
    };
  });
}

// 2. Render Part 1 Baseline Model
function renderPart1(part1) {
  if (!part1) return;

  const testTbody = document.getElementById("p1-test-tbody");
  if (testTbody && part1.test_predictions) {
    testTbody.innerHTML = part1.test_predictions.map(r => `
      <tr>
        <td class="num-cell" style="font-weight: 700;">#${r.original_id}</td>
        <td><span class="badge-tag tag-neutral">${r.state}</span></td>
        <td class="num-cell">${formatCurrency(r.rd_spend)}</td>
        <td class="num-cell">${formatCurrency(r.admin_spend)}</td>
        <td class="num-cell">${formatCurrency(r.marketing_spend)}</td>
        <td class="num-cell" style="font-weight: 700;">${formatCurrency(r.actual_profit)}</td>
        <td class="num-cell" style="color: var(--cyan-primary); font-weight: 700;">${formatCurrency(r.predicted_profit)}</td>
        <td class="num-cell" style="color: ${r.residual >= 0 ? 'var(--emerald-optimal)' : 'var(--rose-danger)'}; font-weight: 600;">
          ${r.residual >= 0 ? '+' : ''}${formatCurrency(r.residual)}
        </td>
        <td class="num-cell">${formatCurrency(r.abs_error)}</td>
      </tr>
    `).join("");
  }

  drawPart1Chart(currentP1Chart);
}

// Draw Part 1 Scatter & Residual Charts
function drawPart1Chart(chartType) {
  const canvas = document.getElementById("chart-p1-eval");
  const imgEl = document.getElementById("p1-python-img");
  if (!canvas || !APP_DATA) return;

  if (chartType === "python_img") {
    if (imgEl) imgEl.style.display = "block";
    canvas.style.display = "none";
    return;
  }

  if (imgEl) imgEl.style.display = "none";
  canvas.style.display = "block";

  const preds = APP_DATA.part1.test_predictions;
  if (chartP1Inst) chartP1Inst.destroy();

  const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";
  const textColor = isDark ? "#94a3b8" : "#475569";

  if (chartType === "actual_vs_pred") {
    const points = preds.map(p => ({ x: p.actual_profit, y: p.predicted_profit, id: p.original_id }));
    const minVal = 60000, maxVal = 200000;

    chartP1Inst = new Chart(canvas, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '理想擬合線 (y = x)',
            data: [{x: minVal, y: minVal}, {x: maxVal, y: maxVal}],
            type: 'line',
            borderColor: '#f43f5e',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          },
          {
            label: '測試樣本 (Actual vs Predicted)',
            data: points,
            backgroundColor: '#06b6d4',
            borderColor: '#fff',
            borderWidth: 1.5,
            pointRadius: 6,
            pointHoverRadius: 9
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor, font: { family: 'Inter', size: 12 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => `Sample #${ctx.raw.id}: 實際 ${formatCurrency(ctx.raw.x)} | 預測 ${formatCurrency(ctx.raw.y)}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: '實際獲利 (Actual Profit $)', color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor, callback: v => '$' + (v / 1000) + 'k' }
          },
          y: {
            title: { display: true, text: '模型預測獲利 (Predicted Profit $)', color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor, callback: v => '$' + (v / 1000) + 'k' }
          }
        }
      }
    });
  } else if (chartType === "residual") {
    const residuals = preds.map(p => ({ x: p.predicted_profit, y: p.residual, id: p.original_id }));

    chartP1Inst = new Chart(canvas, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '零殘差基準線 (e = 0)',
            data: [{x: 60000, y: 0}, {x: 200000, y: 0}],
            type: 'line',
            borderColor: '#f59e0b',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          },
          {
            label: '預測殘差 (Residuals)',
            data: residuals,
            backgroundColor: (ctx) => (ctx.raw && ctx.raw.y >= 0 ? '#10b981' : '#f43f5e'),
            borderColor: '#fff',
            borderWidth: 1.5,
            pointRadius: 6,
            pointHoverRadius: 9
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor, font: { family: 'Inter', size: 12 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => `Sample #${ctx.raw.id}: 殘差 ${formatCurrency(ctx.raw.y)}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: '預測獲利 (Fitted Value $)', color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor, callback: v => '$' + (v / 1000) + 'k' }
          },
          y: {
            title: { display: true, text: '殘差 (Residuals $)', color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor, callback: v => formatCurrency(v) }
          }
        }
      }
    });
  }
}

// 3. Render Part 2 Feature Selection & Occam's Razor Valley
function renderPart2(part2) {
  if (!part2) return;

  renderSummaryTable();
  updateKPIsForCurrentSelection();
  renderPart2Charts();
}

// Safe metric display helper (preserves 0, shows fallback on missing)
function displayMetric(val, formatter, fallback = "未計算") {
  if (val === null || val === undefined || Number.isNaN(val) || val === "") {
    return fallback;
  }
  return formatter ? formatter(val) : String(val);
}

// Normalize Algorithm Row into consistent schema
function normalizeAlgoRow(r) {
  const test_rmse = typeof r.test_rmse === "number" ? r.test_rmse : (typeof r.rmse === "number" ? r.rmse : null);
  const test_r2 = typeof r.test_r2 === "number" ? r.test_r2 : (typeof r.r2 === "number" ? r.r2 : null);
  const test_mae = typeof r.test_mae === "number" ? r.test_mae : (typeof r.mae === "number" ? r.mae : null);
  const test_mse = typeof r.test_mse === "number" ? r.test_mse : null;
  const train_rmse = typeof r.train_rmse === "number" ? r.train_rmse : null;
  const train_r2 = typeof r.train_r2 === "number" ? r.train_r2 : null;
  const adj_r2 = typeof r.train_adj_r2 === "number" ? r.train_adj_r2 : (typeof r.adj_r2 === "number" ? r.adj_r2 : null);
  const f_stat = typeof r.train_f_stat === "number" ? r.train_f_stat : (typeof r.f_stat === "number" ? r.f_stat : null);

  const features = Array.isArray(r.features) 
    ? r.features 
    : (r.selected_features_str ? r.selected_features_str.split(", ").filter(Boolean) : []);

  return {
    ...r,
    k: r.k || r.num_features || features.length,
    features,
    test_rmse,
    test_r2,
    test_mae,
    test_mse,
    train_rmse,
    train_r2,
    adj_r2,
    f_stat,
    // Aliases for compatibility
    rmse: test_rmse,
    r2: test_r2,
    mae: test_mae
  };
}

// Retrieve data points for selected algorithm (NO unauthorized fallback to other algorithms)
function getAlgorithmData(algoName) {
  if (!APP_DATA) return [];
  if (APP_DATA.algorithms_benchmark && APP_DATA.algorithms_benchmark[algoName]) {
    return APP_DATA.algorithms_benchmark[algoName].map(normalizeAlgoRow);
  }
  // If Forward SFS not in benchmark map, check part2.progression_table for that specific algorithm only
  if (algoName.includes("Forward SFS") && APP_DATA.part2 && APP_DATA.part2.progression_table) {
    return APP_DATA.part2.progression_table.map(normalizeAlgoRow);
  }
  // Do NOT fake results with another algorithm
  return [];
}

// Render Summary Table for Current Algorithm
function renderSummaryTable() {
  const summaryTbody = document.getElementById("p2-summary-tbody");
  if (!summaryTbody || !APP_DATA) return;

  const algoData = getAlgorithmData(currentAlgo);
  if (!algoData || algoData.length === 0) {
    summaryTbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--rose-danger); padding: 1.5rem;">尚未收錄演算法「${currentAlgo}」之實測數據（不套用假資料）。</td></tr>`;
    return;
  }

  // Find minimum test_rmse to identify optimal valley objectively
  const validRmses = algoData.map(r => r.test_rmse).filter(v => typeof v === "number");
  const minRmse = validRmses.length > 0 ? Math.min(...validRmses) : null;

  summaryTbody.innerHTML = algoData.map(row => {
    const isOptimal = minRmse !== null && row.test_rmse === minRmse;
    const lastFeat = row.features.length > 0 ? row.features[row.features.length - 1] : "—";
    const featStr = row.features.length > 0 ? row.features.join(" + ") : "無特徵";

    return `
      <tr style="${isOptimal ? 'background: rgba(16, 185, 129, 0.08);' : ''}">
        <td class="num-cell" style="font-weight: 800; color: ${isOptimal ? 'var(--emerald-optimal)' : 'var(--text-main)'};">k = ${row.k}</td>
        <td><strong>${lastFeat}</strong></td>
        <td><code style="font-size: 0.8rem; color: var(--text-code);">${featStr}</code></td>
        <td class="num-cell" style="font-weight: 800; color: ${isOptimal ? 'var(--emerald-optimal)' : 'var(--text-main)'};">
          ${displayMetric(row.test_rmse, formatCurrency)}
        </td>
        <td class="num-cell" style="font-weight: 800; color: ${isOptimal ? 'var(--emerald-optimal)' : 'var(--text-main)'};">
          ${displayMetric(row.test_r2, v => formatNumber(v, 4))}
        </td>
        <td class="num-cell" style="color: var(--text-dim);">${displayMetric(row.train_rmse, formatCurrency)}</td>
        <td class="num-cell" style="color: var(--text-dim);">${displayMetric(row.train_r2, v => formatNumber(v, 4))}</td>
        <td>
          ${isOptimal 
            ? '<span class="badge-tag tag-sweet">⭐ 最優谷地 (最低 Test RMSE)</span>' 
            : (row.k === 5 ? '<span class="badge-tag tag-info">全特徵 (基準模型)</span>' : '<span class="badge-tag tag-neutral">候選步進</span>')}
        </td>
      </tr>
    `;
  }).join("");
}

// Update 5 KPI Cards for current K and Algorithm
function updateKPIsForCurrentSelection() {
  const algoData = getAlgorithmData(currentAlgo);
  if (!algoData || algoData.length === 0) {
    ["val-kpi-rmse", "val-kpi-r2", "val-kpi-adjr2", "val-kpi-mae", "val-kpi-fstat"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = "未計算";
    });
    return;
  }

  const row = algoData.find(x => x.k === currentK) || algoData[0];
  if (!row) return;

  const validRmses = algoData.map(r => r.test_rmse).filter(v => typeof v === "number");
  const minRmse = validRmses.length > 0 ? Math.min(...validRmses) : null;
  const isOptimal = minRmse !== null && row.test_rmse === minRmse;

  // 1. Test RMSE Card
  const rmseEl = document.getElementById("val-kpi-rmse");
  if (rmseEl) rmseEl.textContent = displayMetric(row.test_rmse, formatCurrency);
  const rmseBadge = document.getElementById("badge-kpi-rmse");
  if (rmseBadge) {
    rmseBadge.textContent = isOptimal ? `最低點谷地 (k = ${row.k})` : `特徵數 k = ${row.k}`;
    rmseBadge.className = isOptimal ? "badge-tag tag-sweet" : "badge-tag tag-neutral";
  }
  const rmseCard = document.getElementById("card-kpi-rmse");
  if (rmseCard) {
    rmseCard.className = isOptimal ? "kpi-card highlight-optimal" : "kpi-card";
  }

  // 2. Test R² Card
  const r2El = document.getElementById("val-kpi-r2");
  if (r2El) r2El.textContent = displayMetric(row.test_r2, v => formatNumber(v, 4));
  const r2Badge = document.getElementById("badge-kpi-r2");
  if (r2Badge) {
    r2Badge.textContent = typeof row.test_r2 === "number" ? `${(row.test_r2 * 100).toFixed(1)}% FIT` : "—";
  }

  // 3. Adjusted R² Card (Train/Model OLS Adj R2)
  const adjR2El = document.getElementById("val-kpi-adjr2");
  if (adjR2El) adjR2El.textContent = displayMetric(row.adj_r2, v => formatNumber(v, 4));

  // 4. Test MAE Card
  const maeEl = document.getElementById("val-kpi-mae");
  if (maeEl) maeEl.textContent = displayMetric(row.test_mae, formatCurrency);

  // 5. F-Statistic Card
  const fstatEl = document.getElementById("val-kpi-fstat");
  if (fstatEl) fstatEl.textContent = displayMetric(row.f_stat, v => formatNumber(v, 2));

  // Update Valley Hero Banner description
  const descEl = document.getElementById("valley-desc-text");
  if (descEl) {
    const featStr = row.features.length > 0 ? row.features.join(", ") : "未選取";
    descEl.innerHTML = `當前演算法：<strong>${currentAlgo}</strong>，特徵數 <strong>k = ${row.k}</strong>（包含特徵：<span style="color: var(--cyan-primary);">${featStr}</span>）。實測 Test RMSE：<strong>${displayMetric(row.test_rmse, formatCurrency)}</strong>，Test R²：<strong>${displayMetric(row.test_r2, v => formatNumber(v, 4))}</strong>。`;
  }
}

// Render Part 2 Charts (Supports Single Algorithm & Compare All 5)
function renderPart2Charts() {
  const canvasRmse = document.getElementById("chart-rmse");
  const canvasR2 = document.getElementById("chart-r2");
  if (!canvasRmse || !canvasR2 || !APP_DATA) return;

  const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";
  const textColor = isDark ? "#94a3b8" : "#475569";

  const isCompareAll = currentAlgo === "COMPARE_ALL";

  // Palette for 5 Algorithms
  const ALGO_COLORS = {
    'Forward SFS (Lecture Stepwise)': '#f97316', // Orange
    'Forward SFS': '#06b6d4',                   // Cyan
    'Lasso (L1)': '#8b5cf6',                     // Violet
    'RFE': '#ec4899',                            // Pink
    'ElasticNet': '#3b82f6',                     // Blue
    'Exhaustive': '#10b981'                      // Emerald
  };

  const algosToRender = isCompareAll 
    ? ['Forward SFS (Lecture Stepwise)', 'Forward SFS', 'Lasso (L1)', 'RFE', 'ElasticNet', 'Exhaustive']
    : [currentAlgo];

  // 1. Prepare Datasets for RMSE
  const datasetsRmse = algosToRender.map(algo => {
    const dataList = getAlgorithmData(algo);
    const color = ALGO_COLORS[algo] || '#06b6d4';
    return {
      label: algo,
      data: dataList.map(item => item.rmse),
      borderColor: color,
      backgroundColor: isCompareAll ? 'transparent' : 'rgba(249, 115, 22, 0.08)',
      borderWidth: isCompareAll ? 2.5 : 3.5,
      tension: 0.2,
      fill: !isCompareAll,
      pointRadius: (ctx) => {
        if (!isCompareAll && ctx.dataIndex === 1) return 10; // Highlight k=2 (Optimal Valley)
        return isCompareAll ? 5 : 6;
      },
      pointBackgroundColor: (ctx) => {
        if (ctx.dataIndex === 1) return '#10b981'; // Emerald for k=2
        return color;
      },
      pointBorderColor: '#ffffff',
      pointBorderWidth: 2
    };
  });

  // 2. Prepare Datasets for R2
  const datasetsR2 = algosToRender.map(algo => {
    const dataList = getAlgorithmData(algo);
    const color = ALGO_COLORS[algo] || '#10b981';
    return {
      label: algo,
      data: dataList.map(item => item.r2),
      borderColor: color,
      backgroundColor: isCompareAll ? 'transparent' : 'rgba(16, 185, 129, 0.08)',
      borderWidth: isCompareAll ? 2.5 : 3.5,
      tension: 0.2,
      fill: !isCompareAll,
      pointRadius: (ctx) => {
        if (!isCompareAll && ctx.dataIndex === 1) return 10;
        return isCompareAll ? 5 : 6;
      },
      pointBackgroundColor: (ctx) => {
        if (ctx.dataIndex === 1) return '#10b981';
        return color;
      },
      pointBorderColor: '#ffffff',
      pointBorderWidth: 2
    };
  });

  // Build RMSE Chart
  if (chartRmseInst) chartRmseInst.destroy();
  chartRmseInst = new Chart(canvasRmse, {
    type: 'line',
    data: {
      labels: ['k=1 (1特徵)', 'k=2 ⭐谷地', 'k=3 (3特徵)', 'k=4 (4特徵)', 'k=5 (全特徵)'],
      datasets: datasetsRmse
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: isCompareAll,
          labels: { color: textColor, font: { family: 'Inter', size: 11 } }
        },
        tooltip: {
          callbacks: {
            title: (items) => {
              const idx = items[0].dataIndex;
              return idx === 1 ? '⭐ 最優谷地 (Valley @ k=2)' : `特徵數量：k = ${idx + 1}`;
            },
            label: (ctx) => `${ctx.dataset.label}: ${formatCurrency(ctx.raw)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'Inter', weight: 600 } }
        },
        y: {
          min: 8000,
          max: 9350,
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            callback: (val) => '$' + Number(val).toLocaleString()
          },
          title: {
            display: true,
            text: 'Test RMSE ($) - 越低越好',
            color: textColor,
            font: { family: 'Inter', size: 12, weight: 600 }
          }
        }
      }
    }
  });

  // Build R² Chart
  if (chartR2Inst) chartR2Inst.destroy();
  chartR2Inst = new Chart(canvasR2, {
    type: 'line',
    data: {
      labels: ['k=1 (1特徵)', 'k=2 ⭐峰值', 'k=3 (3特徵)', 'k=4 (4特徵)', 'k=5 (全特徵)'],
      datasets: datasetsR2
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: isCompareAll,
          labels: { color: textColor, font: { family: 'Inter', size: 11 } }
        },
        tooltip: {
          callbacks: {
            title: (items) => {
              const idx = items[0].dataIndex;
              return idx === 1 ? '⭐ 最高解釋峰值 (Peak @ k=2)' : `特徵數量：k = ${idx + 1}`;
            },
            label: (ctx) => `${ctx.dataset.label}: R² = ${formatNumber(ctx.raw, 4)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: 'Inter', weight: 600 } }
        },
        y: {
          min: 0.930,
          max: 0.952,
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            callback: (val) => Number(val).toFixed(3)
          },
          title: {
            display: true,
            text: 'Test R² (Variance Explained) - 越高越好',
            color: textColor,
            font: { family: 'Inter', size: 12, weight: 600 }
          }
        }
      }
    }
  });
}

// 4. Render Empirical Discussion
function renderDiscussion(discussion) {
  const container = document.getElementById("insights-container");
  if (!container || !discussion || !discussion.insights) return;

  container.innerHTML = discussion.insights.map(item => `
    <div class="card">
      <div class="card-header">
        <div class="card-title-group">
          <h3><i class="fa-solid fa-lightbulb" style="color: var(--amber-warning);"></i> ${item.title}</h3>
        </div>
      </div>
      <p style="color: var(--text-muted); font-size: 0.92rem; line-height: 1.7;">${item.content}</p>
    </div>
  `).join("");
}

// 5. Interactive Real-Time Prediction Simulator
function initSimulator() {
  const sliderRd = document.getElementById("slider-sim-rd");
  const sliderAdmin = document.getElementById("slider-sim-admin");
  const sliderMarketing = document.getElementById("slider-sim-marketing");
  const stateRadios = document.querySelectorAll("input[name='sim-state']");

  const lblRd = document.getElementById("lbl-sim-rd");
  const lblAdmin = document.getElementById("lbl-sim-admin");
  const lblMarketing = document.getElementById("lbl-sim-marketing");
  const resultProfit = document.getElementById("sim-predicted-profit");

  function calculatePrediction() {
    if (!sliderRd || !sliderAdmin || !sliderMarketing) return;

    const rd = parseFloat(sliderRd.value);
    const admin = parseFloat(sliderAdmin.value);
    const marketing = parseFloat(sliderMarketing.value);

    let state = "California";
    stateRadios.forEach(r => { if (r.checked) state = r.value; });

    let stateFlorida = state === "Florida" ? 1 : 0;
    let stateNewYork = state === "New York" ? 1 : 0;

    // Read full precision coefficients dynamically from export payload
    const p1Params = APP_DATA && APP_DATA.part1 && APP_DATA.part1.model_parameters;
    let intercept = 42554.17203744044;
    let coefFL = -959.284160065287;
    let coefNY = 699.3690525796245;
    let coefRD = 0.7734671927326042;
    let coefAdmin = 0.03288459468940801;
    let coefMkt = 0.03661002585141011;

    if (p1Params && p1Params.coefficients) {
      if (typeof p1Params.intercept === "number") intercept = p1Params.intercept;
      const c = p1Params.coefficients;
      if (typeof c["State_Florida"] === "number") coefFL = c["State_Florida"];
      if (typeof c["State_New York"] === "number") coefNY = c["State_New York"];
      if (typeof c["R&D Spend"] === "number") coefRD = c["R&D Spend"];
      if (typeof c["Administration"] === "number") coefAdmin = c["Administration"];
      if (typeof c["Marketing Spend"] === "number") coefMkt = c["Marketing Spend"];
    }

    const profit = intercept 
      + (coefFL * stateFlorida) 
      + (coefNY * stateNewYork) 
      + (coefRD * rd) 
      + (coefAdmin * admin) 
      + (coefMkt * marketing);

    if (lblRd) lblRd.textContent = formatCurrency(rd);
    if (lblAdmin) lblAdmin.textContent = formatCurrency(admin);
    if (lblMarketing) lblMarketing.textContent = formatCurrency(marketing);
    if (resultProfit) resultProfit.textContent = formatCurrency(profit);
  }

  if (sliderRd) sliderRd.oninput = calculatePrediction;
  if (sliderAdmin) sliderAdmin.oninput = calculatePrediction;
  if (sliderMarketing) sliderMarketing.oninput = calculatePrediction;
  stateRadios.forEach(r => r.onchange = calculatePrediction);

  calculatePrediction();
}

// ====================================================================
// TOPIC 2: BOSTON HOUSING RENDERING & CHARTS
// ====================================================================
function initBostonHousing() {
  const chartToggleBtns = document.querySelectorAll("#boston-chart-toggle .segmented-btn");
  chartToggleBtns.forEach(btn => {
    btn.onclick = () => {
      chartToggleBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentBostonChart = btn.dataset.chart;
      drawBostonChart(currentBostonChart);
    };
  });
}

function renderBostonHousing(bostonData) {
  if (!bostonData) return;

  const rowCountEl = document.getElementById("boston-row-count");
  const r2El = document.getElementById("boston-r2");
  const rmseEl = document.getElementById("boston-rmse");
  const maeEl = document.getElementById("boston-mae");
  const formulaEl = document.getElementById("boston-formula-text");

  if (bostonData.metrics && bostonData.metrics.test) {
    const tm = bostonData.metrics.test;
    if (r2El) r2El.textContent = `${tm.r2.toFixed(4)} (${(tm.r2 * 100).toFixed(1)}%)`;
    if (rmseEl) rmseEl.textContent = `$${(tm.rmse * 1000).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    if (maeEl) maeEl.textContent = `$${(tm.mae * 1000).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  if (formulaEl && bostonData.formula) {
    formulaEl.innerHTML = `<strong>MEDV</strong> = ${bostonData.formula.replace("medv = ", "")}`;
  }

  drawBostonChart(currentBostonChart);
}

function drawBostonChart(mode = "actual_vs_pred") {
  const canvas = document.getElementById("chart-boston");
  if (!canvas || !APP_DATA || !APP_DATA.boston_housing || !APP_DATA.boston_housing.test_predictions) return;

  const preds = APP_DATA.boston_housing.test_predictions;
  const ctx = canvas.getContext("2d");

  if (chartBostonInst) {
    chartBostonInst.destroy();
    chartBostonInst = null;
  }

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.08)";
  const textColor = isDark ? "#94a3b8" : "#475569";

  if (mode === "actual_vs_pred") {
    const scatterData = preds.map(p => ({ x: p.actual, y: p.predicted }));
    const allVals = preds.flatMap(p => [p.actual, p.predicted]);
    const minVal = Math.min(...allVals) * 0.9;
    const maxVal = Math.max(...allVals) * 1.1;

    chartBostonInst = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "波士頓測試集樣本 (102 筆)",
            data: scatterData,
            backgroundColor: "rgba(139, 92, 246, 0.8)",
            borderColor: "rgba(139, 92, 246, 1)",
            borderWidth: 1.5,
            pointRadius: 5,
            pointHoverRadius: 7
          },
          {
            type: "line",
            label: "理想擬合線 (y = x)",
            data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
            borderColor: "#ef4444",
            borderDash: [6, 6],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor } },
          tooltip: {
            callbacks: {
              label: (ctx) => `實際: $${(ctx.parsed.x * 1000).toFixed(0)} | 預測: $${(ctx.parsed.y * 1000).toFixed(0)}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: "實際中位數房價 MEDV (千美元)", color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor }
          },
          y: {
            title: { display: true, text: "預測中位數房價 MEDV (千美元)", color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor }
          }
        }
      }
    });
  } else {
    const residualData = preds.map(p => ({ x: p.predicted, y: p.residual }));

    chartBostonInst = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "測試集殘差分佈 (Residuals)",
            data: residualData,
            backgroundColor: "rgba(239, 68, 68, 0.8)",
            borderColor: "rgba(239, 68, 68, 1)",
            borderWidth: 1.5,
            pointRadius: 5
          },
          {
            type: "line",
            label: "零誤差基準線 (Zero Residual)",
            data: [
              { x: Math.min(...preds.map(p => p.predicted)), y: 0 },
              { x: Math.max(...preds.map(p => p.predicted)), y: 0 }
            ],
            borderColor: "#94a3b8",
            borderDash: [4, 4],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor } }
        },
        scales: {
          x: {
            title: { display: true, text: "預測房價 MEDV (千美元)", color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor }
          },
          y: {
            title: { display: true, text: "殘差值 e = y - ŷ (千美元)", color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor }
          }
        }
      }
    });
  }
}

// ====================================================================
// 6. CUSTOM DATASET ENGINE (加分功能 2: 全面真實純前端回歸引擎)
// ====================================================================
function initCustomDatasetEngine() {
  const urlInput = document.getElementById("custom-dataset-url");
  const btnFetchUrl = document.getElementById("btn-fetch-custom-url");
  const fileInput = document.getElementById("custom-file-input");
  const statusEl = document.getElementById("custom-dataset-status");
  const configPanel = document.getElementById("custom-config-panel");
  const resultsPanel = document.getElementById("custom-dataset-results");

  const targetSelect = document.getElementById("custom-target-select");
  const featuresContainer = document.getElementById("custom-features-container");
  const btnRunTraining = document.getElementById("btn-run-custom-training");
  const btnSelectAll = document.getElementById("btn-select-all-features");
  const btnClearAll = document.getElementById("btn-clear-all-features");

  const setStatus = (type, message) => {
    statusEl.style.display = "block";
    if (type === "loading") {
      statusEl.style.background = "rgba(6, 182, 212, 0.15)";
      statusEl.style.border = "1px solid rgba(6, 182, 212, 0.4)";
      statusEl.style.color = "var(--cyan-primary)";
      statusEl.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> ${message}`;
    } else if (type === "success") {
      statusEl.style.background = "rgba(16, 185, 129, 0.15)";
      statusEl.style.border = "1px solid rgba(16, 185, 129, 0.4)";
      statusEl.style.color = "var(--emerald-optimal)";
      statusEl.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${message}`;
    } else if (type === "error") {
      statusEl.style.background = "rgba(244, 63, 94, 0.15)";
      statusEl.style.border = "1px solid rgba(244, 63, 94, 0.4)";
      statusEl.style.color = "var(--rose-danger)";
      statusEl.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${message}`;
    }
  };

  const processRawCSV = (csvText, sourceName) => {
    try {
      if (!window.RegressionEngine) {
        throw new Error("RegressionEngine 函式庫未載入，請確認 regression_engine.js 存在！");
      }

      setStatus("loading", `正在解析 ${sourceName}...`);
      const parsed = window.RegressionEngine.parseCSV(csvText);
      const analysis = window.RegressionEngine.analyzeDataset(parsed.headers, parsed.rows);

      currentCustomParsed = parsed;
      currentCustomResult = null; // 重設前次成果

      // 隱藏前次結果，顯示設定面板
      if (resultsPanel) resultsPanel.style.display = "none";
      if (configPanel) configPanel.style.display = "block";

      // 更新統計摘要
      document.getElementById("cfg-row-count").textContent = parsed.rows.length;
      document.getElementById("cfg-col-count").textContent = parsed.headers.length;

      // 預覽前 5 筆
      const thead = document.getElementById("custom-preview-thead");
      const tbody = document.getElementById("custom-preview-tbody");
      thead.innerHTML = `<tr>${parsed.headers.map(h => `<th>${h}</th>`).join("")}</tr>`;
      tbody.innerHTML = parsed.rows.slice(0, 5).map(r => `<tr>${r.map(v => `<td>${v}</td>`).join("")}</tr>`).join("");

      // 填充目標 y 下拉
      targetSelect.innerHTML = "";
      parsed.headers.forEach(h => {
        const isNum = analysis.columnStats[h] ? analysis.columnStats[h].isNumeric : false;
        const opt = document.createElement("option");
        opt.value = h;
        opt.textContent = `${h} ${isNum ? '(數值型)' : '(類別型-不可為y)'}`;
        if (!isNum) opt.disabled = true;
        targetSelect.appendChild(opt);
      });

      // 智慧預設目標欄位
      const candTargets = ["Profit", "profit", "medv", "MEDV", "target", "price", "Price"];
      let defTarget = candTargets.find(t => parsed.headers.includes(t));
      if (!defTarget) {
        // 預設倒數第一個數值欄位
        for (let j = parsed.headers.length - 1; j >= 0; j--) {
          const col = parsed.headers[j];
          if (analysis.columnStats[col] && analysis.columnStats[col].isNumeric) {
            defTarget = col;
            break;
          }
        }
      }
      if (defTarget) targetSelect.value = defTarget;

      // 渲染特徵 X 勾選清單
      const renderFeaturesCheckboxes = () => {
        const currTarget = targetSelect.value;
        featuresContainer.innerHTML = "";

        parsed.headers.forEach(h => {
          if (h === currTarget) return; // y 不得作為特徵

          const lbl = document.createElement("label");
          lbl.style.display = "flex";
          lbl.style.alignItems = "center";
          lbl.style.gap = "6px";
          lbl.style.cursor = "pointer";
          lbl.style.fontSize = "0.82rem";
          lbl.style.padding = "4px 8px";
          lbl.style.borderRadius = "4px";
          lbl.style.background = "rgba(255, 255, 255, 0.05)";

          const chk = document.createElement("input");
          chk.type = "checkbox";
          chk.value = h;
          chk.checked = true; // 預設全選
          chk.className = "custom-feat-chk";

          const stat = analysis.columnStats[h];
          const typeBadge = stat && stat.isNumeric ? "" : " (One-Hot)";

          lbl.appendChild(chk);
          lbl.appendChild(document.createTextNode(`${h}${typeBadge}`));
          featuresContainer.appendChild(lbl);
        });
      };

      renderFeaturesCheckboxes();
      targetSelect.onchange = renderFeaturesCheckboxes;

      setStatus("success", `成功讀取 ${sourceName}！共 ${parsed.rows.length} 筆資料、${parsed.headers.length} 個欄位。請檢閱預覽並設定回歸變數。`);
    } catch (err) {
      if (configPanel) configPanel.style.display = "none";
      if (resultsPanel) resultsPanel.style.display = "none";
      setStatus("error", `讀取或解析失敗：${err.message}`);
    }
  };

  // 1. 網址 Fetch
  if (btnFetchUrl && urlInput) {
    btnFetchUrl.onclick = async () => {
      const url = urlInput.value.trim();
      if (!url) {
        alert("請輸入有效的 CSV 檔案公開網址！");
        return;
      }

      setStatus("loading", `正在從 ${url} 下載資料...`);
      try {
        const resp = await fetch(url, { mode: "cors" });
        if (!resp.ok) throw new Error(`HTTP 狀態碼錯誤: ${resp.status}`);
        const text = await resp.text();
        processRawCSV(text, url);
      } catch (e) {
        setStatus("error", `無法取得遠端檔案：${e.message}。若為跨網域問題 (CORS)，請改用「上傳本機 CSV」進行無跨域限制直接分析！`);
      }
    };
  }

  // 2. 本機檔案上傳
  if (fileInput) {
    fileInput.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;

      setStatus("loading", `正在讀取本機檔案：${file.name}...`);
      const reader = new FileReader();
      reader.onload = (evt) => {
        processRawCSV(evt.target.result, `本機檔案 [${file.name}]`);
      };
      reader.onerror = () => {
        setStatus("error", `讀取本機檔案失敗！`);
      };
      reader.readAsText(file);
    };
  }

  // 3. 範例預設按鈕
  document.querySelectorAll(".btn-sm-preset").forEach(btn => {
    btn.onclick = () => {
      const preset = btn.dataset.preset;
      if (preset === "50startups") {
        urlInput.value = "https://raw.githubusercontent.com/harimittapalli/Mulitple-Linear-Reggression/master/50_Startups.csv";
        btnFetchUrl.click();
      } else if (preset === "boston") {
        urlInput.value = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv";
        btnFetchUrl.click();
      } else if (preset === "small") {
        // 內建 20 筆多特徵合成測試 CSV
        const smallCSV = `YearsExperience,TestScore,EducationLevel,Salary
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
6.0,95,PhD,93940`;
        processRawCSV(smallCSV, "小樣本多特徵測試資料 (20 筆)");
      }
    };
  });

  // 4. 特徵全選 / 清除
  if (btnSelectAll) {
    btnSelectAll.onclick = () => {
      document.querySelectorAll(".custom-feat-chk").forEach(c => c.checked = true);
    };
  }
  if (btnClearAll) {
    btnClearAll.onclick = () => {
      document.querySelectorAll(".custom-feat-chk").forEach(c => c.checked = false);
    };
  }

  // 5. 執行真實線性回歸擬合
  if (btnRunTraining) {
    btnRunTraining.onclick = () => {
      if (!currentCustomParsed) {
        alert("請先載入 CSV 資料集！");
        return;
      }

      const targetCol = targetSelect.value;
      const selectedFeatures = Array.from(document.querySelectorAll(".custom-feat-chk:checked")).map(c => c.value);

      if (selectedFeatures.length === 0) {
        alert("請至少勾選一個特徵變數 (X)！");
        return;
      }

      const testSizeVal = parseFloat(document.getElementById("custom-test-size").value) || 0.2;
      const randomStateVal = parseInt(document.getElementById("custom-random-state").value, 10) || 0;

      setStatus("loading", "正在於純前端環境求解 OLS 多元線性回歸模型...");

      try {
        const trainResult = window.RegressionEngine.fitAndEvaluate({
          headers: currentCustomParsed.headers,
          rows: currentCustomParsed.rows,
          targetCol,
          featureCols: selectedFeatures,
          testSize: testSizeVal,
          randomState: randomStateVal
        });

        currentCustomResult = trainResult;

        // 顯示結果面板
        if (resultsPanel) resultsPanel.style.display = "block";

        const tm = trainResult.metrics.test;
        const trm = trainResult.metrics.train;
        const ds = trainResult.datasetInfo;

        document.getElementById("custom-test-r2").textContent = `${tm.r2.toFixed(4)} (${(tm.r2 * 100).toFixed(1)}%)`;
        document.getElementById("custom-train-r2").textContent = `訓練集 R²: ${trm.r2.toFixed(4)}`;

        document.getElementById("custom-test-rmse").textContent = tm.rmse.toFixed(4);
        document.getElementById("custom-train-rmse").textContent = `訓練集 RMSE: ${trm.rmse.toFixed(4)}`;

        document.getElementById("custom-test-mae").textContent = tm.mae.toFixed(4);
        document.getElementById("custom-sample-split").textContent = `${ds.trainRows} 訓 / ${ds.testRows} 測`;
        document.getElementById("custom-adj-r2").textContent = `校正 R²: ${tm.adjR2.toFixed(4)}`;

        // 方程式
        document.getElementById("custom-formula-text").textContent = trainResult.model.formula;

        // 圖表繪製
        drawCustomRegressionChart(currentCustomChart);

        setStatus("success", `模型訓練成功！已計算真實測試集指標 (R²: ${tm.r2.toFixed(4)}, RMSE: ${tm.rmse.toFixed(4)})，未採用任何寫死常數。`);
      } catch (err) {
        if (resultsPanel) resultsPanel.style.display = "none";
        setStatus("error", `回歸訓練失敗：${err.message}`);
      }
    };
  }

  // 6. 圖表切換監聽
  document.querySelectorAll("#custom-chart-toggle .segmented-btn").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll("#custom-chart-toggle .segmented-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentCustomChart = btn.dataset.chart;
      drawCustomRegressionChart(currentCustomChart);
    };
  });

  // 7. 下載預測 CSV 與 JSON
  const btnDownCsv = document.getElementById("btn-download-custom-csv");
  const btnDownJson = document.getElementById("btn-download-custom-json");

  if (btnDownCsv) {
    btnDownCsv.onclick = () => {
      if (!currentCustomResult || !currentCustomResult.testPredictions) {
        alert("尚無可供下載的預測結果！");
        return;
      }
      const preds = currentCustomResult.testPredictions;
      let csvContent = "data:text/csv;charset=utf-8,original_row_id,actual,predicted,residual,abs_error\n";
      preds.forEach(p => {
        csvContent += `${p.originalId},${p.actual},${p.predicted},${p.residual},${p.absError}\n`;
      });
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "custom_regression_predictions.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
  }

  if (btnDownJson) {
    btnDownJson.onclick = () => {
      if (!currentCustomResult) {
        alert("尚無可供下載的分析報告！");
        return;
      }
      const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentCustomResult, null, 2));
      const link = document.createElement("a");
      link.setAttribute("href", jsonStr);
      link.setAttribute("download", "custom_regression_report.json");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
  }
}

// 繪製自訂回歸的 Actual vs. Predicted 散佈圖與殘差圖
function drawCustomRegressionChart(mode = "actual_vs_pred") {
  const canvas = document.getElementById("chart-custom-regression");
  if (!canvas || !currentCustomResult || !currentCustomResult.testPredictions) return;

  const preds = currentCustomResult.testPredictions;
  const ctx = canvas.getContext("2d");

  if (chartCustomInst) {
    chartCustomInst.destroy();
    chartCustomInst = null;
  }

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.08)";
  const textColor = isDark ? "#94a3b8" : "#475569";

  if (mode === "actual_vs_pred") {
    const scatterData = preds.map(p => ({ x: p.actual, y: p.predicted }));
    const allVals = preds.flatMap(p => [p.actual, p.predicted]);
    const minVal = Math.min(...allVals) * 0.95;
    const maxVal = Math.max(...allVals) * 1.05;

    chartCustomInst = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: `測試集樣本 (${preds.length} 筆)`,
            data: scatterData,
            backgroundColor: "rgba(6, 182, 212, 0.8)",
            borderColor: "rgba(6, 182, 212, 1)",
            borderWidth: 1.5,
            pointRadius: 5
          },
          {
            type: "line",
            label: "理想擬合線 (y = x)",
            data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
            borderColor: "#ef4444",
            borderDash: [6, 6],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor } }
        },
        scales: {
          x: {
            title: { display: true, text: "實際目標值 Actual y", color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor }
          },
          y: {
            title: { display: true, text: "預測目標值 Predicted ŷ", color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor }
          }
        }
      }
    });
  } else {
    // 殘差圖
    const residualData = preds.map(p => ({ x: p.predicted, y: p.residual }));

    chartCustomInst = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "測試集殘差 (Residuals)",
            data: residualData,
            backgroundColor: "rgba(244, 63, 94, 0.8)",
            borderColor: "rgba(244, 63, 94, 1)",
            borderWidth: 1.5,
            pointRadius: 5
          },
          {
            type: "line",
            label: "零誤差基準線 (Zero Residual)",
            data: [
              { x: Math.min(...preds.map(p => p.predicted)), y: 0 },
              { x: Math.max(...preds.map(p => p.predicted)), y: 0 }
            ],
            borderColor: "#94a3b8",
            borderDash: [4, 4],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor } }
        },
        scales: {
          x: {
            title: { display: true, text: "預測值 Predicted ŷ", color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor }
          },
          y: {
            title: { display: true, text: "殘差 e = y - ŷ", color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor }
          }
        }
      }
    });
  }
}

// ====================================================================
// 7. MASTER PROBLEM VIEW SWITCHER (支援 part1, part2, boston, custom, all)
// ====================================================================
function switchProblemView(target) {
  currentProblemView = target;
  const p1 = document.getElementById("part1");
  const p2 = document.getElementById("part2");
  const boston = document.getElementById("boston-section");
  const custom = document.getElementById("custom-dataset-section");

  // Sync dropdown
  const dropdown = document.getElementById("problem-dropdown");
  if (dropdown && dropdown.value !== target) {
    dropdown.value = target;
  }

  // Sync segmented buttons
  document.querySelectorAll("#problem-switcher .segmented-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.target === target);
  });

  const hideAll = () => {
    if (p1) p1.style.display = "none";
    if (p2) p2.style.display = "none";
    if (boston) boston.style.display = "none";
    if (custom) custom.style.display = "none";
  };

  if (target === "part1") {
    hideAll();
    if (p1) p1.style.display = "block";
    p1.scrollIntoView({ behavior: "smooth", block: "start" });
    setTimeout(() => { drawPart1Chart(currentP1Chart); }, 150);
  } else if (target === "part2") {
    hideAll();
    if (p2) p2.style.display = "block";
    p2.scrollIntoView({ behavior: "smooth", block: "start" });
    setTimeout(() => { renderPart2Charts(); }, 150);
  } else if (target === "boston") {
    hideAll();
    if (boston) boston.style.display = "block";
    boston.scrollIntoView({ behavior: "smooth", block: "start" });
    setTimeout(() => { drawBostonChart(currentBostonChart); }, 150);
  } else if (target === "custom") {
    hideAll();
    if (custom) custom.style.display = "block";
    custom.scrollIntoView({ behavior: "smooth", block: "start" });
    setTimeout(() => { if (currentCustomResult) drawCustomRegressionChart(currentCustomChart); }, 150);
  } else {
    // All-in-One: 全部顯示
    if (p1) p1.style.display = "block";
    if (p2) p2.style.display = "block";
    if (boston) boston.style.display = "block";
    if (custom) custom.style.display = "block";
    p1.scrollIntoView({ behavior: "smooth", block: "start" });
    setTimeout(() => {
      drawPart1Chart(currentP1Chart);
      renderPart2Charts();
      drawBostonChart(currentBostonChart);
      if (currentCustomResult) drawCustomRegressionChart(currentCustomChart);
    }, 150);
  }
}


// 8. Setup All Global Event Listeners
function setupEventListeners() {
  // Problem Dropdown Switcher (加分功能 1)
  const problemDropdown = document.getElementById("problem-dropdown");
  if (problemDropdown) {
    problemDropdown.onchange = (e) => {
      switchProblemView(e.target.value);
    };
  }

  // Segmented Control Switcher (Problem 1 / Problem 2 / All)
  document.querySelectorAll("#problem-switcher .segmented-btn").forEach(btn => {
    btn.onclick = () => {
      const target = btn.dataset.target;
      switchProblemView(target);
    };
  });

  // Algorithm Nav Switcher (Part 2)
  document.querySelectorAll("#algo-tabs .algo-pill").forEach(pill => {
    pill.onclick = () => {
      document.querySelectorAll("#algo-tabs .algo-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      currentAlgo = pill.dataset.algo;

      // Update K labels if needed
      const k3Label = document.getElementById("k3-label");
      if (k3Label) {
        if (currentAlgo.includes("Lecture") || currentAlgo === "RFE" || currentAlgo === "ElasticNet") {
          k3Label.textContent = "+Administration";
        } else {
          k3Label.textContent = "+State_New York";
        }
      }

      renderSummaryTable();
      updateKPIsForCurrentSelection();
      renderPart2Charts();
    };
  });

  // K-Step Selector Buttons (Part 2)
  document.querySelectorAll("#k-steps-selector .k-step-btn").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll("#k-steps-selector .k-step-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentK = parseInt(btn.dataset.k);
      updateKPIsForCurrentSelection();
      renderPart2Charts();
    };
  });

  // Part 2 Toggle View: Interactive vs Python Matplotlib Static
  const btnInteractive = document.getElementById("btn-view-interactive");
  const btnPython = document.getElementById("btn-view-python-p2");
  const interactiveCharts = document.getElementById("p2-interactive-charts");
  const pythonStatic = document.getElementById("p2-python-static-container");

  if (btnInteractive && btnPython) {
    btnInteractive.onclick = () => {
      btnInteractive.classList.add("active");
      btnPython.classList.remove("active");
      if (interactiveCharts) interactiveCharts.style.display = "grid";
      if (pythonStatic) pythonStatic.style.display = "none";
    };

    btnPython.onclick = () => {
      btnPython.classList.add("active");
      btnInteractive.classList.remove("active");
      if (interactiveCharts) interactiveCharts.style.display = "none";
      if (pythonStatic) pythonStatic.style.display = "block";
    };
  }

  // Part 1 Chart Mode Switcher
  document.querySelectorAll("#p1-chart-toggle-group .view-toggle-btn").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll("#p1-chart-toggle-group .view-toggle-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentP1Chart = btn.dataset.chart;
      drawPart1Chart(currentP1Chart);
    };
  });
}
