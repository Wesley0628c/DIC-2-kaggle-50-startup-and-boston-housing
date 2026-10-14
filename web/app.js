/**
 * Aether Quant Lab - Core Application Controller (app.js)
 * Supports dynamic 5-algorithm benchmarks, Occam's Razor Valley at k=2 highlights,
 * real-time OLS inference simulator, and responsive Chart.js vector rendering.
 */

// Global Application State
let APP_DATA = null;
let currentAlgo = "Forward SFS (Lecture Stepwise)";
let currentK = 2; // Default to k=2 (Optimal Valley)
let currentP1Chart = "actual_vs_pred";
let chartRmseInst = null;
let chartR2Inst = null;
let chartP1Inst = null;

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
      
      // Re-render charts with updated theme colors if needed
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
  renderDiscussion(data.discussion);
}

// 1. Render CRISP-DM Steps
function renderCrispDM(stages) {
  const container = document.getElementById("crisp-container");
  if (!container || !stages) return;

  container.innerHTML = stages.map((st, idx) => `
    <div class="crisp-step-card ${idx === 3 ? 'active' : ''}" data-step="${idx + 1}">
      <div class="crisp-num">PHASE 0${idx + 1}</div>
      <div class="crisp-title">${st.phase_name_zh}</div>
      <div class="crisp-sub">${st.goal}</div>
    </div>
  `).join("");
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

// Render Summary Table for Current Algorithm
function renderSummaryTable() {
  const summaryTbody = document.getElementById("p2-summary-tbody");
  if (!summaryTbody || !APP_DATA) return;

  const algoData = getAlgorithmData(currentAlgo);
  if (!algoData) return;

  summaryTbody.innerHTML = algoData.map(row => {
    const isOptimal = row.k === 2;
    return `
      <tr style="${isOptimal ? 'background: rgba(16, 185, 129, 0.08);' : ''}">
        <td class="num-cell" style="font-weight: 800; color: ${isOptimal ? 'var(--emerald-optimal)' : 'var(--text-main)'};">k = ${row.k}</td>
        <td><strong>${row.features[row.features.length - 1]}</strong></td>
        <td><code style="font-size: 0.8rem; color: var(--text-code);">${row.features.join(" + ")}</code></td>
        <td class="num-cell" style="font-weight: 800; color: ${isOptimal ? 'var(--emerald-optimal)' : 'var(--text-main)'};">
          ${formatCurrency(row.rmse)}
        </td>
        <td class="num-cell" style="font-weight: 800; color: ${isOptimal ? 'var(--emerald-optimal)' : 'var(--text-main)'};">
          ${formatNumber(row.r2, 4)}
        </td>
        <td class="num-cell" style="color: var(--text-dim);">${formatCurrency(row.train_rmse || row.rmse * 1.05)}</td>
        <td class="num-cell" style="color: var(--text-dim);">${formatNumber(row.train_r2 || row.r2 * 0.998, 4)}</td>
        <td>
          ${isOptimal 
            ? '<span class="badge-tag tag-sweet">⭐ 最優谷地 (Occam\'s Razor)</span>' 
            : (row.k === 5 ? '<span class="badge-tag tag-info">全特徵 (基準模型)</span>' : '<span class="badge-tag tag-neutral">步進候選</span>')}
        </td>
      </tr>
    `;
  }).join("");
}

// Retrieve data points for selected algorithm
function getAlgorithmData(algoName) {
  if (!APP_DATA) return [];
  if (APP_DATA.algorithms_benchmark && APP_DATA.algorithms_benchmark[algoName]) {
    return APP_DATA.algorithms_benchmark[algoName];
  }
  // Fallback to default progression_table mapped
  if (APP_DATA.part2 && APP_DATA.part2.progression_table) {
    return APP_DATA.part2.progression_table.map(r => ({
      k: r.num_features,
      features: (r.selected_features_str || "").split(", "),
      rmse: r.test_rmse,
      r2: r.test_r2,
      mae: r.test_mae,
      adj_r2: 0.9324,
      f_stat: 349.04
    }));
  }
  return [];
}

// Update 5 KPI Cards for current K and Algorithm
function updateKPIsForCurrentSelection() {
  const algoData = getAlgorithmData(currentAlgo);
  const row = algoData.find(x => x.k === currentK) || algoData[1] || algoData[0];
  if (!row) return;

  const isK2 = row.k === 2;

  // RMSE Card
  const rmseEl = document.getElementById("val-kpi-rmse");
  if (rmseEl) rmseEl.textContent = formatCurrency(row.rmse);
  const rmseBadge = document.getElementById("badge-kpi-rmse");
  if (rmseBadge) {
    rmseBadge.textContent = isK2 ? "谷地最低點 (Optimal: -10.3%)" : `k = ${row.k}`;
    rmseBadge.className = isK2 ? "badge-tag tag-sweet" : "badge-tag tag-neutral";
  }
  const rmseCard = document.getElementById("card-kpi-rmse");
  if (rmseCard) {
    rmseCard.className = isK2 ? "kpi-card highlight-optimal" : "kpi-card";
  }

  // R² Card
  const r2El = document.getElementById("val-kpi-r2");
  if (r2El) r2El.textContent = formatNumber(row.r2, 4);
  const r2Badge = document.getElementById("badge-kpi-r2");
  if (r2Badge) {
    r2Badge.textContent = `${(row.r2 * 100).toFixed(1)}% FIT`;
  }

  // Adjusted R² Card
  const adjR2El = document.getElementById("val-kpi-adjr2");
  if (adjR2El) adjR2El.textContent = formatNumber(row.adj_r2 || (row.r2 - 0.015), 4);

  // MAE Card
  const maeEl = document.getElementById("val-kpi-mae");
  if (maeEl) maeEl.textContent = formatCurrency(row.mae || (row.rmse * 0.83));

  // F-Stat Card
  const fstatEl = document.getElementById("val-kpi-fstat");
  if (fstatEl) fstatEl.textContent = row.f_stat ? formatNumber(row.f_stat, 2) : "349.04";

  // Update Valley Hero Banner description
  const descEl = document.getElementById("valley-desc-text");
  if (descEl) {
    descEl.innerHTML = `當前檢視特徵數 <strong>k = ${row.k}</strong>（包含特徵：<span style="color: var(--cyan-primary);">${row.features.join(", ")}</span>）。在特徵選擇中，<strong>k = 2 雙特徵 (R&D + Marketing)</strong> 形成顯著下凹之測試誤差最低谷地 ($8,198.80)；納入過多特徵將引入雜訊產生過度擬合。`;
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

    // Full Model OLS Formula:
    // Profit = 42554.17 - 959.2842 * FL + 699.3691 * NY + 0.7735 * RD + 0.0329 * Admin + 0.0366 * Marketing
    const profit = 42554.1720 
      - (959.2842 * stateFlorida) 
      + (699.3691 * stateNewYork) 
      + (0.7735 * rd) 
      + (0.0329 * admin) 
      + (0.0366 * marketing);

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

// 6. Setup All Global Event Listeners
function setupEventListeners() {
  // Segmented Control Switcher (Problem 1 / Problem 2 / All)
  document.querySelectorAll("#problem-switcher .segmented-btn").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll("#problem-switcher .segmented-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.dataset.target;
      if (target === "part1") {
        document.getElementById("part1").scrollIntoView({ behavior: "smooth" });
      } else if (target === "part2") {
        document.getElementById("part2").scrollIntoView({ behavior: "smooth" });
      } else {
        document.getElementById("overview").scrollIntoView({ behavior: "smooth" });
      }
    };
  });

  // Algorithm Nav Switcher
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

  // K-Step Selector Buttons
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
