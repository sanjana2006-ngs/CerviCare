const featureList = document.querySelector("#feature-list");
const metricElements = document.querySelectorAll("[data-metric]");
const summaryElements = document.querySelectorAll("[data-summary]");
const metricsError = document.querySelector("#metrics-error");

async function getJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Project data could not be loaded.");
  return data;
}

async function renderFeatures() {
  if (!featureList) return;
  try {
    const data = await getJson("/api/features");
    if (!Array.isArray(data.features) || data.features.length === 0) {
      throw new Error("Feature rankings are unavailable.");
    }
    featureList.replaceChildren();
    const maximum = Math.max(...data.features.map((feature) => feature.importance), 0);

    data.features.forEach((feature) => {
      const row = document.createElement("div");
      row.className = "feature-row";
      const rank = document.createElement("span");
      rank.className = "feature-rank";
      rank.textContent = String(feature.rank).padStart(2, "0");
      const info = document.createElement("div");
      info.className = "feature-bar-info";
      const label = document.createElement("span");
      label.className = "feature-bar-label";
      label.textContent = feature.label;
      const track = document.createElement("div");
      track.className = "feature-bar-track";
      track.setAttribute("aria-label", `${feature.label}: ${(feature.importance * 100).toFixed(2)} percent importance`);
      const fill = document.createElement("span");
      fill.className = "feature-bar-fill";
      fill.style.width = `${maximum > 0 ? (feature.importance / maximum) * 100 : 0}%`;
      track.append(fill);
      info.append(label, track);
      const value = document.createElement("span");
      value.className = "feature-value";
      value.textContent = `${(feature.importance * 100).toFixed(2)}%`;
      row.append(rank, info, value);
      featureList.append(row);
    });
  } catch (error) {
    featureList.replaceChildren();
    const message = document.createElement("p");
    message.className = "form-error";
    message.setAttribute("role", "alert");
    message.textContent = error instanceof TypeError
      ? "Could not reach CerviCare. Check that the Flask application is running."
      : error.message;
    featureList.append(message);
  }
}

function renderComparison(rows) {
  const container = document.querySelector("#comparison-table");
  if (!container || !Array.isArray(rows)) return;
  const table = document.createElement("table");
  const headings = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg. precision"];
  const head = document.createElement("thead");
  const headingRow = document.createElement("tr");
  headings.forEach((heading) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = heading;
    headingRow.append(cell);
  });
  head.append(headingRow);

  const body = document.createElement("tbody");
  rows.forEach((row) => {
    const tableRow = document.createElement("tr");
    [
      row.model,
      row.accuracy,
      row.precision,
      row.recall,
      row.f1_score,
      row.roc_auc,
      row.average_precision,
    ].forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = index === 0
        ? value
        : typeof value === "number"
          ? value.toFixed(3)
          : "N/A";
      tableRow.append(cell);
    });
    body.append(tableRow);
  });
  table.append(head, body);
  container.replaceChildren(table);
}

async function renderMetrics() {
  if (!metricElements.length) return;
  try {
    const data = await getJson("/api/metrics");
    metricElements.forEach((element) => {
      const value = data[element.dataset.metric];
      element.textContent = typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "--";
    });

    const metadata = data.metadata || {};
    const distribution = metadata.class_distribution || {};
    const summary = {
      records: metadata.records,
      original_features: metadata.original_features,
      positive: distribution.positive,
      negative: distribution.negative,
    };
    summaryElements.forEach((element) => {
      const value = summary[element.dataset.summary];
      element.textContent = Number.isInteger(value) ? value.toLocaleString() : "--";
    });

    const rocAuc = document.querySelector("#roc-auc");
    if (rocAuc) {
      rocAuc.textContent = typeof data.roc_auc === "number"
        ? `ROC-AUC: ${data.roc_auc.toFixed(3)}`
        : "ROC-AUC unavailable.";
    }
    const averagePrecision = document.querySelector("#average-precision");
    if (averagePrecision) {
      averagePrecision.textContent = typeof data.average_precision === "number"
        ? `Average precision: ${data.average_precision.toFixed(3)}`
        : "Average precision unavailable.";
    }

    const cvMean = document.querySelector("#cv-mean");
    const cvStd = document.querySelector("#cv-std");
    if (cvMean && data.cross_validation) {
      cvMean.textContent = data.cross_validation.mean.toFixed(3);
      cvStd.textContent = data.cross_validation.standard_deviation.toFixed(3);
    }
    renderComparison(data.model_comparison);

    const list = document.querySelector("#limitations-list");
    if (list && Array.isArray(data.limitations)) {
      list.replaceChildren();
      data.limitations.forEach((limitation) => {
        const item = document.createElement("li");
        item.textContent = limitation;
        list.append(item);
      });
    }
  } catch (error) {
    if (metricsError) {
      metricsError.textContent = error instanceof TypeError
        ? "Could not reach CerviCare. Check that the Flask application is running."
        : error.message;
      metricsError.hidden = false;
    }
  }
}

renderFeatures();
renderMetrics();
