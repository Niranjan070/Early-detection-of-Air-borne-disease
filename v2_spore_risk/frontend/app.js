const state = {
  dashboard: null,
  filter: 'all',
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'n/a'
  }
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

function riskClass(value) {
  return `risk-pill risk-${String(value || 'unknown').toLowerCase()}`
}

function fileClass(present) {
  return present ? 'file-chip is-ready' : 'file-chip is-missing'
}

function clearChildren(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild)
  }
}

function renderSummary(summary) {
  const container = document.getElementById('summaryGrid')
  clearChildren(container)

  const cards = [
    ['Samples', formatNumber(summary.total_samples)],
    ['Average spores', formatNumber(summary.avg_total_count)],
    ['Max spores', formatNumber(summary.max_total_count)],
    ['Model accuracy', summary.accuracy == null ? 'n/a' : `${formatNumber(summary.accuracy * 100, 1)}%`],
  ]

  cards.forEach(([label, value]) => {
    const card = document.createElement('article')
    card.className = 'summary-card'
    card.innerHTML = `<span class="summary-label">${label}</span><strong class="summary-value">${value}</strong>`
    container.appendChild(card)
  })
}

function renderRiskBreakdown(summary) {
  const container = document.getElementById('riskBreakdown')
  clearChildren(container)

  const entries = Object.entries(summary.risk_breakdown || {})
  if (!entries.length) {
    container.innerHTML = '<p class="muted">No predictions found yet.</p>'
    return
  }

  const total = entries.reduce((sum, [, value]) => sum + Number(value || 0), 0)
  entries.forEach(([risk, count]) => {
    const percentage = total ? (Number(count) / total) * 100 : 0
    const row = document.createElement('div')
    row.className = 'stack-row'
    row.innerHTML = `
      <div class="stack-labels">
        <span class="${riskClass(risk)}">${risk}</span>
        <strong>${count} samples</strong>
      </div>
      <div class="stack-track">
        <div class="stack-fill stack-${risk}" style="width:${percentage}%"></div>
      </div>
      <span class="stack-value">${formatNumber(percentage, 1)}%</span>
    `
    container.appendChild(row)
  })
}

function renderFiles(files) {
  const container = document.getElementById('fileStatus')
  clearChildren(container)

  Object.entries(files || {}).forEach(([name, present]) => {
    const item = document.createElement('div')
    item.className = fileClass(Boolean(present))
    item.innerHTML = `
      <span class="file-name">${name}</span>
      <strong>${present ? 'ready' : 'missing'}</strong>
    `
    container.appendChild(item)
  })
}

function renderFeatures(metrics) {
  const container = document.getElementById('featureBars')
  clearChildren(container)

  const items = metrics?.top_feature_importance || []
  if (!items.length) {
    container.innerHTML = '<p class="muted">Train the model to populate feature importance.</p>'
    return
  }

  const maxImportance = Math.max(...items.map((item) => Number(item.importance || 0)), 0)
  items.slice(0, 8).forEach((item) => {
    const width = maxImportance ? (Number(item.importance || 0) / maxImportance) * 100 : 0
    const row = document.createElement('div')
    row.className = 'feature-row'
    row.innerHTML = `
      <div class="feature-meta">
        <span class="feature-name">${item.feature}</span>
        <span class="feature-score">${formatNumber(item.importance, 3)}</span>
      </div>
      <div class="feature-track">
        <div class="feature-fill" style="width:${width}%"></div>
      </div>
    `
    container.appendChild(row)
  })
}

function renderSamples(samples) {
  const tbody = document.getElementById('sampleRows')
  clearChildren(tbody)

  const filtered = (samples || []).filter((sample) => {
    if (state.filter === 'all') {
      return true
    }
    return String(sample.predicted_risk || '').toLowerCase() === state.filter
  })

  if (!filtered.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-cell">No samples match this filter.</td></tr>'
    return
  }

  filtered.forEach((sample) => {
    const row = document.createElement('tr')
    row.innerHTML = `
      <td>
        <div class="sample-name">${sample.sample_id}</div>
        <div class="sample-meta">${sample.image_name || 'n/a'}</div>
      </td>
      <td>${formatNumber(sample.total_count)}</td>
      <td>${formatNumber(sample.mean_confidence, 3)}</td>
      <td><span class="${riskClass(sample.predicted_risk)}">${sample.predicted_risk || 'n/a'}</span></td>
      <td><span class="${riskClass(sample.blast_risk_label)}">${sample.blast_risk_label || 'n/a'}</span></td>
      <td>${formatNumber(sample.probability__high, 3)}</td>
      <td>${formatNumber(sample.probability__medium, 3)}</td>
      <td>${formatNumber(sample.probability__low, 3)}</td>
    `
    tbody.appendChild(row)
  })
}

function renderDashboard(payload) {
  state.dashboard = payload
  renderSummary(payload.summary || {})
  renderRiskBreakdown(payload.summary || {})
  renderFiles(payload.files || {})
  renderFeatures(payload.metrics || {})
  renderSamples(payload.samples || [])
}

async function loadDashboard() {
  const refreshButton = document.getElementById('refreshButton')
  refreshButton.disabled = true
  refreshButton.textContent = 'Refreshing...'

  try {
    const response = await fetch('/api/v2/dashboard')
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`)
    }
    const payload = await response.json()
    renderDashboard(payload)
  } catch (error) {
    const main = document.querySelector('.dashboard')
    const template = document.getElementById('emptyStateTemplate')
    main.innerHTML = ''
    main.appendChild(template.content.cloneNode(true))
    console.error(error)
  } finally {
    refreshButton.disabled = false
    refreshButton.textContent = 'Refresh Data'
  }
}

document.getElementById('refreshButton').addEventListener('click', loadDashboard)
document.getElementById('riskFilter').addEventListener('change', (event) => {
  state.filter = event.target.value
  renderSamples(state.dashboard?.samples || [])
})

loadDashboard()
