import React, { useEffect, useMemo, useState } from 'react'

function todayISO() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function formatDate(iso) {
  try {
    const d = new Date(iso + 'T00:00:00')
    return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
  } catch { return iso }
}

/* ───── Small components ───── */

function RiskBadge({ level }) {
  const l = (level || 'low').toLowerCase()
  const icons = { low: '✅', medium: '⚠️', high: '🔴', critical: '🚨' }
  const labels = { low: 'Safe', medium: 'Moderate', high: 'High Risk', critical: 'Critical' }
  return <span className={`risk-badge ${l}`}>{icons[l] || '❓'} {labels[l] || level}</span>
}

function StatCard({ icon, label, value, unit }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div className="stat-value">{value ?? '—'}{unit ? <span className="stat-unit"> {unit}</span> : null}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

function HistoryRow({ sample }) {
  const risk = (sample.overall_risk || 'low').toLowerCase()
  const icons = { low: '✅', medium: '⚠️', high: '🔴', critical: '🚨' }
  return (
    <div className="history-row">
      <span className="history-date">{formatDate(sample.date)}</span>
      <span className="history-crop">{sample.crop_type || '—'}</span>
      <span className="history-spores">{sample.total_spores ?? '—'} spores</span>
      <span className={`history-risk ${risk}`}>{icons[risk] || '❓'} {risk}</span>
    </div>
  )
}

/* ───── Main App ───── */

export default function App() {
  const [farmerId, setFarmerId] = useState(() => localStorage.getItem('farmerId') || '')
  const [farmerName, setFarmerName] = useState(() => localStorage.getItem('farmerName') || '')
  const [loggedIn, setLoggedIn] = useState(() => !!localStorage.getItem('farmerId'))
  const [cropType, setCropType] = useState('rice')
  const [exposureHours, setExposureHours] = useState(24)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [activeTab, setActiveTab] = useState('upload') // upload | results | history

  const date = useMemo(() => todayISO(), [])

  /* persist farmer id */
  useEffect(() => {
    if (farmerId) localStorage.setItem('farmerId', farmerId)
    if (farmerName) localStorage.setItem('farmerName', farmerName)
  }, [farmerId, farmerName])

  /* fetch today's result on load */
  useEffect(() => {
    if (!loggedIn || !farmerId) return
    ;(async () => {
      try {
        const res = await fetch(`/api/samples/today?farmer_id=${encodeURIComponent(farmerId)}&date=${encodeURIComponent(date)}`)
        if (!res.ok) return
        const data = await res.json()
        if (data?.success) { setResult(data); setActiveTab('results') }
      } catch { /* ignore */ }
    })()
  }, [loggedIn, farmerId, date])

  /* fetch history */
  useEffect(() => {
    if (!loggedIn || !farmerId) return
    ;(async () => {
      try {
        const res = await fetch(`/api/samples/history?farmer_id=${encodeURIComponent(farmerId)}&limit=10`)
        if (!res.ok) return
        const data = await res.json()
        if (data?.success) setHistory(data.samples || [])
      } catch { /* ignore */ }
    })()
  }, [loggedIn, farmerId, result])

  /* handle file preview */
  function onFileChange(e) {
    const f = e.target.files?.[0] || null
    setFile(f)
    if (f) {
      const url = URL.createObjectURL(f)
      setPreview(url)
    } else {
      setPreview(null)
    }
  }

  /* login */
  function onLogin(e) {
    e.preventDefault()
    if (!farmerId.trim()) { setError('Please enter your name or ID'); return }
    setLoggedIn(true)
    setError('')
  }

  /* logout */
  function onLogout() {
    localStorage.removeItem('farmerId')
    localStorage.removeItem('farmerName')
    setFarmerId('')
    setFarmerName('')
    setLoggedIn(false)
    setResult(null)
    setHistory([])
    setActiveTab('upload')
  }

  /* submit sample */
  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    setSuccess('')

    if (!file) { setError('📷 Please select a spore trap image first.'); return }

    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('farmer_id', farmerId.trim())
      form.append('date', date)
      form.append('crop_type', cropType)
      form.append('exposure_hours', String(exposureHours))

      const res = await fetch('/api/samples', { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok || !data?.success) {
        setError(data?.detail || 'Something went wrong. Please try again.')
      } else {
        setResult(data)
        setSuccess('✅ Analysis complete! Check your results below.')
        setActiveTab('results')
        setFile(null)
        setPreview(null)
      }
    } catch (err) {
      setError('Could not connect to server. Please check your internet.')
    } finally {
      setLoading(false)
    }
  }

  const overallRisk = result?.risk_analysis?.overall_risk || 'low'
  const totalSpores = result?.spore_counts?.total ?? null
  const freqPerHour = result?.frequency?.total_per_hour != null
    ? Math.round(result.frequency.total_per_hour * 100) / 100
    : null
  const recommendations = result?.risk_analysis?.recommendations || []

  /* ─── Login screen ─── */
  if (!loggedIn) {
    return (
      <div className="login-screen">
        <div className="login-card">
          <div className="login-icon">🌾</div>
          <h1 className="login-title">Crop Disease Alert</h1>
          <p className="login-subtitle">Protect your crops from disease. Get daily spore trap analysis and risk alerts.</p>

          <form onSubmit={onLogin}>
            <label>Your Name or Farmer ID</label>
            <input
              value={farmerId}
              onChange={(e) => setFarmerId(e.target.value)}
              placeholder="e.g. Ramesh or F-101"
              autoFocus
            />
            <label style={{ marginTop: 12 }}>Your Name (optional)</label>
            <input
              value={farmerName}
              onChange={(e) => setFarmerName(e.target.value)}
              placeholder="e.g. Ramesh Kumar"
            />
            {error && <div className="error-msg">{error}</div>}
            <button type="submit" className="btn-primary btn-large">
              🚜 Start Using
            </button>
          </form>

          <div className="login-footer">Free to use &bull; No sign-up required</div>
        </div>
      </div>
    )
  }

  /* ─── Main dashboard ─── */
  return (
    <div className="app-wrapper">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-left">
          <span className="header-logo">🌾</span>
          <div>
            <div className="header-title">Crop Disease Alert</div>
            <div className="header-date">📅 {formatDate(date)}</div>
          </div>
        </div>
        <div className="header-right">
          <span className="header-user">👤 {farmerName || farmerId}</span>
          <button className="btn-outline btn-sm" onClick={onLogout}>Logout</button>
        </div>
      </header>

      {/* ── Tab navigation ── */}
      <nav className="tab-bar">
        <button className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>
          📷 New Check
        </button>
        <button className={`tab-btn ${activeTab === 'results' ? 'active' : ''}`} onClick={() => setActiveTab('results')}>
          📊 Results
        </button>
        <button className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
          📅 History
        </button>
      </nav>

      <main className="main-content">

        {/* ────────── UPLOAD TAB ────────── */}
        {activeTab === 'upload' && (
          <div className="tab-panel fade-in">
            <div className="section-card">
              <h2 className="section-title">📷 Upload Spore Trap Image</h2>
              <p className="section-desc">Take a photo of your spore trap slide under the microscope and upload it here.</p>

              <form onSubmit={onSubmit}>
                <div className="form-grid">
                  <div className="form-group">
                    <label>🌱 Crop Type</label>
                    <select value={cropType} onChange={(e) => setCropType(e.target.value)}>
                      <option value="rice">🍚 Rice</option>
                      <option value="wheat">🌾 Wheat</option>
                      <option value="barley">🌿 Barley</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>⏱️ Trap Exposure Time</label>
                    <div className="input-with-unit">
                      <input
                        type="number"
                        min="1"
                        max="72"
                        value={exposureHours}
                        onChange={(e) => setExposureHours(Number(e.target.value))}
                      />
                      <span className="input-unit">hours</span>
                    </div>
                  </div>
                </div>

                {/* File upload area */}
                <div className="upload-area" onClick={() => document.getElementById('file-input').click()}>
                  {preview ? (
                    <div className="preview-container">
                      <img src={preview} alt="Preview" className="preview-img" />
                      <div className="preview-label">✅ Image selected — tap to change</div>
                    </div>
                  ) : (
                    <div className="upload-placeholder">
                      <div className="upload-icon">📷</div>
                      <div className="upload-text">Tap here to select image</div>
                      <div className="upload-hint">Supports JPG, PNG, BMP</div>
                    </div>
                  )}
                  <input
                    id="file-input"
                    type="file"
                    accept="image/*"
                    onChange={onFileChange}
                    style={{ display: 'none' }}
                  />
                </div>

                {error && <div className="error-msg">{error}</div>}
                {success && <div className="success-msg">{success}</div>}

                <button type="submit" className="btn-primary btn-large" disabled={loading}>
                  {loading ? (
                    <span className="loading-text">
                      <span className="spinner"></span> Analyzing your sample…
                    </span>
                  ) : (
                    '🔬 Analyze Now'
                  )}
                </button>
              </form>
            </div>

            {/* Quick tips */}
            <div className="section-card tips-card">
              <h3 className="tips-title">💡 Tips for Best Results</h3>
              <ul className="tips-list">
                <li>Use a clean slide and microscope lens</li>
                <li>Take the photo at 10x or 40x magnification</li>
                <li>Make sure the image is well-lit and focused</li>
                <li>Upload one sample per day for accurate tracking</li>
              </ul>
            </div>
          </div>
        )}

        {/* ────────── RESULTS TAB ────────── */}
        {activeTab === 'results' && (
          <div className="tab-panel fade-in">
            {!result ? (
              <div className="empty-state">
                <div className="empty-icon">📊</div>
                <h3>No Results Yet</h3>
                <p>Upload a spore trap image to see your analysis results here.</p>
                <button className="btn-primary" onClick={() => setActiveTab('upload')}>📷 Upload Now</button>
              </div>
            ) : (
              <>
                {/* Risk banner */}
                <div className={`risk-banner risk-${overallRisk.toLowerCase()}`}>
                  <RiskBadge level={overallRisk} />
                  <div className="risk-banner-text">
                    {overallRisk === 'low' && 'Your crops look healthy today! Keep monitoring.'}
                    {overallRisk === 'medium' && 'Some spores detected. Watch closely over the next few days.'}
                    {overallRisk === 'high' && 'High spore count detected! Consider preventive spraying.'}
                    {overallRisk === 'critical' && 'Critical risk! Take immediate action to protect your crop.'}
                  </div>
                </div>

                {/* Stats */}
                <div className="stats-row">
                  <StatCard icon="🦠" label="Spores Found" value={totalSpores} />
                  <StatCard icon="⏱️" label="Spores per Hour" value={freqPerHour} />
                  <StatCard icon="🕐" label="Exposure Time" value={result?.exposure_hours} unit="hrs" />
                </div>

                {/* Spore breakdown */}
                {result?.spore_counts && Object.keys(result.spore_counts).length > 1 && (
                  <div className="section-card">
                    <h3 className="section-title">🦠 Spore Breakdown</h3>
                    <div className="spore-breakdown">
                      {Object.entries(result.spore_counts)
                        .filter(([k]) => k !== 'total')
                        .map(([name, count]) => (
                          <div key={name} className="spore-row">
                            <span className="spore-name">{name.replace(/_/g, ' ')}</span>
                            <span className="spore-count">{count}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {recommendations.length > 0 && (
                  <div className="section-card recommendations-card">
                    <h3 className="section-title">📋 What You Should Do</h3>
                    <ul className="reco-list">
                      {recommendations.map((r, i) => (
                        <li key={i} className="reco-item">
                          <span className="reco-icon">👉</span>
                          <span>{r}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Disease predictions */}
                {result?.predictions?.predictions?.length > 0 && (
                  <div className="section-card">
                    <h3 className="section-title">🔍 Possible Diseases</h3>
                    <div className="disease-list">
                      {result.predictions.predictions.map((p, i) => (
                        <div key={i} className="disease-card">
                          <div className="disease-name">{p.disease || p.disease_name || 'Unknown'}</div>
                          <div className="disease-risk">
                            Risk: <RiskBadge level={p.risk_level || p.risk} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <button className="btn-primary" onClick={() => setActiveTab('upload')} style={{ marginTop: 16 }}>
                  📷 Upload Another Sample
                </button>
              </>
            )}
          </div>
        )}

        {/* ────────── HISTORY TAB ────────── */}
        {activeTab === 'history' && (
          <div className="tab-panel fade-in">
            <div className="section-card">
              <h2 className="section-title">📅 Your Recent Checks</h2>
              {history.length === 0 ? (
                <div className="empty-state small">
                  <p>No previous checks found. Upload your first sample to start tracking!</p>
                </div>
              ) : (
                <div className="history-list">
                  <div className="history-header">
                    <span>Date</span><span>Crop</span><span>Spores</span><span>Risk</span>
                  </div>
                  {history.map((s, i) => <HistoryRow key={i} sample={s} />)}
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>🌾 Crop Disease Alert System &bull; Helping farmers protect their harvest</p>
      </footer>
    </div>
  )
}
