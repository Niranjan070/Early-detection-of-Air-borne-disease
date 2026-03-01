import React, { useEffect, useMemo, useState } from 'react'

function todayISO() {
  const d = new Date()
  const yyyy = d.getFullYear()
  const mm = String(d.getMonth() + 1).padStart(2, '0')
  const dd = String(d.getDate()).padStart(2, '0')
  return `${yyyy}-${mm}-${dd}`
}

function RiskBadge({ level }) {
  const cls = ['badge', (level || 'low').toLowerCase()].join(' ')
  return <span className={cls}>{(level || 'low').toUpperCase()}</span>
}

export default function App() {
  const [farmerId, setFarmerId] = useState(() => localStorage.getItem('farmerId') || '')
  const [cropType, setCropType] = useState('rice')
  const [exposureHours, setExposureHours] = useState(24)
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const date = useMemo(() => todayISO(), [])

  useEffect(() => {
    if (farmerId) localStorage.setItem('farmerId', farmerId)
  }, [farmerId])

  async function fetchToday() {
    if (!farmerId) return
    setError('')
    try {
      const res = await fetch(
        `/api/samples/today?farmer_id=${encodeURIComponent(farmerId)}&date=${encodeURIComponent(date)}`,
      )
      if (!res.ok) return
      const data = await res.json()
      if (data?.success) setResult(data)
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    fetchToday()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function onSubmit(e) {
    e.preventDefault()
    setError('')

    if (!farmerId.trim()) {
      setError('Farmer ID is required (any name/number for demo).')
      return
    }
    if (!file) {
      setError('Please upload a microscope image.')
      return
    }

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
        setError(data?.detail || 'Upload failed')
      } else {
        setResult(data)
      }
    } catch (err) {
      setError(err?.message || 'Network error')
    } finally {
      setLoading(false)
    }
  }

  function onLogout() {
    localStorage.removeItem('farmerId')
    setFarmerId('')
    setResult(null)
  }

  const overallRisk = result?.risk_analysis?.overall_risk || 'low'
  const totalSpores = result?.spore_counts?.total ?? null
  const freqPerHour = result?.frequency?.total_per_hour ?? null

  return (
    <div className="container">
      <div className="row" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h1>Spore Risk Dashboard</h1>
          <div className="muted">Daily check-in: {date}</div>
        </div>
        <div style={{ maxWidth: 220 }}>
          {farmerId ? (
            <button type="button" className="secondary" onClick={onLogout}>
              Logout
            </button>
          ) : null}
        </div>
      </div>

      <div style={{ height: 12 }} />

      <div className="card">
        <form onSubmit={onSubmit}>
          <div className="row">
            <div>
              <label>Farmer ID (demo login)</label>
              <input
                value={farmerId}
                onChange={(e) => setFarmerId(e.target.value)}
                placeholder="e.g., farmer_01"
              />
            </div>
            <div>
              <label>Crop</label>
              <select value={cropType} onChange={(e) => setCropType(e.target.value)}>
                <option value="rice">Rice</option>
                <option value="wheat">Wheat</option>
                <option value="barley">Barley</option>
              </select>
            </div>
            <div>
              <label>Trap exposure (hours)</label>
              <input
                type="number"
                min="1"
                value={exposureHours}
                onChange={(e) => setExposureHours(Number(e.target.value))}
              />
            </div>
          </div>

          <div style={{ height: 12 }} />

          <div className="row">
            <div>
              <label>Microscope image</label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
              <div className="muted" style={{ marginTop: 6 }}>
                Upload again the same day to update today’s result.
              </div>
            </div>
            <div style={{ maxWidth: 220 }}>
              <label>&nbsp;</label>
              <button disabled={loading} type="submit">
                {loading ? 'Analyzing…' : 'Analyze Today'}
              </button>
            </div>
          </div>

          {error ? <div style={{ marginTop: 12, color: '#b91c1c' }}>{error}</div> : null}
        </form>
      </div>

      <div style={{ height: 12 }} />

      <div className="row">
        <div className="card">
          <div className="muted">Overall risk</div>
          <div style={{ marginTop: 6 }}>
            <RiskBadge level={overallRisk} />
          </div>
          <div style={{ height: 10 }} />
          <div className="muted">Total spores (today)</div>
          <div style={{ fontSize: 22, fontWeight: 700, marginTop: 4 }}>{totalSpores ?? '—'}</div>
          <div style={{ height: 10 }} />
          <div className="muted">Frequency (spores/hour)</div>
          <div style={{ fontSize: 22, fontWeight: 700, marginTop: 4 }}>{freqPerHour ?? '—'}</div>
        </div>

        <div className="card">
          <div className="muted">Action</div>
          <div style={{ marginTop: 8, fontWeight: 600 }}>
            {result?.risk_analysis?.recommendations?.[0] || 'Upload an image to get guidance.'}
          </div>
          <div style={{ height: 10 }} />
          <div className="muted">Notes</div>
          <div style={{ marginTop: 6 }}>
            This demo saves one result per farmer per day.
          </div>
        </div>
      </div>

      <div style={{ height: 12 }} />

      <div className="card">
        <div className="muted">Raw response (debug)</div>
        <div style={{ height: 8 }} />
        <pre>{JSON.stringify(result, null, 2)}</pre>
      </div>
    </div>
  )
}
