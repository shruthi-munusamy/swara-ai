import { useEffect, useState } from 'react'

const API_BASE = 'http://localhost:8000'

export default function App() {
  const [file, setFile] = useState(null)
  const [shruthiHz, setShruthiHz] = useState(220)
  const [targetRagam, setTargetRagam] = useState('')
  const [ragams, setRagams] = useState([])
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE}/ragams`)
      .then((r) => r.json())
      .then(setRagams)
      .catch(() => setRagams([]))
  }, [])

  async function handleSubmit(e) {
    e.preventDefault()
    if (!file) {
      setError('Please choose an audio file.')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('shruthi_hz', String(shruthiHz))
    if (targetRagam) formData.append('target_ragam', targetRagam)

    try {
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: formData })
      const body = await res.json()
      if (!res.ok) {
        setError(body.detail || 'Request failed.')
      } else {
        setResult(body)
      }
    } catch (err) {
      setError('Could not reach the backend. Is it running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>Swara AI</h1>
      <p className="subtitle">Carnatic raga identification &amp; ragam adherence</p>

      <form onSubmit={handleSubmit} className="form">
        <label>
          Audio clip
          <input
            type="file"
            accept=".wav,.mp3,.flac,.ogg,.m4a"
            onChange={(e) => setFile(e.target.files[0])}
          />
        </label>

        <label>
          Shruthi (Hz)
          <input
            type="number"
            value={shruthiHz}
            min="80"
            max="400"
            onChange={(e) => setShruthiHz(e.target.value)}
          />
        </label>

        <label>
          Target ragam (optional, for adherence check)
          <select value={targetRagam} onChange={(e) => setTargetRagam(e.target.value)}>
            <option value="">— none —</option>
            {ragams.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>

        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="results">
          <h2>Raga identification</h2>
          {result.predictions.length === 0 ? (
            <p>No swara segments detected.</p>
          ) : (
            <ul className="predictions">
              {result.predictions.map((p) => (
                <li key={p.raga}>
                  <span className="raga-name">{p.raga}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${p.confidence * 100}%` }} />
                  </div>
                  <span className="confidence">{(p.confidence * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          )}

          {result.adherence_score !== null && result.adherence_score !== undefined && (
            <div className="adherence">
              <h2>Adherence to {result.adherence_ragam}</h2>
              <p className="score">
                Score: <strong>{(result.adherence_score * 100).toFixed(1)}%</strong>{' '}
                ({result.num_segments - result.num_out_of_raga}/{result.num_segments} segments in raga)
              </p>
              <ul className="segments">
                {result.segments.map((s, i) => (
                  <li key={i} className={s.in_raga ? 'segment in-raga' : 'segment out-of-raga'}>
                    <span>
                      {s.start_time.toFixed(2)}s–{s.end_time.toFixed(2)}s
                    </span>
                    <span>{s.candidates.join('/')}</span>
                    <span>{s.in_raga ? '✓ in raga' : '✗ out of raga'}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
