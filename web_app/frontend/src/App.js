import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [playerName, setPlayerName] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [useSample, setUseSample] = useState(false);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [sampleFiles, setSampleFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [availablePlayers, setAvailablePlayers] = useState([]);

  const handlePreview = async () => {
    if (!useSample || sampleFiles.length === 0) return;
    
    setLoadingPreview(true);
    try {
      const response = await fetch('/api/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          use_sample: true,
          sample_index: sampleIndex,
          player_name: playerName
        })
      });
      
      const data = await response.json();
      if (data.success) {
        setPreview(data);
        // Update available players for dropdown
        if (data.all_players && data.all_players.length > 0) {
          setAvailablePlayers(data.all_players);
          // Auto-select first player if none selected
          if (!playerName && data.all_players.length > 0) {
            setPlayerName(data.all_players[0]);
          }
        }
      }
    } catch (err) {
      console.error('Preview error:', err);
    } finally {
      setLoadingPreview(false);
    }
  };

  useEffect(() => {
    // Fetch sample files - try direct connection first (more reliable)
    const fetchSamples = async () => {
      const tryFetch = async (url) => {
        try {
          const response = await fetch(url);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          console.log('Sample files response:', data);
          if (data.samples && data.samples.length > 0) {
            setSampleFiles(data.samples);
            console.log(`✓ Loaded ${data.samples.length} sample files from ${url}`);
            return true;
          } else {
            console.warn('No samples in response:', data);
            return false;
          }
        } catch (err) {
          console.error(`Error fetching from ${url}:`, err);
          return false;
        }
      };
      
      // Try direct backend URL first (more reliable)
      const directSuccess = await tryFetch('http://localhost:5001/api/sample-files');
      if (!directSuccess) {
        // Fallback to proxy
        await tryFetch('/api/sample-files');
      }
    };
    fetchSamples();
  }, []);

  // Fetch players when sample is selected
  useEffect(() => {
    const fetchPlayers = async () => {
      if (!useSample || sampleFiles.length === 0) {
        setAvailablePlayers([]);
        setPlayerName('');
        return;
      }
      
      setLoadingPreview(true);
      try {
        // Try to get players directly (more reliable)
        const response = await fetch('/api/players', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            use_sample: true,
            sample_index: sampleIndex
          })
        });
        
        const data = await response.json();
        if (data.success && data.players && data.players.length > 0) {
          setAvailablePlayers(data.players);
          if (!playerName && data.players.length > 0) {
            setPlayerName(data.players[0]);
          }
          console.log(`✓ Loaded ${data.players.length} players`);
        } else {
          console.warn('No players found in response');
        }
      } catch (err) {
        console.error('Error fetching players:', err);
        // Try direct backend URL
        try {
          const response = await fetch('http://localhost:5001/api/players', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              use_sample: true,
              sample_index: sampleIndex
            })
          });
          const data = await response.json();
          if (data.success && data.players) {
            setAvailablePlayers(data.players);
            if (!playerName && data.players.length > 0) {
              setPlayerName(data.players[0]);
            }
          }
        } catch (e) {
          console.error('Fallback also failed:', e);
        }
      } finally {
        setLoadingPreview(false);
      }
    };
    
    fetchPlayers();
    // Also fetch preview for hand info
    if (useSample && sampleFiles.length > 0) {
      handlePreview();
    } else {
      setPreview(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useSample, sampleIndex, sampleFiles.length]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setUseSample(false);
    }
  };

  const handleAnalyze = async () => {
    if (!playerName.trim()) {
      setError('Please enter a player name');
      return;
    }

    if (!useSample && !selectedFile) {
      setError('Please select a file or use a sample file');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      // Prefer direct backend URL (more reliable across environments),
      // fall back to relative path if needed.
      const requestBody = {
        player_name: playerName,
        use_sample: useSample,
        sample_index: sampleIndex,
        file_path: useSample ? null : (selectedFile ? selectedFile.name : null)
      };

      let response;
      try {
        response = await fetch('http://localhost:5001/api/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody)
        });
      } catch (e) {
        console.error('Direct backend analyze request failed, trying proxy /api/analyze:', e);
        response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
          body: JSON.stringify(requestBody)
        });
      }

      const data = await response.json();

      if (data.success) {
        setResults(data.results);
      } else {
        setError(data.error || 'Analysis failed. Please check backend logs.');
      }
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header>
          <h1>🎰 Poker Coach AI</h1>
          <p>Analyze your poker sessions with AI-powered coaching</p>
        </header>

        <div className="card">
          <h2>Session Analysis</h2>
          
          <div className="form-group">
            <label>Player Name:</label>
            {useSample && availablePlayers.length > 0 ? (
              <select
                value={playerName}
                onChange={(e) => setPlayerName(e.target.value)}
                className="player-select"
              >
                <option value="">-- Select a player --</option>
                {availablePlayers.map((player, idx) => (
                  <option key={idx} value={player}>
                    {player}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={playerName}
                onChange={(e) => setPlayerName(e.target.value)}
                placeholder="Enter player name (e.g., Hero)"
              />
            )}
            {useSample && availablePlayers.length === 0 && !loadingPreview && (
              <p style={{fontSize: '0.85rem', color: '#666', marginTop: '5px'}}>
                Select a sample file to see available players
              </p>
            )}
          </div>

          <div className="form-group">
            <label>
              <input
                type="checkbox"
                checked={useSample}
                onChange={(e) => setUseSample(e.target.checked)}
              />
              Use Preloaded Sample File
            </label>
            
            {useSample && (
              <>
                {sampleFiles.length > 0 ? (
                  <select
                    value={sampleIndex}
                    onChange={(e) => setSampleIndex(parseInt(e.target.value))}
                    className="sample-select"
                  >
                    {sampleFiles.map((sample, idx) => (
                      <option key={idx} value={idx}>
                        {sample.name}
                      </option>
                    ))}
                  </select>
                ) : (
                  <div className="error" style={{marginTop: '10px'}}>
                    <p>No sample files available.</p>
                    <p style={{fontSize: '0.9rem', marginTop: '5px'}}>
                      Backend should be running on port 5001.
                      <button 
                        onClick={async () => {
                          try {
                            const response = await fetch('http://localhost:5001/api/sample-files');
                            const data = await response.json();
                            if (data.samples) {
                              setSampleFiles(data.samples);
                              alert(`Loaded ${data.samples.length} sample files!`);
                            }
                          } catch (e) {
                            alert('Could not connect to backend. Make sure it\'s running on port 5001.');
                          }
                        }} 
                        style={{marginLeft: '10px', padding: '5px 10px', cursor: 'pointer', background: '#667eea', color: 'white', border: 'none', borderRadius: '4px'}}
                      >
                        Try Direct Connection
                      </button>
                      <button 
                        onClick={() => window.location.reload()} 
                        style={{marginLeft: '10px', padding: '5px 10px', cursor: 'pointer', background: '#28a745', color: 'white', border: 'none', borderRadius: '4px'}}
                      >
                        Refresh Page
                      </button>
                    </p>
                  </div>
                )}
              </>
            )}
          </div>

          {!useSample && (
            <div className="form-group">
              <label>Upload .phh/.phhs File:</label>
              <input
                type="file"
                accept=".phh,.phhs"
                onChange={handleFileChange}
              />
              {selectedFile && (
                <p className="file-name">Selected: {selectedFile.name}</p>
              )}
            </div>
          )}

          {useSample && preview && (
            <div className="preview-card">
              <h3>File Preview</h3>
              {loadingPreview ? (
                <p>Loading preview...</p>
              ) : (
                <>
                  <p><strong>Total Hands:</strong> {preview.total_hands}</p>
                  {preview.sample_players && preview.sample_players.length > 0 && (
                    <p><strong>Sample Players:</strong> {preview.sample_players.slice(0, 3).join(', ')}...</p>
                  )}
                  {preview.player_found !== null && (
                    <p className={preview.player_found ? 'player-found' : 'player-not-found'}>
                      {preview.player_found ? '✓ Player found in file' : '⚠ Player not found in first 10 hands'}
                    </p>
                  )}
                  <div className="hands-preview">
                    <h4>First 5 Hands:</h4>
                    <table className="preview-table">
                      <thead>
                        <tr>
                          <th>Hand ID</th>
                          <th>Players</th>
                          <th>Actions</th>
                          <th>Date</th>
                        </tr>
                      </thead>
                      <tbody>
                        {preview.preview_hands.map((hand, idx) => (
                          <tr key={idx}>
                            <td>{hand.hand_id}</td>
                            <td>{hand.players.slice(0, 3).join(', ')}{hand.total_players > 3 ? '...' : ''}</td>
                            <td>{hand.actions_count}</td>
                            <td>{hand.date}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="analyze-btn"
          >
            {loading ? 'Analyzing...' : 'Analyze Session'}
          </button>

          {error && <div className="error">{error}</div>}
        </div>

        {results && (
          <div className="card results">
            <h2>Analysis Results</h2>
            
            <div className="metrics">
              <div className="metric player-metric">
                <h3>Player</h3>
                <p className="player-id">{results.player_name}</p>
              </div>
              <div className="metric">
                <h3>Total Hands</h3>
                <p>{results.total_hands}</p>
              </div>
              <div className="metric">
                <h3>Total Decisions</h3>
                <p>{results.total_decisions}</p>
              </div>
              <div className="metric">
                <h3>Agreement Rate</h3>
                <p className="highlight">{results.agreement_rate}%</p>
              </div>
              <div className="metric">
                <h3>High Confidence Mistakes</h3>
                <p className="highlight">{results.high_confidence_mistakes}</p>
              </div>
            </div>

            {results.ai_summary && (
              <div className="ai-summary">
                <h3>AI Coaching Summary</h3>
                <p>{results.ai_summary}</p>
              </div>
            )}

            {results.top_mistakes && results.top_mistakes.length > 0 && (
              <div className="mistakes">
                <h3>Top Mistakes</h3>
                <ul>
                  {results.top_mistakes.map((mistake, idx) => (
                    <li key={idx}>
                      <strong>{mistake[0]}</strong>: {mistake[1]} occurrences
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {results.decisions && results.decisions.length > 0 && (
              <div className="decisions">
                <h3>Sample Decisions (First 10)</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Hand ID</th>
                      <th>Your Action</th>
                      <th>Optimal Action</th>
                      <th>Confidence</th>
                      <th>Match</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.decisions.slice(0, 10).map((decision, idx) => (
                      <tr key={idx}>
                        <td>{decision.hand_id}</td>
                        <td>{decision.player_action}</td>
                        <td>{decision.optimal_action}</td>
                        <td>{(decision.confidence * 100).toFixed(1)}%</td>
                        <td>
                          <span className={decision.agreement ? 'match' : 'mismatch'}>
                            {decision.agreement ? '✓' : '✗'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

