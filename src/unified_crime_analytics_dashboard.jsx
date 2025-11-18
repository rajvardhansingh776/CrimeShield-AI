import React, {useState, useEffect, useRef} from "react";

export default function UnifiedCrimeAnalyticsDashboard(){
  const [tab, setTab] = useState("overview");
  const [files, setFiles] = useState([]);
  const [hotspotsGeoJSON, setHotspotsGeoJSON] = useState(null);
  const [firGeoJSON, setFirGeoJSON] = useState(null);
  const [cdrGeoJSON, setCdrGeoJSON] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [centrality, setCentrality] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const fileInputRef = useRef(null);

  useEffect(()=>{
    const interval = setInterval(()=>{
      if(!hotspotsGeoJSON) return;
      fetch('/api/alerts')
        .then(r=>r.json())
        .then(data=>setAlerts(data || []))
        .catch(()=>{});
    }, 15000);
    return ()=>clearInterval(interval);
  },[hotspotsGeoJSON]);

  async function uploadFiles(e){
    const f = e.target.files;
    const arr = Array.from(f);
    setFiles(arr.map(fi=>({name:fi.name,size:fi.size})));
    const form = new FormData();
    arr.forEach(fi=>form.append('files',fi));
    setLoading(true);
    try{
      await fetch('/api/upload',{method:'POST',body:form});
      setLoading(false);
    }catch(err){
      setLoading(false);
      console.error(err);
    }
  }

  async function runFullPipeline(){
    setLoading(true);
    try{
      const r = await fetch('/api/run_pipeline',{method:'POST'});
      const j = await r.json();
      if(j.hotspots) setHotspotsGeoJSON(j.hotspots);
      if(j.fir) setFirGeoJSON(j.fir);
      if(j.cdr) setCdrGeoJSON(j.cdr);
      if(j.graph) setGraphData(j.graph);
      if(j.centrality) setCentrality(j.centrality);
      if(j.predictions) setPredictions(j.predictions);
    }catch(e){
      console.error(e);
    }
    setLoading(false);
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-semibold">Unified Crime Analytics Dashboard</h1>
            <div className="text-sm text-gray-500">Integrated: FIR · CDR · Demographics · Graphs · Forecasts</div>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={()=>setTab('overview')} className={`px-3 py-2 rounded ${tab==='overview'?'bg-indigo-600 text-white':'bg-gray-100'}`}>Overview</button>
            <button onClick={()=>setTab('ingest')} className={`px-3 py-2 rounded ${tab==='ingest'?'bg-indigo-600 text-white':'bg-gray-100'}`}>Ingest</button>
            <button onClick={()=>setTab('preprocess')} className={`px-3 py-2 rounded ${tab==='preprocess'?'bg-indigo-600 text-white':'bg-gray-100'}`}>Preprocess</button>
            <button onClick={()=>setTab('hotspots')} className={`px-3 py-2 rounded ${tab==='hotspots'?'bg-indigo-600 text-white':'bg-gray-100'}`}>Hotspots</button>
            <button onClick={()=>setTab('graph')} className={`px-3 py-2 rounded ${tab==='graph'?'bg-indigo-600 text-white':'bg-gray-100'}`}>Graph</button>
            <button onClick={()=>setTab('link')} className={`px-3 py-2 rounded ${tab==='link'?'bg-indigo-600 text-white':'bg-gray-100'}`}>Link Analysis</button>
            <button onClick={()=>setTab('viz')} className={`px-3 py-2 rounded ${tab==='viz'?'bg-indigo-600 text-white':'bg-gray-100'}`}>Cross-Jurisdiction</button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4">
        {loading && <div className="mb-4 p-3 bg-yellow-50 border-l-4 border-yellow-400">Running... please wait</div>}

        {tab==='overview' && (
          <section className="grid grid-cols-3 gap-4">
            <div className="col-span-2 bg-white p-4 rounded shadow">
              <h2 className="text-xl font-medium mb-2">Quick Actions</h2>
              <div className="flex gap-3">
                <button onClick={()=>fileInputRef.current.click()} className="px-4 py-2 bg-indigo-600 text-white rounded">Upload Data</button>
                <button onClick={runFullPipeline} className="px-4 py-2 bg-green-600 text-white rounded">Run Full Pipeline</button>
                <button onClick={()=>fetch('/api/export').then(()=>alert('exported'))} className="px-4 py-2 bg-gray-800 text-white rounded">Export GeoJSON</button>
              </div>
              <div className="mt-4">
                <h3 className="font-semibold">Recent Alerts</h3>
                <ul className="mt-2 space-y-2">
                  {alerts.length===0 && <li className="text-sm text-gray-500">No recent alerts</li>}
                  {alerts.map((a,i)=>(<li key={i} className="text-sm bg-red-50 p-2 rounded">{a}</li>))}
                </ul>
              </div>
            </div>
            <div className="bg-white p-4 rounded shadow">
              <h3 className="font-medium">Status</h3>
              <ul className="mt-2 text-sm space-y-2">
                <li>Files uploaded: {files.length}</li>
                <li>Hotspot cells: {hotspotsGeoJSON ? hotspotsGeoJSON.features?.length || 'unknown' : 'none'}</li>
                <li>Graph nodes: {graphData ? graphData.nodes?.length || 'unknown' : 'none'}</li>
                <li>Top influencers: {centrality ? centrality.top_influencers?.length || 'unknown' : 'none'}</li>
              </ul>
            </div>
            <div className="col-span-3 mt-4 bg-white p-4 rounded shadow">
              <h3 className="font-medium">Notes & Actions</h3>
              <p className="text-sm text-gray-600">This single-page app combines data ingestion, preprocessing, geospatial hotspot forecasting, knowledge-graph construction, SNA, and cross-jurisdiction visualization. Configure backend endpoints (/api/*) to run the heavy tasks. The UI supports drill-down and export.</p>
            </div>
          </section>
        )}

        {tab==='ingest' && (
          <section className="bg-white p-4 rounded shadow">
            <h2 className="text-xl font-medium mb-3">Data Ingestion</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <input ref={fileInputRef} type="file" multiple onChange={uploadFiles} className="hidden" />
                <div className="p-4 border rounded">
                  <p className="text-sm text-gray-600">Upload FIR, CDR, Demographics files (CSV, GeoJSON)</p>
                  <button onClick={()=>fileInputRef.current.click()} className="mt-3 px-4 py-2 bg-indigo-600 text-white rounded">Select Files</button>
                </div>
                <div className="mt-4">
                  <h4 className="font-medium">Uploaded</h4>
                  <ul className="mt-2 space-y-1 text-sm">
                    {files.map((f,i)=>(<li key={i}>{f.name} · {Math.round(f.size/1024)} KB</li>))}
                    {files.length===0 && <li className="text-gray-500">No files yet</li>}
                  </ul>
                </div>
              </div>
              <div>
                <h4 className="font-medium">Ingest Controls</h4>
                <div className="mt-3 space-y-3">
                  <label className="block text-sm">Auto-detect formats</label>
                  <label className="block text-sm">PII Anonymization</label>
                  <div className="flex gap-2 mt-2">
                    <button className="px-3 py-2 bg-yellow-500 rounded text-white">Anonymize</button>
                    <button className="px-3 py-2 bg-green-600 rounded text-white">Validate Schema</button>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {tab==='preprocess' && (
          <section className="bg-white p-4 rounded shadow">
            <h2 className="text-xl font-medium mb-3">Preprocessing & Feature Engineering</h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-2">
                <div className="p-3 border rounded">
                  <h4 className="font-semibold">Pipeline controls</h4>
                  <div className="mt-3 grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm">Geocoding missing FIR locations</label>
                      <input type="checkbox" />
                    </div>
                    <div>
                      <label className="block text-sm">Impute missing demographics</label>
                      <input type="checkbox" />
                    </div>
                    <div>
                      <label className="block text-sm">Anonymize caller IDs</label>
                      <input type="checkbox" />
                    </div>
                    <div>
                      <label className="block text-sm">Temporal alignment window (mins)</label>
                      <input type="number" defaultValue={60} className="w-24" />
                    </div>
                  </div>
                  <div className="mt-4">
                    <button onClick={()=>fetch('/api/preprocess',{method:'POST'})} className="px-4 py-2 bg-indigo-600 text-white rounded">Run Preprocess</button>
                  </div>
                </div>
                <div className="mt-4 p-3 border rounded">
                  <h4 className="font-semibold">Preview: Feature sample</h4>
                  <div className="mt-2 text-sm text-gray-600">(Schema preview will appear here after preprocessing)</div>
                </div>
              </div>
              <div className="p-3 border rounded">
                <h4 className="font-semibold">Quick stats</h4>
                <div className="mt-2 text-sm">
                  <p>FIR records: --</p>
                  <p>CDR records: --</p>
                  <p>Missing geo: --</p>
                </div>
              </div>
            </div>
          </section>
        )}

        {tab==='hotspots' && (
          <section className="bg-white p-4 rounded shadow">
            <h2 className="text-xl font-medium mb-3">Hotspot Forecasting</h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-2">
                <div className="h-96 border rounded p-2 overflow-hidden" id="map-root">
                  <iframe title="map" src="/map_placeholder.html" className="w-full h-full border-0" />
                </div>
                <div className="mt-3 flex gap-3">
                  <button onClick={()=>fetch('/api/predict?horizon=24').then(()=>alert('started'))} className="px-4 py-2 bg-indigo-600 text-white rounded">Predict 24h</button>
                  <button onClick={()=>fetch('/api/predict?horizon=168').then(()=>alert('started'))} className="px-4 py-2 bg-indigo-600 text-white rounded">Predict 7d</button>
                  <button onClick={()=>fetch('/api/export_hotspots').then(()=>alert('exported'))} className="px-3 py-2 bg-gray-800 text-white rounded">Export GeoJSON</button>
                </div>
              </div>
              <div className="p-3 border rounded">
                <h4 className="font-semibold">Legend & Controls</h4>
                <div className="mt-2 text-sm">
                  <p>Color ramp: low → green, medium → yellow, high → red</p>
                  <p>Cell size: adjustable (250m–1000m)</p>
                </div>
              </div>
            </div>
          </section>
        )}

        {tab==='graph' && (
          <section className="bg-white p-4 rounded shadow">
            <h2 className="text-xl font-medium mb-3">Knowledge Graph</h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-2">
                <div className="p-3 border rounded h-96 overflow-auto">
                  <h4 className="font-semibold">Graph builder</h4>
                  <div className="mt-2 text-sm">Create nodes: Person, Phone, Device, Case, Location</div>
                  <div className="mt-3 flex gap-2">
                    <button onClick={()=>fetch('/api/build_graph',{method:'POST'})} className="px-3 py-2 bg-indigo-600 text-white rounded">Build Graph</button>
                    <button onClick={()=>fetch('/api/export_graph').then(()=>alert('graph exported'))} className="px-3 py-2 bg-gray-700 text-white rounded">Export GraphML</button>
                  </div>
                </div>
                <div className="mt-4 p-3 border rounded">
                  <h4 className="font-semibold">Graph preview</h4>
                  <div className="mt-2 text-sm">Nodes: {graphData ? graphData.nodes?.length || 0 : 0} · Edges: {graphData ? graphData.links?.length || 0 : 0}</div>
                </div>
              </div>
              <div className="p-3 border rounded">
                <h4 className="font-semibold">Centrality & Metrics</h4>
                <div className="mt-2 text-sm">
                  <button onClick={()=>fetch('/api/compute_centrality').then(()=>alert('ok'))} className="px-3 py-2 bg-indigo-600 text-white rounded">Compute Centrality</button>
                  <div className="mt-2">Top influencers preview</div>
                  <ol className="text-sm mt-2">
                    {(centrality?.top_influencers || []).slice(0,5).map((x,i)=>(<li key={i}>{x}</li>))}
                  </ol>
                </div>
              </div>
            </div>
          </section>
        )}

        {tab==='link' && (
          <section className="bg-white p-4 rounded shadow">
            <h2 className="text-xl font-medium mb-3">Link Analysis & Prediction</h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-2">
                <div className="p-3 border rounded h-96 overflow-auto">
                  <h4 className="font-semibold">Compute heuristics (Adamic-Adar, Jaccard)</h4>
                  <div className="mt-3 flex gap-2">
                    <button onClick={()=>fetch('/api/link/heuristics').then(r=>r.json()).then(j=>setPredictions(j))} className="px-3 py-2 bg-indigo-600 text-white rounded">Run Heuristics</button>
                    <button onClick={()=>fetch('/api/link/supervised',{method:'POST'}).then(r=>r.json()).then(j=>setPredictions(j))} className="px-3 py-2 bg-green-600 text-white rounded">Train Supervised</button>
                  </div>
                  <div className="mt-4">
                    <h5 className="font-medium">Top predicted links</h5>
                    <ul className="text-sm mt-2 space-y-2">
                      {(predictions?.top || []).map((p,i)=>(<li key={i}>{p.u} ↔ {p.v} · score: {p.score?.toFixed(3)}</li>))}
                      {(!predictions) && <li className="text-gray-500">No predictions yet</li>}
                    </ul>
                  </div>
                </div>
              </div>
              <div className="p-3 border rounded">
                <h4 className="font-semibold">Bridge & Influencer detection</h4>
                <div className="mt-2 text-sm">Run PageRank, Betweenness to surface leaders and bridges.</div>
                <div className="mt-3">
                  <button onClick={()=>fetch('/api/link/top_influencers').then(r=>r.json()).then(j=>setCentrality(j))} className="px-3 py-2 bg-indigo-600 text-white rounded">Top Influencers</button>
                </div>
              </div>
            </div>
          </section>
        )}

        {tab==='viz' && (
          <section className="bg-white p-4 rounded shadow">
            <h2 className="text-xl font-medium mb-3">Cross-Jurisdiction Visualization</h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-2">
                <div className="h-96 border rounded overflow-hidden">
                  <iframe src="/cross_jurisdiction_map.html" className="w-full h-full border-0" title="cross-jurisdiction" />
                </div>
                <div className="mt-3 flex gap-2">
                  <button onClick={()=>fetch('/api/filter?crime=auto_theft').then(()=>alert('filtered'))} className="px-3 py-2 bg-indigo-600 text-white rounded">Filter: Auto theft</button>
                  <button onClick={()=>fetch('/api/filter?time_last=7d').then(()=>alert('filtered'))} className="px-3 py-2 bg-indigo-600 text-white rounded">Last 7 days</button>
                </div>
              </div>
              <div className="p-3 border rounded">
                <h4 className="font-semibold">Filters</h4>
                <div className="mt-2">
                  <label className="block text-sm">Crime type</label>
                  <select className="w-full mt-1 p-2 border rounded"><option>All</option><option>Auto theft</option></select>
                  <label className="block text-sm mt-3">Date range</label>
                  <input type="date" className="w-full p-2 border rounded" />
                </div>
              </div>
            </div>
          </section>
        )}

      </main>

      <footer className="max-w-7xl mx-auto p-4 text-sm text-gray-500">Built for investigative analytics · wire backend endpoints (/api/*) to enable full features</footer>
    </div>
  );
}
