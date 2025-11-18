Crime Hotspot Forecasting and Criminal Network Disruption - Project README
Team Number: 35
Team Members: - 
1.	Raj Vardhan Singh, 9123167641, rajvardhansingh776@gmail.com
2.	Rimjhim Mohanty, 7752033923, lonimohanty2003@gmail.com 

Institute Name(s): ITER SOA, Bhubaneshwar

Project name
Crime Hotspot Forecasting and Criminal Network Disruption
Short description
A web-based system that provides unified tools for crime analysis: - Hotspot visualization (map / choropleth) - Predictive modeling and forecasting of crime hotspots - Cross-jurisdiction network visualizer - Link analysis and missing-link prediction - Graph database ingestion (Neo4j or NetworkX) - Data preprocessing utilities
The project contains a React frontend and a Python backend (Flask). All datasets required for demo runs should be included in the datasets/ folder in the zip.
________________________________________
Zip file contents (required)
When you create the submission zip file, it must include the following at the top level:
/project-root
│
├─ frontend/                     # React frontend (CrimeDashboardReact.jsx + full project)
├─ backend/                      # Flask backend (API endpoints that call Python modules)
├─ modules/                      # Your six Python modules (visualization.py, data preprocessing.py, predictive Modelling.py, cross jurisdictional visualisation.py, Link analysis.py, graph Database.py)
├─ datasets/                     # All datasets used for demo / testing
│   ├─ sample_fir.csv
│   ├─ sample_cdr.csv
│   ├─ hotspots.geojson
│   ├─ demographics.geojson
│   └─ demo_graph.graphml
├─ requirements.txt              # Python dependencies for backend
├─ package.json                  # Frontend dependencies
├─ README.md                     # This file
└─ run_demo.sh                   # Convenience script to run backend + frontend locally
Important: Put all datasets required to demonstrate each functionality inside datasets/. If a module expects a different filename, list that mapping in the README (see next section).
________________________________________
Dataset list & mapping
Include these example dataset files (rename or add more as needed). In the zip, include small demo versions (few hundred rows) so the app runs fast.
•	datasets/sample_fir.csv — FIR dataset (columns: fir_id, date_time, crime_type, latitude, longitude, accused, victim, details)
•	datasets/sample_cdr.csv — CDR dataset (caller_id, callee_id, timestamp, cell_tower_id, latitude, longitude)
•	datasets/hotspots.geojson — GeoJSON for hot regions used by visualization
•	datasets/demographics.geojson — demographic polygons, optional
•	datasets/demo_graph.graphml — small sample graph for link analysis (or edgelist CSV)
•	datasets/README-datasets.md — short README describing each dataset, field names, and source
Place any other additional datasets required by your modules in the same folder and list them in datasets/README-datasets.md.
________________________________________
System requirements (OS / Tools)
I tested these instructions on Ubuntu 20.04 and Windows 10 / WSL2. They should work on macOS too.
•	OS: Linux / macOS / Windows 10 (WSL2 recommended)
•	Python 3.10+ (3.8+ usually OK)
•	Node.js 18+ and npm/yarn
•	Neo4j (optional) — Neo4j Desktop or Neo4j Aura; Neo4j 4.x or 5.x supported
•	Git (optional)
•	Recommended RAM: 8GB+ (for larger datasets)
________________________________________
Backend — Python (Flask) setup
1.	Create a virtual environment and activate it:
 	python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
2.	Install dependencies:
 	pip install -r backend/requirements.txt
 	Example backend/requirements.txt (should be included in zip):
 	flask
flask-cors
pandas
geopandas
shapely
fiona
pyproj
rtree
folium
streamlit-folium
networkx
pyvis
scikit-learn
neo4j
openpyxl
 	(Adjust versions to match your environment.)
3.	Environment variables (create backend/.env):
 	FLASK_APP=app.py
FLASK_ENV=development
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
4.	Running backend (development):
 	cd backend
source ../.venv/bin/activate
flask run --port 5000
 	The backend will expose endpoints used by the frontend:
o	POST /api/hotspots/upload — upload hotspots file
o	POST /api/predict/run — run predictive pipeline (accepts FIR file)
o	POST /api/cross/build — build cross-jurisdiction graph (accepts multiple files)
o	POST /api/link/analyze — run link analysis (accepts graph file)
o	POST /api/graph/ingest — ingest graph into Neo4j or build networkx graph
o	POST /api/preprocess/upload — preprocessing endpoints
5.	Note about modules:
o	Place your six Python modules under modules/. The Flask app will import them (e.g. from modules import visualization as viz).
o	The README should list any non-obvious function names the backend expects (e.g. predict_hotspots, load_fir).
________________________________________
Frontend — React (Next.js recommended) setup
I provided a single-file React component CrimeDashboardReact.jsx as a starting point. For a full project:
1.	Create a Next.js app (recommended) or CRA:
 	npx create-next-app crime-frontend
cd crime-frontend
npm install axios react-leaflet leaflet recharts react-force-graph
npm install tailwindcss postcss autoprefixer
npx tailwindcss init -p
2.	Add the provided component CrimeDashboardReact.jsx in pages (Next.js) or src/components (CRA). Wire in Tailwind config (optional).
3.	Configure proxy for local development (to avoid CORS):
o	In Next.js, use next.config.js rewrites or set axios.defaults.baseURL = 'http://localhost:5000' in the frontend.
4.	Run frontend:
 	npm run dev         # Next.js (default port 3000)
5.	Open http://localhost:3000 to see the UI. Use the file-upload components to call the Flask APIs.
________________________________________
Neo4j setup (optional)
If you want Neo4j ingestion:
1.	Install Neo4j Desktop or run Docker:
 	docker run --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j:5
2.	Configure backend/.env with the bolt URI, username, password.
3.	The backend uses the official neo4j Python driver to write nodes/relationships. Ensure the GraphDB module exposes an ingestion function build_graph_from_files(..., mode='neo4j', neo4j_config=cfg) (matching the Streamlit prototype).
________________________________________
How to run the full demo locally (step-by-step)
4.	Unzip the submission zip:
 	unzip unified-crime-dashboard.zip -d unified-crime-dashboard
cd unified-crime-dashboard
5.	Create virtualenv & install backend deps:
 	python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
6.	Install frontend deps:
 	cd frontend
npm install
7.	Start backend:
 	cd ../backend
flask run --port 5000
8.	Start frontend (new terminal):
 	cd frontend
npm run dev
9.	Open http://localhost:3000 and interact with the dashboard. Use provided sample datasets from /datasets folder for each feature.
________________________________________
GUI description & walkthrough
•	Hotspot Visualization: upload hotspots.geojson or sample_fir.csv. Map displays cells and popups with scores.

•	Predictive Modeling: upload sample_fir.csv, click run. Displays top cells and a small forecast chart.

•	Cross-Jurisdiction Network: upload multiple FIR/CDR files; backend merges and constructs a graph, frontend shows force-graph.

•	Link Analysis: upload demo_graph.graphml or edgelist CSV to see centralities, communities, and predicted missing links.

•	Graph DB Ingestion: upload FIR/CDR and choose neo4j or networkx. For Neo4j, ensure DB running and credentials set.

•	Data Preprocessing: quick tools to standardize columns, geocode missing lat/lon from addresses (if supported), normalize timestamps.
________________________________________
Troubleshooting & tips
•	If GeoPandas install fails on Windows, use WSL2 or Conda.

•	If the ForceGraph fails to render in Next.js, ensure dynamic import with ssr: false.

•	Use small sample datasets during demo; large files will slow down.

•	If Neo4j shows authentication errors, verify NEO4J_AUTH and NEO4J_URI.
________________________________________
Preview Link(another prototype with firebase)
https://9000-firebase-studio-1763103019315.cluster-c36dgv2kibakqwbbbsgmia3fny.cloudworkstations.dev 
