🚀 OrbitalSight – Hackathon 2025 Final Submission

AI-powered orbital debris visualization, clustering & risk prediction

📌 Overview

Space debris is one of the biggest threats to satellite operators, astronauts, and space missions.
OrbitalSight is a real-time platform that helps track, cluster, and assess orbital debris risks using AI + data visualization.

This project was built as part of ASTRA Hackathon 2025 (DTU) by Team MindFirst.

✨ Features

✅ TLE ingestion – loads orbital elements from CelesTrak
 (with fallback support)
✅ Orbit propagation – simulates trajectories using the SGP4 model
✅ Clustering – DBSCAN clustering to detect debris clouds
✅ Close approach detection – KDTree proximity search for risky debris near satellites
✅ Risk scoring – logistic-style scoring model with relative velocity + distance
✅ Risk zones heatmap – 2D density map with hotspots
✅ Risk trends – time-series forecasting with peak detection & confidence bands
✅ Interactive dashboard – 3D scatter, globe view, textured Earth, timelapse, risk zones, and cluster insights
✅ Export reports – one-click CSV report download
✅ Professional UI – gradient theme, branding, predictive alerts, and insights

📂 Project Structure
OrbitalSight/
│── orbital_sight.py       # Main dashboard + pipeline
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│── .gitignore             # Ignore cache files
│── demo.png               # Dashboard screenshot
│── debris_sample.tle      # Sample fallback TLE data

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/<your-username>/OrbitalSight.git
cd OrbitalSight

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the Project
python orbital_sight.py


The interactive dashboard will launch at:
👉 http://127.0.0.1:8050

📊 Dashboard Tabs

🌍 3D Scatter → orbital debris points, colored by clusters

🌐 Globe View → orthographic Earth view with debris distribution

🌍 Textured Earth → realistic Earth rendering with debris overlay

🎥 Time-lapse → debris movement over time (2D/3D animated)

🚨 Close Approaches → risk table of satellites vs debris

🔥 Risk Zones → 2D heatmap + hotspot detection

📈 Risk Trend → forecast of risk levels with confidence bands

📊 Cluster Insights → per-cluster risk analysis with 🟢🟡🔴 risk categories

📥 Reports → export CSV risk report

🚀 Future Scope → planned extensions for ML and real mission integration

📖 Tech Stack

Python 3.12

SGP4 – orbit propagation

scikit-learn – DBSCAN clustering

SciPy – KDTree nearest neighbor search

Plotly + Dash – visualization & interactive dashboard

Joblib – parallel propagation

📊 Sample Output
Risk Heatmap with Hotspots

🚀 Future Scope

🔹 Integration with ISRO / NASA Space Situational Awareness systems
🔹 Deep learning (LSTM / Transformers) for orbit forecasting
🔹 Probabilistic collision modeling with covariance data
🔹 Crew health and mission safety extensions (Vyommitra-inspired)

👨‍💻 Team

Team MindFirst – Finalists, ASTRA Hackathon 2025 (DTU)

Sahil Ravi Kumar Jain

Varad Kishan kotalwar

📜 License

MIT License – free to use, modify, and share with attribution.