# 🚀 Final Deployment & Major Project Viva Checklist

### 1. ⏱️ 15 Minutes Before Presentation
- [ ] **Run Locally First:** Double-click `run.bat` or run:
  ```bash
  streamlit run app.py
  ```
- [ ] Open your browser to `http://localhost:8501`.
- [ ] Test that the browser window is zoomed to **90% or 100%** so the dark command center metrics, charts, and 4-patch satellite images fit comfortably on the classroom projector.
- [ ] Click through:
  - Tab 1 (Overview & KPIs)
  - Tab 2 (Priority Triage Dispatch)
  - Tab 3 (Building Deep-Dive - check building `1139290306` to confirm heatmaps load)
  - Tab 5 (Calibration & Reliability)
  - Tab 8 (Viva Defense & Methodology)
  - Sidebar: click "Download Field Incident PDF Report" to make sure the PDF opens.

### 2. 🌐 Cloud Deployment Options (If Showing on a Remote Link)
If your college requires a public live URL:
1. **Option A: Streamlit Community Cloud (Recommended - 100% Free)**
   - Push this repo to your GitHub account (`git push origin main`).
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Click **"New App"** -> select your repo -> Main file path: `app.py`.
   - Click **Deploy**! (Takes ~2 minutes).
2. **Option B: Hugging Face Spaces (Free Docker/Streamlit)**
   - Create a new Space on Hugging Face -> choose **Docker** SDK.
   - Push your code; the included `Dockerfile` will build and run automatically.
3. **Option C: Local Tunneling (Ngrok / Localhost.run - Zero Cloud Setup)**
   - If running locally on your laptop and you want professors to open it on their phones/laptops:
     ```bash
     ssh -R 80:localhost:8501 localhost.run
     ```
     This immediately gives you a temporary public `https://...` link for free without installing anything!

### 3. ⚠️ Critical "Night-Before" Gotchas & How We Prevented Them
- **GitHub 100MB File Limit:** Checkpoint files (`.pt`) over 100MB are kept in `.gitignore`. The repo now includes all precomputed JSONs, evaluation plots, and 40 paired heatmaps so the app works **100% offline** on any computer without needing the raw 10GB dataset or model weights!
- **Missing Dependencies on College PC:** `requirements.txt` has been locked with all exact packages (`torch`, `torchvision`, `kornia`, `h5py`, `fpdf2`, `opencv-python-headless`, etc.).
- **No Internet in the Exam Hall:** Since the app runs locally on `http://localhost:8501` and uses local cached assets, **you do NOT need internet to run the presentation!**

---

### 🎤 5-Minute Presentation Speaking Script

- **Minute 1: The Disaster Problem**
  *"Respected teachers, after an earthquake like the 2023 Türkiye-Syria event, first responders have a 72-hour golden window. Optical satellite assessment fails during dust storms, clouds, and at night, and suffers from false alarms due to roof shadows and tarps."*

- **Minute 2: The Multi-Modal SAR Solution**
  *"We built DamageTriage, fusing Sentinel-1 SAR and optical imagery. SAR uses active microwave radar to penetrate weather and detects building collapse via loss of the wall-ground double-bounce reflection."*

- **Minute 3: Technical Rigor (Handling Imbalance & Calibration)**
  *"Earthquake data has a 96:4 intact-to-damaged imbalance. We solved this with Binary Focal Loss and Weighted Sampling. Furthermore, instead of treating triage as a naive 50% cutoff, we formulated a 4-tier calibrated priority queue with temperature scaling, validated by monotonic damage rates across tiers."*

- **Minute 4: Live Demo Walkthrough**
  *(Show Tab 1 KPIs -> Tab 2 Priority Queue -> Tab 3 Paired Grad-CAM inspection -> Tab 8 Empirical Ablation showing 0% false alarms with SAR vs 25% with optical only).*

- **Minute 5: Rescue Report Generation & Conclusion**
  *(Click 'Download Field Incident PDF Report', show the generated PDF, and conclude with field readiness).*
