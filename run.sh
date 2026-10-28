#!/bin/bash
echo "====================================================================="
echo " DamageTriage: Multi-Modal SAR + Optical Disaster Response Dashboard"
echo "====================================================================="
echo "Starting Streamlit Command Center on http://localhost:8501 ..."
echo ""

streamlit run app.py --server.port 8501 --server.headless true
