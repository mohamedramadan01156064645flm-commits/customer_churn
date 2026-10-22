@echo off
title Telco Customer Churn Prediction Studio
echo ========================================================
echo   Starting Telco Customer Churn Prediction Live Demo...
echo ========================================================
echo.
echo URL: http://localhost:8502
echo.
.venv\Scripts\streamlit.exe run app.py --server.port 8502
pause
