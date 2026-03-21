@echo off
echo ================================
echo  AAPL ML Project - Setup
echo ================================

python -m venv venv
call venv\Scripts\activate

echo Installing packages...
pip install --upgrade pip
pip install yfinance pandas numpy pyarrow ta pandas-ta scikit-learn matplotlib plotly jupyter

echo.
echo Setup complete! To activate your environment next time:
echo   call venv\Scripts\activate
echo.
echo To run the pipeline:
echo   python pipeline\01_fetch_data.py
echo   python pipeline\02_features.py
echo   python pipeline\03_labels.py
pause
