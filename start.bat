@echo off
echo ========================================
echo   AI-Powered Semantic Search Setup
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo.
echo Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Running setup script...
python setup.py
if %errorlevel% neq 0 (
    echo ERROR: Setup script failed
    echo Please check your PostgreSQL configuration in .env file
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Setup completed successfully!
echo ========================================
echo.
echo Choose an option to start the application:
echo 1. FastAPI web interface
echo 2. Streamlit app
echo 3. Run demo script
echo 4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting FastAPI server...
    python fastapi_app.py
) else if "%choice%"=="2" (
    echo Starting Streamlit app...
    streamlit run streamlit_app.py
) else if "%choice%"=="3" (
    echo Running demo script...
    python demo.py
) else (
    echo Goodbye!
    pause
    exit /b 0
)

pause
