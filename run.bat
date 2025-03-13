@echo off

REM 1) Check if Python is installed
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed.
    echo Please install Python from https://www.python.org/downloads/
    echo Follow the installation instructions and make sure to add Python to your PATH.
    pause
    exit /b 1
)

REM 2) Check for and activate a virtual environment
if exist venv (
    echo Virtual environment 'venv' already exists. Activating...
    call venv\Scripts\activate
) else (
    echo Creating virtual environment 'venv'...
    python -m venv venv
    call venv\Scripts\activate
    echo Virtual environment 'venv' created and activated.
)

REM 3) Check and install requirements from requirements.txt
if exist requirements.txt (
    echo Checking and installing requirements from requirements.txt...
    pip install --upgrade pip
    pip install -r requirements.txt
    echo Requirements installed.
) else (
    echo requirements.txt not found. If your project has dependencies, create a requirements.txt file.
)

REM 4) Run main.py
echo Running main.py...
python main.py

pause