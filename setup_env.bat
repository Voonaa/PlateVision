@echo off
echo ===================================================
echo   Setup Virtual Environment - Tugas Citra Digital
echo ===================================================
echo.

:: Cek Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak terdeteksi! Silakan instal Python terlebih dahulu.
    pause
    exit /b 1
)

:: Buat virtual environment jika belum ada
if exist ".venv" (
    echo [INFO] Virtual environment .venv sudah ada.
    goto ACTIVATE
)

echo [INFO] Membuat virtual environment .venv...
python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Gagal membuat virtual environment!
    pause
    exit /b 1
)

:ACTIVATE
:: Aktivasi virtual environment
echo [INFO] Mengaktifkan virtual environment...
call .venv\Scripts\activate

:: Upgrade pip
echo [INFO] Mengupgrade pip...
python -m pip install --upgrade pip

:: Install PyTorch CPU (untuk menghemat bandwidth dan disk space, ~180MB dibanding CUDA ~2GB)
echo [INFO] Menginstal PyTorch CPU lebih cepat dan ringan...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo [WARNING] Gagal menginstal PyTorch CPU. Akan mencoba instalasi standar.
)

echo [INFO] Menginstal dependensi dari requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Gagal menginstal dependensi!
    pause
    exit /b 1
)

echo.
echo ===================================================
echo   SETUP BERHASIL!
echo ===================================================
echo   Untuk menjalankan aplikasi Gradio, ketik:
echo   python main.py
echo ===================================================
echo.
pause
