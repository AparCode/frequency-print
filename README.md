# frequency-print
Capstone Project

# How To Use

## Run the web app on Windows (PowerShell)

Use one command from the project root:

```powershell
.\start_server.ps1
```

Optional custom port:

```powershell
.\start_server.ps1 -Port 8080
```

Then open:

```text
http://127.0.0.1:8000
```

## Why `source .venv/bin/activate` fails on Windows

That command is for Linux/macOS shells. In Windows PowerShell, use:

```powershell
.\venv\Scripts\Activate.ps1
```

For conda, use:

```powershell
C:\Users\Apar2\miniconda3\Scripts\activate
conda activate frequency-print
```
