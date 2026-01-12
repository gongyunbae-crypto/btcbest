---
description: Force restart the Streamlit backend automatically without user confirmation
---

// turbo-all

1. Restart the application by killing existing processes and running app.py
   Command: `Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -like "*streamlit run app.py*"} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Sleep 2; python -m streamlit run app.py`
