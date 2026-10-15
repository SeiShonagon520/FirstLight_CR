@echo off
setlocal
cd /d "%~dp0"
echo 正在启动 FirstLight CR 控制台界面...
".venv\Scripts\python.exe" -X utf8 -m native_runner.user_interface --skip-preflight-dialog %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [错误] 控制台启动异常退出，代码：%ERRORLEVEL%
    pause
)
exit /b %ERRORLEVEL%
