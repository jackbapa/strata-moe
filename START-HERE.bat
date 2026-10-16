@echo off
rem First run installs Strata-Moe; later runs reuse the model and compiled engine.
call "%~dp0SETUP.bat" %*
exit /b %errorlevel%
