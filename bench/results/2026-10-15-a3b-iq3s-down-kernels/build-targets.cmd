@echo off
call "D:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -arch=x64
if errorlevel 1 exit /b %errorlevel%
cmake --build "G:\Strata-Moe\build-a3b" --target strata-moe-iq3s-fused-gu-parity strata-moe --parallel 8
exit /b %errorlevel%
