echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v252\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "C:\PROGRA~1\ANSYSI~1\v252\fluent\ntbin\win64\tell.exe" DESKTOP-BBN4MI6 61374 CLEANUP_EXITING
timeout /t 1
"C:\PROGRA~1\ANSYSI~1\v252\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-BBN4MI6" (%KILL_CMD% 34436) 
if /i "%LOCALHOST%"=="DESKTOP-BBN4MI6" (%KILL_CMD% 29320) 
if /i "%LOCALHOST%"=="DESKTOP-BBN4MI6" (%KILL_CMD% 42808) 
if /i "%LOCALHOST%"=="DESKTOP-BBN4MI6" (%KILL_CMD% 50128) 
if /i "%LOCALHOST%"=="DESKTOP-BBN4MI6" (%KILL_CMD% 48388) 
if /i "%LOCALHOST%"=="DESKTOP-BBN4MI6" (%KILL_CMD% 49472)
del "C:\Users\radie\Desktop\TrainingSIMS\CADModels\cleanup-fluent-DESKTOP-BBN4MI6-48388.bat"
