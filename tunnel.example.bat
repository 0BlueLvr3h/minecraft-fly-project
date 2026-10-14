@echo off
title Mosca tunnel -^> VM
rem Template: copy it to tunnel.bat (excluded from git) and fill in the three entries below.
rem Forwards the brain (port 8765 on this PC) to the VM, where the plugin finds it at
rem ws://GATEWAY:8765. Keep this window open; if the connection drops it reconnects.

set GATEWAY=172.18.0.1
set KEY=%USERPROFILE%\.ssh\vm-key.key
set VM=ubuntu@VM_IP

:loop
echo [%time%] Opening the tunnel to the VM...
ssh -N -R %GATEWAY%:8765:127.0.0.1:8765 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -i "%KEY%" %VM%
echo [%time%] Tunnel dropped, retrying in 5 seconds... (Ctrl+C to close)
timeout /t 5 >nul
goto loop
