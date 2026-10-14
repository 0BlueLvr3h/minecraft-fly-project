@echo off
title Tunnel mosca -^> VM
rem Modello: copialo in tunnel.bat (escluso da git) e completa le tre voci qui sotto.
rem Porta il cervello (porta 8765 di questo PC) alla VM, dove il plugin lo trova su
rem ws://GATEWAY:8765. Lascia aperta questa finestra; se la connessione cade si ricollega.

set GATEWAY=172.18.0.1
set CHIAVE=%USERPROFILE%\.ssh\chiave-della-vm.key
set VM=ubuntu@IP_DELLA_VM

:loop
echo [%time%] Apro il tunnel verso la VM...
ssh -N -R %GATEWAY%:8765:127.0.0.1:8765 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -i "%CHIAVE%" %VM%
echo [%time%] Tunnel caduto, riprovo tra 5 secondi... (Ctrl+C per chiudere)
timeout /t 5 >nul
goto loop
