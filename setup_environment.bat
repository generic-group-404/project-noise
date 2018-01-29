@ECHO OFF


:choice
set /P c=Do you have python 64-bit installed [Y,N]?
if /I "%c%" EQU "Y" goto :install
if /I "%c%" EQU "N" goto :choice

:install
pip3 install virtualenv
virtualenv .env
.env\Scripts\activate & pip3 install -r requirements.txt
exit