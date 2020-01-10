:loop1
..\build\Release\performTestCase.exe D:\Seafile\GraphsAndMatrices\floridadata 1 0 1 256,3,2,4,4,16,512,8 3 f
if %ERRORLEVEL% NEQ 0 goto loop1

:loop2
..\build\Release\performTestCase.exe D:\Seafile\GraphsAndMatrices\floridadata 1 0 1 256,3,2,4,4,16,512,8 3 d
if %ERRORLEVEL% NEQ 0 goto loop2