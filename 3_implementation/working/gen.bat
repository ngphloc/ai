@echo off

cd ..

call .\env.bat
%JAVA_CMD% -Xmx6g net.ea.ann.gen.ConvGenModelAssoc

cd working

@echo on
