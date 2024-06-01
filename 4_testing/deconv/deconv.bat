@echo off

cd ..

call .\env.bat
%JAVA_CMD% net.ea.ann.conv.ConvNetworkImpl2

cd working

@echo on
