@echo off

echo "You should set environmental variable JDK_HOME or JAVA_HOME"

set ANT_HOME=.\tools\ant
echo ANT_HOME = %ANT_HOME%

if "%AI_OLD_PATH%" == "" set AI_OLD_PATH=%PATH%

set PATH=.;%JDK_HOME%\bin;%JAVA_HOME%\bin;%ANT_HOME%\bin;%AI_OLD_PATH%

echo PATH=%PATH%

set CLASSPATH=./hudup-core.jar;./hudup.jar;./sim.jar;./ai.jar;./hudup-runtime-lib.jar;./sim-runtime-lib.jar;./ai-runtime-lib.jar;./bin;./lib/*;./working/lib/*;%EXTRA_CLASSPATH%

echo CLASSPATH=%CLASSPATH%

set JAVA_CMD=java -cp %CLASSPATH%

set JAVAW_CMD=javaw -cp %CLASSPATH%

@echo on
