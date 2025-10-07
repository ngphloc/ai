EXTRA_CLASSPATH=./ea.jar:./sim-ea.jar

. env.sh

eval $JAVA_CMD net.ea.ann.adapter.gen.ui.GenUIClassifier $1 $2 $3 $4