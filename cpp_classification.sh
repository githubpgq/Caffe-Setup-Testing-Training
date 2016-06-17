#Classifying ImageNet: using C++ API

CAFFE_ROOT=/home/cv-lab/caffe-master

PROTOTXT=$CAFFE_ROOT/models/bvlc_reference_caffenet/deploy.prototxt
#PROTOTXT=$CAFFE_ROOT/models/vgg_ilsvrc/VGG_ILSVRC_16_layers_deploy.prototxt

CAFFEMODEL=$CAFFE_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
#CAFFEMODEL=$CAFFE_ROOT/models/vgg_ilsvrc/VGG_ILSVRC_16_layers.caffemodel

MEANBINARYPROTO=$CAFFE_ROOT/data/ilsvrc12/imagenet_mean.binaryproto

WORDS=$CAFFE_ROOT/data/ilsvrc12/synset_words.txt


$CAFFE_ROOT/build/examples/cpp_classification/classification.bin $PROTOTXT $CAFFEMODEL $MEANBINARYPROTO $WORDS /home/cv-lab/Pictures/aero.jpg
