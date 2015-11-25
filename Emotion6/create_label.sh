cd images
x=0
echo `pwd`
for i in `ls`
do
	#cd "$i"
	#cd images/fear
	for j in `ls $i`
	do
		#convert -resize 256x256! "`pwd`/$i/$j" "`pwd`/$i/$j"
		echo "images/$i/$j $x" >> /home/omair/caffe-master/Emotion6/test.txt
		#echo "images/fear/$j $x" >> /home/omair/caffe-master/Emotion6/train.txt
	done
	x=$((x+1))
done
