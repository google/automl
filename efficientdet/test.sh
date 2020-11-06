for file in `find $PWD/efficientdet -name '*_test.py'`
do
  echo $file
  python $file
done
