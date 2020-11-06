for file in `find $PWD -name '*_test.py'`
do
  echo $file
  python $file
done
