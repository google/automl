for file in `find $PWD/efficientdet -name '*.py'`
do
  pylint --rcfile=.pylintrc file
done

for file in `find $PWD/efficientdet -name '*_test.py'`
do
  python $file
done
