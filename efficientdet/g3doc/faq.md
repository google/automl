# EfficientDet FQA

go/g3doc-canonical-go-links

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'tanmingxing' reviewed: '2020-05-09' }
*-->


[TOC]

## For Users

### How can I convert the saved model to tflite?

Unfortunately, there is no way to do that with the current public tensorflow release due to some issues in tf converter.
We have some internal fixes, which could potentially be available with the next TensorFlow release.

## For Developers

### How can I format my code for PRs?

    !pylint --rcfile=../.pylintrc your_file.py

### How can I run all tests?

    !export PYTHONPATH="`pwd`:$PYTHONPATH"
    !find . -name "*_test.py" | parallel python &> test.log \
      && echo "All passed" || echo "Failed! Search keyword FAILED in test.log"
