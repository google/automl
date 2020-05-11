# EfficientDet FQA

go/g3doc-canonical-go-links

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'tanmingxing' reviewed: '2020-05-09' }
*-->

[TOC]

## For Users

### How can I convert the saved model to tflite?

Unfortunately, there is no way to do that with the current public tensorflow
release due to some issues in tf converter. We have some internal fixes, which
could potentially be available with the next TensorFlow release.

## For Developers

### How can I format my code for PRs?

Please use [yapf](https://github.com/google/yapf) with option
--style='{based_on_style: google, indent_width: 2}'. You can also save the
following file to ~/.config/yapf/style:

    [style]
    based_on_style = google
    indent_width = 2

If you want to check the format with lint, please run:

    !pylint --rcfile=../.pylintrc your_file.py

### How can I run all tests?

    !export PYTHONPATH="`pwd`:$PYTHONPATH"
    !find . -name "*_test.py" | parallel python &> /tmp/test.log \
      && echo "All passed" || echo "Failed! Search keyword FAILED in /tmp/test.log"
