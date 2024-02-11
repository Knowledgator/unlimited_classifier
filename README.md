# Ultimited classifier

Universal text classifier for generative models

## Install

``` console
pip install -U unlimited-classifier
```

## How to use

Initialize TextClassifier with labels for classification and transformers model and tokenizer (names or actual model/tokenizer):

``` python
TextClassifier(
    labels=['positive', 'negative', 'neutral'],
    model="t5-base",
    tokenizer="t5-base",
)
```

Call method that suits you best:

- invoke: for single text;
- ainvoke: asynchronous for single text;
- invoke_batch: for multiple texts;
- ainvoke_batch: asynchronous for multiple texts;

``` python
text = '''text for classification'''
texts = [
    '''text for classification 1''',
    '''text for classification 2'''
]

output = classifier.invoke(text)

OR

output = classifier.ainvoke(text)

OR

output = classifier.invoke_batch(texts)

OR

output = classifier.ainvoke_batch(texts)
```