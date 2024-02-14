# Ultimited classifier

![Tree](images/tree.jpeg)

This is a universal text classifier that uses text generation with constraints for large-scale zero-shot text categorization. We represent labels as a tree of tokens and limit the possible variants of generation to provided labels. Such approach allows to make classification of text having millions of labels. 

## Install

``` console
pip install -U unlimited-classifier
```

## How to use

Initialize TextClassifier with labels for classification and transformers model and tokenizer (names or actual model/tokenizer):

``` python
TextClassifier(
    labels=[
        '''label for classification 1''',
        '''label for classification 2'''    
    ],
    model='''model name''',
    tokenizer='''tokenizer name''',
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

## Scorer:

Without a scorer, the results yield beam scores that are difficult to interpret.

You can utilize the Scorer class to interpret scores within the range of 0 to 1.

Initialization with scorer:

``` python
TextClassifier(
    labels=[
        '''label for classification 1''',
        '''label for classification 2'''    
    ],
    model='''model name''',
    tokenizer='''tokenizer name''',
    scorer=Scorer()
)
```

Scorere initialized with "knowledgator/comprehend_it-base" model.

## Examples:

``` python
classifier = TextClassifier(
    labels=["positive", "negative", "neutral"],
    model="t5-base",
    tokenizer="t5-base",
)

text = "Characterize movie review: I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered controversial I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men. What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it\"s not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films. I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn\"t have much of a plot."

output = classifier.invoke(text)
# [('negative', 0.0007813576376065612), ('positive', 0.0006674602045677602), ('neutral', 0.00023184997553471476)]
```

With scorer:

``` python
from unlimited_classifier.scorer import Scorer

classifier = TextClassifier(
    labels=["positive", "negative", "neutral"],
    model="t5-base",
    tokenizer="t5-base",
    scorer=Scorer()
)

output = classifier.invoke(text)
# [('positive', 0.9700802564620972), ('negative', 0.9582770466804504), ('neutral', 0.6801289319992065)]
```