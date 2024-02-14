import asyncio

from transformers import ( # type: ignore
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM
)

from unlimited_classifier.main import TextClassifier
from unlimited_classifier.scorer import Scorer

LABELS = ["positive", "negative", "neutral"]
TEXT = "I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered controversial I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it\"s not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn\"t have much of a plot."

def initialize_test_classifier():
    model_name = "t5-base"
    return TextClassifier(
        labels=LABELS,
        model=model_name,
        tokenizer=AutoTokenizer.from_pretrained(model_name), # type: ignore
        max_new_tokens=30000,
    )


def test_sync():
    classifier = initialize_test_classifier()
    output = classifier.invoke(TEXT)
    assert len(output) == len(LABELS)
    assert all(label in LABELS for label, _ in output)


def test_sync_batch():
    half = len(TEXT) // 2

    classifier = initialize_test_classifier()
    output = classifier.invoke_batch([TEXT[:half], TEXT[half:]])
    assert len(output) == 2
    assert len(output[0]) == len(output[1])
    assert all(label in LABELS for label, _ in output[0])
    assert all(label in LABELS for label, _ in output[1])


def test_async():
    classifier = initialize_test_classifier()
    output = asyncio.run(classifier.ainvoke(TEXT))
    assert len(output) == len(LABELS)
    assert all(label in LABELS for label, _ in output)


def test_async_batch():
    half = len(TEXT) // 2

    classifier = initialize_test_classifier()
    output = asyncio.run(
        classifier.ainvoke_batch([TEXT[:half], TEXT[half:]])
    )
    assert len(output) == 2
    assert len(output[0]) == len(output[1])
    assert all(label in LABELS for label, _ in output[0])
    assert all(label in LABELS for label, _ in output[1])


def test_not_generative_model():
    try:
        _ = TextClassifier(
            labels=LABELS,
            model="distilbert-base-uncased",
            tokenizer="distilbert-base-uncased",
        )        
        raise Exception("Should raise an error")
    except ValueError as e:
        assert e.args == ("Expected generative model.",)


def initialize_test_llm_classifier():
    model_name = "openai-community/gpt2"
    config = AutoConfig.from_pretrained(model_name) # type: ignore
    model = AutoModelForCausalLM.from_config(config) # type: ignore
    return TextClassifier(
        labels=LABELS,
        model=model, # type: ignore
        tokenizer=AutoTokenizer.from_pretrained(model_name), # type: ignore
        max_new_tokens=30000,
    )


def test_sync_llm():
    classifier = initialize_test_llm_classifier()
    output = classifier.invoke(TEXT)
    assert len(output) == len(LABELS)
    assert all(label in LABELS for label, _ in output)


def test_sync_batch_llm():
    half = len(TEXT) // 2

    classifier = initialize_test_llm_classifier()
    output = classifier.invoke_batch([TEXT[:half], TEXT[half:]])
    assert len(output) == 2
    assert len(output[0]) == len(output[1])
    assert all(label in LABELS for label, _ in output[0])
    assert all(label in LABELS for label, _ in output[1])


def initialize_text_classifier_with_scorer():
    return TextClassifier(
        labels=LABELS,
        model="t5-base",
        tokenizer="t5-base",
        max_new_tokens=30000,
        scorer=Scorer()
    )


def test_sync_with_scorer():
    classifier = initialize_text_classifier_with_scorer()
    output = classifier.invoke(TEXT)

    assert len(output) == len(LABELS)
    assert all(label in LABELS for label, _ in output)


def test_sync_batch_with_scorer():
    half = len(TEXT) // 2

    classifier = initialize_text_classifier_with_scorer()
    output = classifier.invoke_batch([TEXT[:half], TEXT[half:]])
    assert len(output) == 2
    assert len(output[0]) == len(output[1])
    assert all(label in LABELS for label, _ in output[0])
    assert all(label in LABELS for label, _ in output[1])


def test_async_with_scorer():
    classifier = initialize_text_classifier_with_scorer()
    output = asyncio.run(classifier.ainvoke(TEXT))
    assert len(output) == len(LABELS)
    assert all(label in LABELS for label, _ in output)


def test_async_batch_with_scorer():
    half = len(TEXT) // 2

    classifier = initialize_text_classifier_with_scorer()
    output = asyncio.run(
        classifier.ainvoke_batch([TEXT[:half], TEXT[half:]])
    )
    assert len(output) == 2
    assert len(output[0]) == len(output[1])
    assert all(label in LABELS for label, _ in output[0])
    assert all(label in LABELS for label, _ in output[1])