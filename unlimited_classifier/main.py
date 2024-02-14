from typing import Tuple, List, Union, Optional

from transformers import ( # type: ignore
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    GenerationMixin,
    TFGenerationMixin,
    FlaxGenerationMixin,
    AutoConfig,
    AutoModelForCausalLM
)
from transformers.utils import ModelOutput # type: ignore
import torch
import pyximport # type: ignore
pyximport.install() # type: ignore

from unlimited_classifier.labels_trie import LabelsTrie # type: ignore
from unlimited_classifier.scorer import Scorer

class TextClassifier:
    """
    Class for text classification
    """

    model: PreTrainedModel

    def initialize_labels_trie(self, labels: List[str]) -> None:
        """
        Initializing the labels trie

        Args:
            labels (List[str]): Labels that will be used.
        """ 
        tokenized_labels = []
        for label in labels:
            tokens = self.tokenizer.encode(label)
            if tokens[0] == self.tokenizer.bos_token_id:
                tokens = tokens[1:]
            tokenized_labels.append([self.pad_token] + tokens)
        self.trie: LabelsTrie = LabelsTrie( tokenized_labels )

    def initiaize_model(self, model: str) -> PreTrainedModel:
        try:
            config = AutoConfig.from_pretrained(model)
        except Exception:
            raise ValueError("The path to the model does't exist or it's unavailable on Hugging Face.")

        if config.is_encoder_decoder:
            return ( # type: ignore
                AutoModelForSeq2SeqLM # type: ignore
                .from_pretrained(model)
                .to(self.device)
            )
        else:
            return ( # type: ignore
                AutoModelForCausalLM # type: ignore
                .from_pretrained(model)
                .to(self.device)
            ) 

    def __init__(
        self,
        labels: List[str], 
        model: Union[str, PreTrainedModel],
        tokenizer: Union[
            str, PreTrainedTokenizer, PreTrainedTokenizerFast
        ],
        prompt: str = "Classifity the following text:\n {}\nLabel:",
        device: str="cpu",
        num_beams: int=5,
        max_new_tokens: int=512,
        pad_token: Optional[int]=None,
        eos_token: Optional[int]=None,
        scorer: Optional[Scorer]=None,
    ) -> None:
        """
        Args:
            labels (List[str]): Labels for classification.
            
            model (Union[str, PreTrainedModel]): Model.
            
            tokenizer (Union[
                str, PreTrainedTokenizer, PreTrainedTokenizerFast
            ]): Tokenizer.
            
            device (str, optional): Device. Defaults to "cpu".
            
            num_beams (int, optional): Number of beams. Defaults to 5.

            max_new_tokens (int, optional): Maximum newly generated tokens.
        Defaults to 512.

        Raises:
            ValueError: If no labels are provided.

            ValueError: If a generative model is expected but not provided.
        """        
        if not labels:
            raise ValueError("No labels provided.")
        
        self.device = device
        self.num_beams = min(num_beams, len(labels))
        self.max_new_tokens = max_new_tokens
        self.scorer = scorer
        self.prompt = prompt

        if isinstance(model, str):
            self.model = self.initiaize_model(model)
        else:
            self.model = model

        if (
            not any(
                isinstance(self.model, t) for t in ( # type: ignore
                    GenerationMixin, 
                    TFGenerationMixin, 
                    FlaxGenerationMixin
                ) 
            ) 
            and not self.model.config.is_decoder # type: ignore
            and not self.model.config.is_encoder_decoder # type: ignore
        ): 
            raise ValueError("Expected generative model.")
        
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) # type: ignore
        else:
            self.tokenizer = tokenizer

        self.tokenizer.padding_side = "left" # type: ignore
        
        if eos_token is not None:
            self.tokenizer.eos_token_id = eos_token

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pad_token = (
            pad_token
            if not pad_token is None
            else self.tokenizer.pad_token_id
        )

        self.encoder_decoder = self.model.config.is_encoder_decoder

        self.initialize_labels_trie(labels)

    def _get_candidates(self, sent, prompt_len):
        gen_sent = sent.tolist()
        if not self.encoder_decoder:
            gen_sent = gen_sent[prompt_len:]
            gen_sent.insert(0, self.pad_token)

        return (self.trie.get(gen_sent) # type: ignore
                            or [self.tokenizer.eos_token_id])
    
    def predict(self, prompts: List[str]) -> ModelOutput:
        """
        Model prediction

        Args:
            prompts (List[str]): Texts joined with prompts.

        Returns:
            ModelOutput: Output generated by the model.
        """
        tokenized_prompt = self.tokenizer( # type: ignore
                prompts, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
        prompt_len = tokenized_prompt['input_ids'].shape[-1]
        outputs = self.model.generate( # type: ignore
            **tokenized_prompt,
            pad_token_id=self.pad_token,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            num_return_sequences=self.num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            prefix_allowed_tokens_fn=(
                lambda _, sent: self._get_candidates(sent, prompt_len)
            )
        )
        return outputs # type: ignore


    def invoke(self, text: str) -> List[Tuple[str, float]]:
        """
        Invokation for single text

        Args:
            text (str): Text for classification.

        Returns:
            List[Tuple[str, float]]: List of tuples containing classes and
        their corresponding scores. The tuples are sorted by score in 
        descending order.
        """        
        return self.invoke_batch([text])[0]


    def invoke_batch(self, texts: List[str]) -> List[List[Tuple[str, float]]]:
        """
        Invokation for multiple texts

        Args:
            texts (List[str]): Texts for classification.

        Returns:
            List[List[Tuple[str, float]]]: A list of lists where each inner
        list contains tuples of classes and their corresponding scores for the
        corresponding texts. The tuples are sorted by score in descending order.
        The order of the inner lists matches the order of the input texts.
        """
        inputs = [self.prompt.format(text) for text in texts]
        outputs = self.predict(inputs)
        decodes = self.tokenizer.batch_decode( # type: ignore
            outputs.sequences, # type: ignore
            skip_special_tokens=True
        )

        if self.num_beams == 1:
            scores = torch.ones(len(decodes))
        else:
            scores = outputs.sequences_scores # type: ignore
        
        outputs2scores: List[List[Tuple[str, float]]] = []
        for text_id in range(len(texts)):
            input_ = inputs[text_id]
            input_len = len(input_)
            batch: List[Tuple[str, float]] = []
            for beam_id in range(self.num_beams):
                score = scores[text_id*self.num_beams+beam_id] # type: ignore
                p = torch.exp(score).item() # type: ignore
                label = decodes[text_id*self.num_beams+beam_id]
                if not self.encoder_decoder:
                    label = label[input_len:].strip()
                batch.append((label, p))
            outputs2scores.append(
                self.scorer.score(texts[text_id], batch) 
                if self.scorer 
                else batch
            )
        return outputs2scores
    

    async def ainvoke(self, text: str) -> List[Tuple[str, float]]:
        """
        Async invoke

        Args:
            text (str): Text for classification.

        Returns:
            List[Tuple[str, float]]: List of tuples containing classes and
        their corresponding scores. The tuples are sorted by score in 
        descending order.
        """        
        return self.invoke(text)


    async def ainvoke_batch(self, texts: List[str]) -> List[List[Tuple[str, float]]]:
        """
        Async invokation for multiple texts

        Args:
            texts (List[str]): Texts for classification.

        Returns:
            List[List[Tuple[str, float]]]: A list of lists where each inner
        list contains tuples of classes and their corresponding scores for the
        corresponding texts. The tuples are sorted by score in descending order.
        The order of the inner lists matches the order of the input texts.
        """    
        return self.invoke_batch(texts)