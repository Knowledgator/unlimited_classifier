from typing import Tuple, List, Union, Optional, cast

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
    trie: LabelsTrie

    def initialize_labels_trie(self, labels: List[str]) -> None:
        """
        Initializing the labels trie

        Args:
            labels (List[str]): Labels that will be used.
        """ 
        tokenized_labels = []
        for label in labels:
            tokens = self.tokenizer.encode(label) # type: ignore
            if tokens[0] == self.tokenizer.bos_token_id:
                tokens = tokens[1:]
            tokenized_labels.append([self.pad_token_id] + tokens) # type: ignore
        self.trie = LabelsTrie(tokenized_labels)


    def initialize_model(self, model: str) -> PreTrainedModel:
        """
        Model initialization

        Args:
            model (str): Model name or path.

        Raises:
            ValueError: If model doesnt exist or cant be found.

            ValueError: If provided model is not generative.
        
        Returns:
            PreTrainedModel: Initialized model.
        """        
        try:
            config = AutoConfig.from_pretrained(model) # type: ignore
        except Exception:
            raise ValueError(
                "The path to the model does't exist or it's unavailable on"
                " Hugging Face."
            )

        if config.is_encoder_decoder: # type: ignore
            return ( # type: ignore
                AutoModelForSeq2SeqLM # type: ignore
                .from_pretrained(model)
                .to(self.device)
            )
        else:
            try:
                return ( # type: ignore
                    AutoModelForCausalLM # type: ignore
                    .from_pretrained(model)
                    .to(self.device)
                )
            except:
                raise ValueError("Expected generative model.")


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
        pad_token_id: Optional[int]=None,
        eos_token_id: Optional[int]=None,
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

            pad_token_id (int, optional): Pad token that will be used.
        If not provided will be equal to provided in tokenizer config. 

            eos_token_id (int, optional): EOS token that will be used.
        If not provided will be equal to provided in tokenizer config.

            scorer (Scorer, optional): Scorer class. Used for rescoring.

        Raises:
            ValueError: If no labels are provided.

            ValueError: If provided model is not generative.
        """        
        if not labels:
            raise ValueError("No labels provided.")
        
        self.device = device
        self.num_beams = min(num_beams, len(labels))
        self.max_new_tokens = max_new_tokens
        self.scorer = scorer
        self.prompt = prompt

        if isinstance(model, str):
            self.model = self.initialize_model(model)
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
        
        if eos_token_id is not None:
            self.tokenizer.eos_token_id = eos_token_id

        if pad_token_id is not None:
            self.tokenizer.pad_token_id = pad_token_id

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pad_token_id: int = cast(int, self.tokenizer.pad_token_id)

        self.encoder_decoder: bool = self.model.config.is_encoder_decoder # type: ignore

        self.initialize_labels_trie(labels)


    def _get_candidates(
        self, sent: torch.Tensor, prompt_len: int
    ) -> List[int]:
        """
        Get next possible candidates

        Args:
            sent (torch.Tensor): Tensor.
            prompt_len (int): Prompt length.

        Returns:
            List[int]: Possible next tokens.
        """        
        gen_sent: List[int] = sent.tolist() # type: ignore
        if not self.encoder_decoder:
            gen_sent = [self.pad_token_id, *gen_sent[prompt_len:]]

        return (
            self.trie.get(gen_sent) # type: ignore
            or [self.tokenizer.eos_token_id]
        )
    

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
        prompt_len = tokenized_prompt['input_ids'].shape[-1] # type: ignore
        outputs = self.model.generate( # type: ignore
            **tokenized_prompt, # type: ignore
            pad_token_id=self.pad_token_id,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            num_return_sequences=self.num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            prefix_allowed_tokens_fn=(
                lambda _, sent: self._get_candidates(sent, prompt_len) # type: ignore
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