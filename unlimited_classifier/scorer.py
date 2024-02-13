from typing import List, Tuple

from transformers import pipeline # type: ignore

class Scorer:
    '''
    Class for creating more interpritable scores
    '''
    def __init__(
        self, device: str='cpu', batch_size: int=8
    ) -> None:
        """
        Args:
            device (str, optional): Device. Defaults to 'cpu'.
            batch_size (int, optional): Batch size. Defaults to 8.
        """        
        self.pipeline = pipeline(
            "zero-shot-classification",
            model="knowledgator/comprehend_it-base",
            device=device,
            batch_size=batch_size,
        )

    
    def score(
        self, text: str, scored_labels: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Rescoring of labels

        Args:
            text (str): Input text.
            scored_labels (List[Tuple[str, float]]): Labels with beam score.

        Returns:
            List[Tuple[str, float]]: List of tuples containing classes and
        their corresponding scores. The tuples are sorted by score in 
        descending order.
        """        
        labels = [label for label, _ in scored_labels]
        scores = self.pipeline( # type: ignore
            text, labels, multi_label=True
        )
        return [
            (str(label), float(score)) for label, score in zip(
                scores['labels'], scores['scores'] # type: ignore
            )
        ]