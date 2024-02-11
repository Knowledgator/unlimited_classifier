from __future__ import annotations
from typing import List, Optional, Dict

class TokenNode:
    """
    LabelsTrie node
    """
    children: Dict[int, TokenNode]
    token_id: Optional[int]
    
    def __init__(self, token_id: Optional[int]=None) -> None:
        """
        Args:
            token_id (Optional[int], optional): token id. 
        Defaults to None.
        """        
        self.children: Dict[int, TokenNode] = {}
        self.token_id: Optional[int] = token_id


    def __repr__(self) -> str:
        return f"<Node(token_id={self.token_id})"


class LabelsTrie:
    def __init__(
        self, labels: Optional[List[List[int]]]=None
    ) -> None:
        """
        Args:
            labels (Optional[List[List[int]]], optional): 
        labels for classification. Defaults to None.
        """        
        self.root = TokenNode()

        if labels is not None:
            self.add_batch(labels)


    def _add_token_id(self, token_id: int, root: TokenNode) -> TokenNode:
        """
        Adds token id if not already added

        Args:
            token_id (int): token id
            root (TokenNode): parent node

        Returns:
            TokenNode: current node
        """        
        if token_id not in root.children:
            root.children[token_id] = TokenNode(token_id)
        return root.children[token_id]


    def add(self, tokens: List[int]) -> None:
        """
        Adds tokenized label

        Args:
            tokens (List[int]): tokenized label  
        """        
        current_node = self.root
        for token_id in tokens:
            current_node = self._add_token_id(token_id, current_node)


    def add_batch(self, tokenized_labels: List[List[int]]) -> None:
        """
        Adds multiple labels at once

        Args:
            tokenized_labels (List[List[int]]): _description_
        """        
        for tokens in tokenized_labels:
            self.add(tokens)


    def get(
        self, tokens: List[int]
    ) -> List[int]:
        """
        Returns probable next tokens

        Args:
            tokens (List[int]): tokens for wich probable 
        continuation will be found

        Returns:
            List[int]: probale next tokens
        """        
        current_node = self.root
        for token_id in tokens:
            if token_id in current_node.children:
                current_node = current_node.children[token_id]
            else:
                return []
        return [
            i.token_id
            for i in current_node.children.values()
            if i.token_id is not None
        ]
    

    def __repr__(self) -> str:
        return f"Trie(root={self.root.token_id})"