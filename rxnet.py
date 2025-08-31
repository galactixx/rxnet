"""
Feed-forward character-level network for next-character prediction.

This model embeds character indices, flattens the context window, and applies
two fully connected layers with ReLU activations and dropout, followed by a
final linear layer to produce vocabulary-sized logits.
"""

from typing import Tuple

import torch


class RxNet(torch.nn.Module):
    """Simple MLP over embedded character contexts.

    Attributes:
        embs (torch.nn.Embedding): Embedding table of shape (vocab, 32).
        fc1 (torch.nn.Linear): Linear layer from (context * 32) to hidden[0].
        fc2 (torch.nn.Linear): Linear layer from hidden[0] to hidden[1].
        fc3 (torch.nn.Linear): Linear layer from hidden[1] to vocab size.
        relu (torch.nn.ReLU): Non-linearity applied after `fc1` and `fc2`.
        dropout (torch.nn.Dropout): Dropout regularization (p=0.2).
    """

    def __init__(self, context: int, hidden: Tuple[int, int], vocab: int) -> None:
        """Initialize the network modules.

        Args:
            context (int): Size of the left context window (number of tokens).
            hidden (Tuple[int, int]): Hidden layer widths (hid1, hid2).
            vocab (int): Vocabulary size (number of distinct characters).
        """
        super().__init__()
        hid1, hid2 = hidden
        self.embs = torch.nn.Embedding(vocab, 32)
        self.fc1 = torch.nn.Linear(context * 32, hid1)
        self.fc2 = torch.nn.Linear(hid1, hid2)
        self.fc3 = torch.nn.Linear(hid2, vocab)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute vocabulary logits for the next-character prediction.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, context) containing
                integer character ids (dtype torch.long) for the left context.

        Returns:
            torch.Tensor: Logits of shape (batch_size, vocab).
        """
        # Embed context indices → (batch_size, context, emb_dim)
        x = self.embs(x)
        # Flatten the embedded context → (batch_size, context * emb_dim)
        x = x.view(x.size(0), -1)
        # Two-layer MLP with ReLU and dropout regularization
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Final projection to vocabulary-sized logits
        x = self.fc3(x)
        return x
