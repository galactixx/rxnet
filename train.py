"""
Training script for a character-level next-character prediction model on RxNorm.

This module prepares context/target pairs from normalized RxNorm names, builds
PyTorch datasets and data loaders, and trains the `RxNet` model with mixed
precision, AdamW, LR scheduling (ReduceLROnPlateau), and EMA for evaluation.
"""

from typing import List, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from data import ContextPair, RxNormDataConfig, load_model_config, load_rxnorm_data
from rxnet import RxNet

SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocessing(config: RxNormDataConfig) -> List[ContextPair]:
    """Create fixed-window context/target pairs from RxNorm names.

    For each name, the left context is initialized with `CONTEXT` start tokens
    ("^") and slides one character at a time. The end-of-sequence token ("$")
    is emitted as the final target for each name.

    Args:
        config (RxNormDataConfig): Vocabulary and mapping utilities.

    Returns:
        List[ContextPair]: All context/target pairs across names.
    """
    pairs: List[ContextPair] = []

    for name in config.names:
        length = len(name)
        idx = 0
        # Initialize the left context with start tokens.
        context = config.encode(context="^" * config.context)

        while idx < length:
            target = name[idx]
            pair = config.get_pair(target=target, context=context)

            context = context[1:]
            context.append(config.char_to_id[target])
            pairs.append(pair)
            idx += 1

        # Append a terminal pair with end-of-sequence as the target.
        pair = config.get_pair(target="$", context=context)
        pairs.append(pair)

    return pairs


def evaluate(
    model: RxNet,
    loader: DataLoader,
    criterion: CrossEntropyLoss,
    ema: ExponentialMovingAverage,
) -> Tuple[float, float]:
    """Evaluate with EMA-averaged weights on a data loader.

    Temporarily swaps model parameters with their EMA averages for evaluation
    and restores them afterward.

    Args:
        model (RxNet): Trained model.
        loader (DataLoader): Evaluation data loader.
        criterion (CrossEntropyLoss): Loss function.
        ema (ExponentialMovingAverage): EMA wrapper over model parameters.

    Returns:
        Tuple[float, float]: Average loss and accuracy over the dataset.
    """
    model.eval()
    ema.store()
    ema.copy_to()

    correct = total = running_loss = 0

    with torch.no_grad():
        for ctxs, targets in tqdm(loader, desc="Evaluation: "):
            ctxs, targets = ctxs.to(device), targets.to(device)

            logits = model(ctxs)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)

            correct += (preds == targets).sum().item()
            total += ctxs.size(0)

            running_loss += loss.item() * ctxs.size(0)

    ema.restore()
    return running_loss / len(loader.dataset), correct / total


if __name__ == "__main__":

    config = load_rxnorm_data()
    model_config = load_model_config(config=config)

    pairs = preprocessing(config=config)

    targets = [pair.target for pair in pairs]
    train_pairs, test_pairs, _, _ = train_test_split(
        pairs, targets, test_size=0.2, random_state=SEED
    )

    class ContextPairs(Dataset):
        """Dataset wrapping a list of `ContextPair` examples.

        Attributes:
            pairs (List[ContextPair]): Stored training/eval examples.
        """

        def __init__(self, pairs: List[ContextPair]) -> None:
            super().__init__()
            self.pairs = pairs

        def __len__(self) -> int:
            """Return dataset size.

            Returns:
                int: Number of stored pairs.
            """
            return len(self.pairs)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            """Get one sample.

            Args:
                idx (int): Index into the dataset.

            Returns:
                Tuple[torch.Tensor, int]: Encoded context tensor and target id.
            """
            pair = self.pairs[idx]
            return torch.tensor(pair.context), pair.target

    train_dataset = ContextPairs(pairs=train_pairs)
    test_dataset = ContextPairs(pairs=test_pairs)

    g = torch.Generator()
    g.manual_seed(SEED)

    # Data loaders: shuffle for training, deterministic for evaluation.
    trainloader = DataLoader(train_dataset, shuffle=True, generator=g, batch_size=64)
    testloader = DataLoader(test_dataset, shuffle=False, batch_size=32)

    EPOCHS = 200
    PATIENCE = 7

    no_improve, best_loss = 0, float("inf")

    # Hidden layer widths scale with context size.

    model = RxNet(
        context=model_config.context,
        hidden=model_config.hidden,
        vocab=model_config.vocab,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    criterion = CrossEntropyLoss(label_smoothing=0.05)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for ctxs, targets in tqdm(trainloader, desc=f"Epoch {epoch+1}"):
            ctxs, targets = ctxs.to(device), targets.to(device)

            optimizer.zero_grad()

            with autocast():
                logits = model(ctxs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            running_loss += loss.item() * ctxs.size(0)

        train_loss = running_loss / len(trainloader.dataset)
        val_loss, val_acc = evaluate(model, testloader, criterion, ema)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS}.. "
            f"Train loss: {train_loss:.3f}.. "
            f"Val loss: {val_loss:.3f}.. "
            f"Accuracy: {val_acc:.3f}.."
        )

        if val_loss < best_loss:
            no_improve = 0
            best_loss = val_loss
        else:
            no_improve += 1
            if no_improve > PATIENCE:
                break
