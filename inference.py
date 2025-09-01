"""
Inference script for generating RxNorm-like names with a trained RxNet.

This module loads the character vocabulary and a saved checkpoint, then
samples new names one character at a time using top-k sampling with an
optional temperature. It pads the left context with start tokens ("^")
and stops when an end-of-sequence token ("$") is produced or a maximum
length is reached.
"""

import torch

from data import load_model_config, load_rxnorm_data
from rxnet import RxNet

TOPK = 5  # Sample from the top-K most probable next characters
MAXIMUM_CHARS = 13  # Safety cap on generated sequence length
NAMES_TO_GENERATE = 10  # Number of names to generate in this run

TEMPERATURE = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load vocabulary/mappings and restore the trained model
    config = load_rxnorm_data()
    model_config = load_model_config(config=config)
    START_CONTEXT = "^" * model_config.context

    # Restore the trained model with the correct configuration
    model = RxNet(
        context=model_config.context,
        hidden=model_config.hidden,
        vocab=model_config.vocab,
    )
    # Restore weights from checkpoint and move to the active device
    model.load_state_dict(
        torch.load("rxnet.pth", map_location=device, weights_only=True)
    )
    model.to(device)

    model.eval()

    with torch.no_grad():
        for _ in range(NAMES_TO_GENERATE):
            # Per-sample state
            char = None
            chars = 0
            start_ctx = config.encode(context=START_CONTEXT)

            # Predict next char from rolling left context
            while char != "$" and chars < MAXIMUM_CHARS:
                ctx = torch.tensor(start_ctx[-model_config.context :]).to(device)
                ctx = ctx.unsqueeze(0)

                # Forward pass and temperature scaling
                logits = model(ctx)
                logits = logits / TEMPERATURE
                logits = logits.squeeze(0)

                # Top-K sampling to avoid low-probability tails
                values, indices = torch.topk(logits, TOPK)
                values = values.softmax(dim=0)
                choice = torch.multinomial(values, 1).item()

                # Convert sampled index back to a character and append
                code = indices[choice].item()
                char = config.id_to_char[code]

                start_ctx.append(code)
                chars += 1

            # Remove padding and end tokens, then print
            name = "".join(config.id_to_char[i] for i in start_ctx)
            name = name.lstrip("^")
            name = name.rstrip("$")
            print(name)
