# End-to-End Experiments

These small research scripts run recent Spyx reference models on real Tonic-backed datasets through the existing Spyx loaders.

## Included experiments

- `nmnist_logpolar.py`: `LogPolarFoveatedConvSNN` on NMNIST.
- `nmnist_event_pooling.py`: `EventDrivenPoolingSNN` on NMNIST.
- `shd_tiny_transformer.py`: `TinySpikingTransformerSNN` on SHD.

## Requirements

- Install loader dependencies: `pip install spyx[loaders]`
- Run from the repository root so `research.end_to_end` is importable.

## Example commands

```bash
python research/end_to_end/nmnist_logpolar.py --epochs 3 --train-limit 1024 --eval-limit 256
python research/end_to_end/nmnist_event_pooling.py --epochs 3 --train-limit 1024 --eval-limit 256
python research/end_to_end/shd_tiny_transformer.py --epochs 3 --channels 72 --sample-t 64 --train-limit 2048 --eval-limit 512
```

The defaults are intentionally small so they serve as quick end-to-end sanity runs rather than full benchmark recipes.