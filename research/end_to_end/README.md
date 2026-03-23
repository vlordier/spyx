# End-to-End Experiments

These small research scripts run recent Spyx reference models on real Tonic-backed datasets through the existing Spyx loaders.

## Included experiments

- `nmnist_logpolar.py`: `LogPolarFoveatedConvSNN` on NMNIST.
- `nmnist_event_pooling.py`: `EventDrivenPoolingSNN` on NMNIST.
- `dvs_gesture_foveated.py`: `IntegratedWTAFoveatedSNN` or `EventDrivenSparseFoveatedSNN` on DVS Gesture.
- `shd_tiny_transformer.py`: `TinySpikingTransformerSNN` on SHD.
- `shd_spike_frequency.py`: `SpikeFrequencyCodedSNN` on SHD.

## Requirements

- Install the project in editable mode with loader extras: `pip install -e .[loaders]`

The experiments are packaged via `pyproject.toml`, so they can be run either as modules or as console scripts after installation.

## Example commands

```bash
python -m research.end_to_end.nmnist_logpolar --epochs 3 --train-limit 1024 --eval-limit 256
python -m research.end_to_end.nmnist_event_pooling --epochs 3 --train-limit 1024 --eval-limit 256
python -m research.end_to_end.dvs_gesture_foveated --variant integrated-wta --epochs 3 --train-limit 256 --eval-limit 64
python -m research.end_to_end.shd_tiny_transformer --epochs 3 --channels 72 --sample-t 64 --train-limit 2048 --eval-limit 512
python -m research.end_to_end.shd_spike_frequency --epochs 3 --channels 72 --sample-t 64 --train-limit 2048 --eval-limit 512

spyx-exp-nmnist-logpolar --epochs 3 --train-limit 1024 --eval-limit 256
spyx-exp-nmnist-event-pooling --epochs 3 --train-limit 1024 --eval-limit 256
spyx-exp-dvs-gesture-foveated --variant integrated-wta --epochs 3 --train-limit 256 --eval-limit 64
spyx-exp-shd-tiny-transformer --epochs 3 --channels 72 --sample-t 64 --train-limit 2048 --eval-limit 512
spyx-exp-shd-spike-frequency --epochs 3 --channels 72 --sample-t 64 --train-limit 2048 --eval-limit 512
```

The defaults are intentionally small so they serve as quick end-to-end sanity runs rather than full benchmark recipes.