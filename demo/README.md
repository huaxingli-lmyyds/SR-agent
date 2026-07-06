# SR-agent Demo

This demo runs the real SpeechBrain ECAPA-TDNN HPO path through `main.py`.

Dry run:

```bash
python demo/run_ecapa_hpo.py --data-folder /tmp/voxceleb1 --dry-run
```

Run:

```bash
python demo/run_ecapa_hpo.py --data-folder /tmp/voxceleb1
```

To add a new model, register its runner adapter under `agent/runners`, add a
training config under `configs`, then create a demo JSON file with the desired
`runner`, `implementation`, `model_family`, budget, and search space.
