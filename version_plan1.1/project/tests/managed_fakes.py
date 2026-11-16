"""Importable process runner; intentionally never imports experiment storage."""
import json
import os
from pathlib import Path
import time
from tests.fakes import FakeRunnerAdapter


class ProcessRunner(FakeRunnerAdapter):
    runner = "process_fake"

    def __init__(self):
        super().__init__(runner="process_fake")

    def run_training(self, config_path, overrides):
        folder = Path(overrides["output_folder"])
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / "fake.ckpt"
        checkpoint.write_text(json.dumps({"lr": overrides.get("lr", .001), "seed": overrides.get("seed", 0), "pid": os.getpid()}))
        time.sleep(.05)
        return {"status": "success", "metrics": {"valid_error_rate": .1},
                "model_paths": [str(checkpoint)], "output_folder": str(folder)}

    def run_evaluation(self, config_path, model_path, data_path, overrides):
        value = json.loads(Path(model_path).read_text())
        return {"status": "success", "metrics": {"eer": value["lr"] + (value["seed"] % 7)*.00001, "min_dcf": .1}}
