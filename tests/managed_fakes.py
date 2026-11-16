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


class ControlledProcessRunner(ProcessRunner):
    runner = "controlled_process_fake"
    supports_training_control = True

    def __init__(self):
        FakeRunnerAdapter.__init__(self, runner=self.runner)

    def run_training(self, config_path, overrides):
        from agent.hpo.training_control import EpochTrainingController

        result = super().run_training(config_path, overrides)
        spec = overrides.get("_hpo_training_control")
        if spec:
            controller = EpochTrainingController(spec, Path(overrides["output_folder"]) / "training_control.json")
            for epoch in range(1, overrides["number_of_epochs"] + 1):
                loss = 1.0 if overrides["lr"] > 0.001 else 1.0 / epoch
                decision = controller.observe(epoch, {"valid_loss": loss})
                if decision["should_stop"]:
                    break
            result.update(control=decision, epoch_data=controller.history)
        return result

    def normalize_training_result(self, raw):
        result = super().normalize_training_result(raw)
        result.extensions["training_control"] = {"runtime": {"training_control": raw.get("control") or {}}, "epoch_data": raw.get("epoch_data") or []}
        return result

    def run_evaluation(self, config_path, model_path, data_path, overrides):
        checkpoint = Path(model_path)
        checkpoint.with_suffix(".evaluated").write_text("evaluated")
        return super().run_evaluation(config_path, model_path, data_path, overrides)
