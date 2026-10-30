#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""

import os
import random
import sys
from functools import partial

import torch
from hyperpyyaml import load_hyperpyyaml

from agent.runners.speechbrain_dependency import patch_torchaudio_compatibility
from agent.utils.logger import progressbar_enabled

patch_torchaudio_compatibility()

import speechbrain as sb
from recipes.voxceleb.audio_compat import audio_io
from recipes.voxceleb.batch_compat import with_padded_batch
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main



class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"""

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Feature extraction and normalization
        if (
            hasattr(self.hparams, "use_tacotron2_mel_spec")
            and self.hparams.use_tacotron2_mel_spec
        ):
            feats = self.hparams.compute_features(audio=wavs)
            feats = torch.transpose(feats, 1, 2)
        else:
            feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            spkid = self.hparams.wav_augment.replicate_labels(spkid)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def audio_pipeline(
    wav, start, stop, duration, *, sample_rate, sentence_len, random_chunk,
    crop_mode=None,
):
    """Load a training chunk using worker-serializable configuration."""
    snt_len_sample = int(sample_rate * sentence_len)
    if crop_mode not in {None, "random", "center", "bounds"}:
        raise ValueError("validation_crop must be 'random', 'center', or 'bounds'")
    use_random = random_chunk if crop_mode is None else crop_mode == "random"
    if crop_mode == "center":
        start = max(0, (int(float(duration) * sample_rate) - snt_len_sample) // 2)
        stop = start + snt_len_sample
    elif use_random:
        duration_sample = max(0, int(float(duration) * sample_rate))
        if duration_sample > snt_len_sample:
            start = random.randint(0, duration_sample - snt_len_sample)
        else:
            start = 0
        stop = start + snt_len_sample
    else:
        start = max(0, int(start))
        stop = max(start + 1, int(stop))
    num_frames = max(1, stop - start)
    sig, _sample_rate = audio_io.load(wav, num_frames=num_frames, frame_offset=start)
    return sig.transpose(0, 1).squeeze(1)


def label_pipeline(spk_id, *, label_encoder):
    """Encode a static speaker label after fitting the shared encoder."""
    return label_encoder.encode_sequence_torch([spk_id])


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Bind only small, serializable configuration; do not capture the entire
    # HyperPyYAML graph (model, optimizer, checkpointer) in a local closure.
    validation_crop = hparams.get("validation_crop")
    if validation_crop not in {None, "random", "center", "bounds"}:
        raise ValueError("validation_crop must be 'random', 'center', or 'bounds'")
    for data, crop in ((train_data, None), (valid_data, validation_crop)):
        sb.dataio.dataset.add_dynamic_item(
            [data],
            partial(
                audio_pipeline,
                sample_rate=hparams["sample_rate"],
                sentence_len=hparams["sentence_len"],
                random_chunk=hparams["random_chunk"],
                crop_mode=crop,
            ),
            takes=["wav", "start", "stop", "duration"],
            provides=["sig"],
        )

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="spk_id",
    )
    if hparams.get("out_n_neurons") is not None:
        label_encoder.expect_len(int(hparams["out_n_neurons"]))
    sb.dataio.dataset.add_dynamic_item(
        datasets, partial(label_pipeline, label_encoder=label_encoder),
        takes=["spk_id"], provides=["spk_id_encoded"],
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder


if __name__ == "__main__":
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    #用来解析命令行语句的
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exclude verification sentences from train)
    veri_file_path = os.path.join(
        hparams["save_folder"], os.path.basename(hparams["verification_file"])
    )
    download_file(hparams["verification_file"], veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from recipes.voxceleb.voxceleb_prepare import prepare_voxceleb  # noqa

    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "verification_pairs_file": veri_file_path,
            "splits": ["train", "dev"],
            "split_ratio": hparams["split_ratio"],
            "split_unit": hparams.get("split_unit", "utterance"),
            "seg_dur": hparams["sentence_len"],
            "random_segment": hparams["random_chunk"],
            "skip_prep": hparams["skip_prep"],
        },
    )
    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    dataloader_options = with_padded_batch(hparams["dataloader_options"])

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=dataloader_options,
        valid_loader_kwargs=dataloader_options,
        progressbar=progressbar_enabled(run_opts),
    )
