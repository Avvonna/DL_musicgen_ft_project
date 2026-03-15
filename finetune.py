import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils.data_utils import copy_configs
from utils.dir_utils import AUDIOCRAFT_DIR

os.environ["WANDB_MODE"] = "online"

if "USER" not in os.environ:
    os.environ["USER"] = "Vlad"

AUDIOCRAFT_PATH = Path(AUDIOCRAFT_DIR)

MODEL_SCALE = "small"
PRETRAINED_CHECKPOINT = f"//pretrained/facebook/musicgen-{MODEL_SCALE}"

DATASET_CONFIG_NAME = "musiccaps_train"
CONDITIONER_CONFIG_NAME = "text2music"

DURATION = 10
BATCH_SIZE = 4
NUM_WORKERS = 0

EPOCHS = 3
UPDATES_PER_EPOCH = 1100    # 4400 сэмплов
OPTIMIZER = "adamw"
LEARNING_RATE = 1e-5
MAX_NORM = 0.5

LR_SCHEDULER = "cosine"
COSINE_WARMUP = 100

VALID_NUM_SAMPLES = 100     # 400 сэмплов
GEN_NUM_SAMPLES = 4

SEED = random.randint(0, 999999)

def build_command() -> list[str]:
    return [
        sys.executable, "-m", "dora", "run",
        f"seed={SEED}",

        "solver=musicgen/musicgen_base_32khz",
        f"model/lm/model_scale={MODEL_SCALE}",
        f"conditioner={CONDITIONER_CONFIG_NAME}",
        f"continue_from={PRETRAINED_CHECKPOINT}",

        f"dset={DATASET_CONFIG_NAME}",

        f"dataset.batch_size={BATCH_SIZE}",
        f"dataset.num_workers={NUM_WORKERS}",
        f"dataset.segment_duration={DURATION}",

        f"+dataset.train.min_audio_duration={DURATION-1}",
        f"+dataset.valid.min_audio_duration={DURATION-1}",
        f"+dataset.evaluate.min_audio_duration={DURATION-1}",
        f"+dataset.generate.min_audio_duration={DURATION-1}",

        f"dataset.valid.num_samples={VALID_NUM_SAMPLES}",
        "+dataset.valid.shuffle=false",

        f"evaluate.every={EPOCHS+1}",
        "generate.every=1",
        f"dataset.generate.num_samples={GEN_NUM_SAMPLES}",
        "generate.lm.unprompted_samples=true",
        "generate.lm.prompted_samples=false",
        "generate.lm.gen_gt_samples=false",

        f"optim.epochs={EPOCHS}",
        f"optim.updates_per_epoch={UPDATES_PER_EPOCH}",
        f"optim.optimizer={OPTIMIZER}",
        f"optim.lr={LEARNING_RATE}",
        f"optim.max_norm={MAX_NORM}",
        "optim.ema.use=false",

        "checkpoint.save_last=true",
        "checkpoint.save_every=1",
        "checkpoint.keep_last=5",

        f"schedule.lr_scheduler={LR_SCHEDULER}",
        f"schedule.cosine.warmup={COSINE_WARMUP}",

        "logging.log_tensorboard=false",
        "logging.log_wandb=true",
        "wandb.project=musicgen-hw4",
        f"wandb.name=run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",

        "mp_start_method=spawn",
    ]


def main():
    configs_to_copy = [
        (f"configs/{DATASET_CONFIG_NAME}.yaml", f"audiocraft/config/dset/{DATASET_CONFIG_NAME}.yaml"),
        (f"configs/{CONDITIONER_CONFIG_NAME}.yaml", f"audiocraft/config/conditioner/{CONDITIONER_CONFIG_NAME}.yaml"),
    ]
    copy_configs(configs_to_copy)

    command = build_command()

    try:
        subprocess.run(command, check=True, cwd=str(AUDIOCRAFT_PATH))
    except subprocess.CalledProcessError as e:
        print(f"\nОшибка запуска dora: {e}")
        raise


if __name__ == "__main__":
    main()
