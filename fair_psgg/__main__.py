from argparse import ArgumentParser
import json
import torch
from .config import Config
from .trainer import Trainer
from .utils import detect_task_spooler


def cli():
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument(
        "output",
        help="Path to output folder. Note that all contents inside that folder will be overwritten",
    )
    parser.add_argument(
        "--anno", required=True, help="Path to PSG JSON annotation file"
    )
    parser.add_argument("--img", required=True, help="Directory that contains images")
    parser.add_argument(
        "--seg", required=True, help="Directory that contains segmentation masks"
    )
    parser.add_argument(
        "--epochs", default=40, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="Number of workers for the data loaders"
    )
    parser.add_argument("--model-state", default=None)
    parser.add_argument("--no-bpbar", default=False, action="store_true")
    parser.add_argument(
        "--cfg",
        nargs="+",
        help="""Overrides for the config file. Must be specified as key=value pairs.
You can list multiple entries like --cfg architecture.transformer_depth=3 data.source=psg""",
    )
    args = parser.parse_args()

    anno_path = args.anno
    img_dir = args.img
    seg_dir = args.seg

    config = Config.from_file(args.config)
    if args.cfg:
        config = config.with_cli_overrides(args.cfg)

    if args.model_state is None:
        model_state_dict = None
    else:
        model_state_dict = torch.load(args.model_state, map_location="cpu")["model"]

    if args.no_bpbar:
        hide_batch_progress = True
    elif detect_task_spooler():
        print("Detected task-spooler. Batch progress will be hidden")
        hide_batch_progress = True
    else:
        hide_batch_progress = False

    trainer = Trainer(
        anno_path=anno_path,
        img_dir=img_dir,
        seg_dir=seg_dir,
        out_dir=args.output,
        config=config,
        num_workers=args.workers,
        start_state_dict=model_state_dict,
        hide_batch_progress=hide_batch_progress,
    )
    if trainer.out_dir:
        with open(trainer.out_dir / "args.json", "w") as f:
            json.dump(args.__dict__, f)
    trainer.run(epochs=args.epochs)
    # to easily check which experiments ran to the end
    if trainer.out_dir:
        with open(trainer.out_dir / "done.txt", "w") as f:
            f.write("done")

    print(
        "Best value for ",
        trainer.critical_metric,
        ": ",
        trainer.best_metric_value,
        sep="",
    )


if __name__ == "__main__":
    cli()
