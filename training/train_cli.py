"""
python training/train_cli.py
python training/train_cli.py --override epochs=150 device=cpu
"""
import argparse, yaml
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="training/configs/tlogo_v11m.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="key=val overrides")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    for kv in args.override:  # пример: --override epochs=150 device=cpu
        k, v = kv.split("=", 1)
        try: v = int(v)
        except:
            try: v = float(v)
            except: pass
        cfg[k] = v

    model = YOLO(cfg.pop("model"))
    model.train(**cfg)

if __name__ == "__main__":
    main()
