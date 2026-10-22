import argparse
import logging
import sys

from stockbench.config import load_config
from stockbench.run import run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="stockbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment config end to end")
    r.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s"
    )
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
