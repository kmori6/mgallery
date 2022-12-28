from argparse import ArgumentParser
from logging import INFO, Formatter, StreamHandler, getLogger


def get_logger(name: str, level: int = INFO):
    logger = getLogger(name)
    logger.setLevel(INFO)
    logger.propagate = False
    hdlr = StreamHandler()
    hdlr.setLevel(level)
    fmt = Formatter("%(asctime)s (%(name)s) %(levelname)s: %(message)s")
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    return logger


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    return parser
