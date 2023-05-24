import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
import ijson

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder", type=str, default="../airborne-detection-starter-kit/utility/data",
                        help="the folder that contain part1 part2 part3")
    parser.add_argument("--target-folder", type=str,
                        help="where to place the converted dataset format")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_folder).rglob()
