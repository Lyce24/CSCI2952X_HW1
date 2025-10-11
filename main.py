import time

from phase1_ssl import ssl
from phase2_linear_probing import lp

import argparse
import yaml

def main(config: dict):
    start_time = time.time()
    ssl(config)
    lp(config)
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
