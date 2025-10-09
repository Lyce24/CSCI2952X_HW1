import argparse
import yaml
import time

from phase1_ssl import ssl
from phase2_linear_probing import lp

def main(config):
    print("Starting SSL phase...")
    t0 = time.perf_counter()
    ssl(config)
    t1 = time.perf_counter()
    print(f"SSL phase completed in {t1 - t0:.2f} seconds.\n")

    print("Starting Linear Probing phase...")
    t2 = time.perf_counter()
    lp(config)
    t3 = time.perf_counter()
    print(f"Linear Probing phase completed in {t3 - t2:.2f} seconds.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)