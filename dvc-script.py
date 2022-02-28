import sys
import os.path as osp
from pathlib import Path

if __name__ == "__main__":
    name = sys.argv[1]
    dirpath = osp.join("logs/experiments/runs", name)

    # 가장 마지막에 생성된 log 폴더
    path= sorted(Path(dirpath).iterdir(), key=osp.getmtime)[-1]
    with open(osp.join(path, "train.log"), mode="r") as f:
        lines = f.readlines()

    ckpt = lines[-1].split()[-1]
    print(ckpt)