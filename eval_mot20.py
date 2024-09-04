import glob
import os
from eval.eval import eval


if __name__ == '__main__':
    dataset_path = "./data/MOT20/train"
    out_path = "output/mot20"
    exp_name = "val"

    seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

    # 生成val_seqmap.txt文件
    with open(seqmap,"w") as f:
        f.write("name\n")
        for file in glob.glob(os.path.join(out_path, exp_name, "*txt")):
            if "MOT" not in file:
                continue
            f.write(f"{file.split(os.sep)[-1].split('.')[0]}\n")

    HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name,1,False)
    print(f"{HOTA}, {IDF1}, {MOTA}, {AssA}")

def eval_HOTA(wx, wy, a, vmax, dataset_path, out_path, exp_name):
    seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

    # 生成val_seqmap.txt文件
    with open(seqmap,"w") as f:
        f.write("name\n")
        for file in glob.glob(os.path.join(out_path, exp_name, "*txt")):
            if "MOT" not in file:
                continue
            f.write(f"{file.split(os.sep)[-1].split('.')[0]}\n")

    HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name,1,False)

    sum_ = wx / 30 + wy / 30 + vmax / 3 + a / 100

    HOTA = float(HOTA) - sum_ / 400

    return -HOTA

def eval_AssA(wx, wy, a, vmax, dataset_path, out_path, exp_name):
    seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

    # 生成val_seqmap.txt文件
    with open(seqmap,"w") as f:
        f.write("name\n")
        for file in glob.glob(os.path.join(out_path, exp_name, "*txt")):
            if "MOT" not in file:
                continue
            f.write(f"{file.split(os.sep)[-1].split('.')[0]}\n")

    HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name,1,False)

    sum_ = wx / 30 + wy / 30 + vmax / 3 + a / 100

    AssA = float(AssA) - sum_ / 400

    return -AssA

def eval_IDF1(wx, wy, a, vmax, out_path, exp_name):
    dataset_path = "./data/MOT17/train"

    seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

    # 生成val_seqmap.txt文件
    with open(seqmap,"w") as f:
        f.write("name\n")
        for file in glob.glob(os.path.join(out_path, exp_name, "*txt")):
            if "MOT" not in file:
                continue
            f.write(f"{file.split(os.sep)[-1].split('.')[0]}\n")

    HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name,1,False)

    sum_ = wx / 30 + wy / 30 + vmax / 3 + a / 100

    IDF1 = float(IDF1) - sum_ / 400

    return -IDF1
