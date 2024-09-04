import glob
import os
from eval.eval import eval_kitti as eval


if __name__ == '__main__':
    dataset_path = "./data/Kitti/training"
    out_path = "run_output/kitti"
    exp_name = "val"

    seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

    # 生成val_seqmap.txt文件
    with open(seqmap,"w") as f:
        f.write("name\n")
        for file in glob.glob(os.path.join(out_path, exp_name, "*txt")):
            if 'seqmap' in file:
                continue
            f.write(f"{file.split(os.sep)[-1].split('.')[0]}\n")

    HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name)
    print(f"{HOTA}, {IDF1}, {MOTA}, {AssA}")

def eval_AssA(wx, wy, a, vmax, out_path, exp_name):
    dataset_path = "./data/gt/data/gt/kitti/kitti_2d_box_train"

    seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

    # 生成val_seqmap.txt文件
    with open(seqmap,"w") as f:
        f.write("name\n")
        for file in glob.glob(os.path.join(out_path, exp_name, "*txt")):
            if file == os.path.join(out_path,exp_name, "val_seqmap.txt"):
                continue
            f.write(f"{file.split(os.sep)[-1].split('.')[0]}\n")

    HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name)

    sum_ = wx / 30 + wy / 30 + vmax / 3 + a / 100

    print(AssA)

    AssA = float(AssA) - sum_ / 400

    return -AssA

def eval_IDF1(wx, wy, a, vmax, out_path, exp_name):
    dataset_path = "./data/gt/data/gt/kitti/kitti_2d_box_train"

    seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

    # 生成val_seqmap.txt文件
    with open(seqmap,"w") as f:
        f.write("name\n")
        for file in glob.glob(os.path.join(out_path, exp_name, "*txt")):
            if file == os.path.join(out_path,exp_name, "val_seqmap.txt"):
                continue
            f.write(f"{file.split(os.sep)[-1].split('.')[0]}\n")

    HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name)

    sum_ = wx / 30 + wy / 30 + vmax / 3 + a / 100

    print(IDF1)

    IDF1 = float(IDF1) - sum_ / 400

    return -IDF1

def eval_HOTA(wx, wy, a, vmax, out_path, exp_name):
    dataset_path = "./data/gt/data/gt/kitti/kitti_2d_box_train"

    seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

    # 生成val_seqmap.txt文件
    with open(seqmap,"w") as f:
        f.write("name\n")
        for file in glob.glob(os.path.join(out_path, exp_name, "*txt")):
            if file == os.path.join(out_path,exp_name, "val_seqmap.txt"):
                continue
            f.write(f"{file.split(os.sep)[-1].split('.')[0]}\n")

    HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name)

    sum_ = wx / 30 + wy / 30 + vmax / 3 + a / 100

    print(HOTA)

    HOTA = float(HOTA) - sum_ / 400

    return -HOTA
