import glob
import os
from eval.eval import eval


dataset_path = "./data/MOT17/train"
out_path = "output/mot17"
exp_name = "val"

seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

# 生成val_seqmap.txt文件
with open(seqmap,"w") as f:
    f.write("name\n")
    for file in glob.glob(os.path.join(out_path, exp_name, "*txt")):
        if "seqmap" in file:
            continue
        f.write(f"{file.split(os.sep)[-1].split('.')[0]}\n")

HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name,1,False)
print(f"{HOTA}, {IDF1}, {MOTA}, {AssA}")