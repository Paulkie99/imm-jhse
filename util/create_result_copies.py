import shutil
import glob

target_dir = "output/mot17/test"
files = glob.glob(f"{target_dir}/*.txt")

for file in files:
    shutil.copy(file, file.split("-SDP")[0] + "-FRCNN.txt")
    shutil.copy(file, file.split("-SDP")[0] + "-DPM.txt")
