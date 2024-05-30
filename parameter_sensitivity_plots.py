import os
import shutil
import matplotlib
import numpy as np
from eval_mot17 import eval_HOTA
from util.run_ucmc import make_args, run_ucmc

det_path = "det_results/mot17/yolox_x_ablation"
cam_path = "cam_para/MOT17"
gmc_path = "gmc/mot17"
out_path = "output/mot17"
exp_name = "val"
dataset = "MOT17"
args = make_args()

seqs = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
dynamic_seqs = ["MOT17-05", "MOT17-10", "MOT17-11", "MOT17-13"]
static_seqs = ["MOT17-02", "MOT17-04", "MOT17-09"]

default_params = {
    "MOT17-02": {
        "wx": 1e-3,
        "wy": 1.1373125000000002,
        "a": 54.859375,
        "P": 0.9,
        "vmax": 0.5,
        "cdt": 30,
        "fps": 30
    },
    "MOT17-04": {
        "wx": 3.8977578124999996,
        "wy": 4.53134375,
        "a": 7.0,
        "P": 0.9,
        "vmax": 2.5782656250000002,
        "cdt": 30,
        "fps": 30
    },
    "MOT17-05": {
        "wx": 1.50596875,
        "wy": 0.18806250000000002,
        "a": 45.96484375,
        "P": 0.24375000000000002,
        "vmax": 1.624625,
        "cdt": 10,
        "fps": 14
    },
    "MOT17-09": {
        "wx": 0.8514921875,
        "wy": 2.171640625,
        "a": 18.71875,
        "P": 0.7125,
        "vmax": 0.921734375,
        "cdt": 30,
        "fps": 30
    },
    "MOT17-10": {
        "wx": 2.891046875,
        "wy": 4.0041054687500015,
        "a": 16.40625,
        "P": 1.0,
        "vmax": 0.001,
        "cdt": 10,
        "fps": 30
    },
    "MOT17-11": {
        "wx": 4.297015625,
        "wy": 4.12126953125,
        "a": 50.8984375,
        "P": 1.0,
        "vmax": 2.96485546875,
        "cdt": 10,
        "fps": 30
    },
}

if os.path.exists(out_path):
    shutil.rmtree(out_path)
# Vary cam noise
noise_degrees = np.linspace(-7, 7, 1)
results = []
for noise_degree in noise_degrees:
    args.add_cam_noise = noise_degree
    for seq in seqs:
        args.seq = seq
        seq_params = default_params[seq]
        args.wx = seq_params["wx"]
        args.wy = seq_params["wy"]
        args.a = seq_params["a"]
        args.P = seq_params["P"]
        args.vmax = seq_params["vmax"]
        args.cdt = seq_params["cdt"]
        args.fps = seq_params["fps"]

        run_ucmc(args, det_path, cam_path, gmc_path, out_path, exp_name, dataset)

    HOTA = eval_HOTA()
    results.append(HOTA)
