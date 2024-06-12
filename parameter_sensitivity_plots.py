import os
import shutil
import matplotlib.pyplot as plt
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
        "wy": 0.522201171875,
        "a": 9.455078125,
        "P": -4.4296875,
        "vmax": 0.6230058593750002,
        "sigma_m": 0.05509765625,
        "cdt": 30,
        "fps": 30
    },
    "MOT17-04": {
        "wx": 0.6736406250000001,
        "wy": 0.44141796875,
        "a": 7.0,
        "P": -9,
        "vmax": 0.28215625,
        "sigma_m": 0.07718749999999999,
        "cdt": 30,
        "fps": 30
    },
    "MOT17-05": {
        "wx": 0.041417968750000006,
        "wy": 2.4141796875,
        "a": 15.44921875,
        "P": -1.109375,
        "vmax": 0.1221962890625,
        "sigma_m": 0.056796875000000004,
        "cdt": 10,
        "fps": 14
    },
    "MOT17-09": {
        "wx": 0.470708984375,
        "wy": 2.874625,
        "a": 8.08984375,
        "P": -9,
        "vmax": 2.415376953125,
        "sigma_m": 0.05,
        "cdt": 30,
        "fps": 30
    },
    "MOT17-10": {
        "wx": 2.1880625,
        "wy": 1.1335859375,
        "a": 7,
        "P": -8,
        "vmax": 1.1677460937500002,
        "sigma_m": 0.056796875000000004,
        "cdt": 10,
        "fps": 30
    },
    "MOT17-11": {
        "wx": 1.30933203125,
        "wy": 1.1335859375,
        "a": 7,
        "P": 1.0,
        "vmax": 1.09745703125,
        "sigma_m": 0.27089843750000003,
        "cdt": 10,
        "fps": 30
    },
    "MOT17-13": {
        "wx": 1e-3,
        "wy": 1.7194062499999996,
        "a": 44.359375,
        "P": -5.34375,
        "vmax": 1.2731796874999994,
        "sigma_m": 0.05,
        "cdt": 10,
        "fps": 25
    }
}

if os.path.exists(out_path):
    shutil.rmtree(out_path)
# Vary cam noise
# noise_degrees = np.linspace(0, 4, 9)
# print(noise_degrees)
# for axis in ['z', 'y', 'x']:
#     args.axis = axis
#     results = []
#     for noise_degree in noise_degrees:
#         args.add_cam_noise = noise_degree
#         for seq in seqs:
#             args.seq = seq
#             seq_params = default_params[seq]
#             args.wx = seq_params["wx"]
#             args.wy = seq_params["wy"]
#             args.a = seq_params["a"]
#             args.P = seq_params["P"]
#             args.vmax = seq_params["vmax"]
#             args.cdt = seq_params["cdt"]
#             args.fps = seq_params["fps"]
#             args.sigma_m = seq_params["sigma_m"]

#             run_ucmc(args, det_path, cam_path, gmc_path, out_path, exp_name, dataset)

#         HOTA = eval_HOTA()
#         results.append(float(HOTA))
#     plt.plot(noise_degrees, results, label=f"{axis}")
# plt.title("HOTA vs camera error degree")
# plt.xlabel("Error degree")
# plt.ylabel("HOTA")
# plt.legend()
# plt.savefig("cam_error_sens.png")

sigma_ms = np.linspace(0.05, 0.3, 6)
print(sigma_ms)
results = []
for sigma_m in sigma_ms:
    args.sigma_m = sigma_m
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
    results.append(float(HOTA))
plt.plot(sigma_ms, results)
ax = plt.gca()
ax.set_ylim([55, 85])
plt.title("HOTA vs $\\sigma_m$")
plt.xlabel("$\\sigma_m$")
plt.ylabel("HOTA")
plt.savefig("sigma_m_sens.png")
