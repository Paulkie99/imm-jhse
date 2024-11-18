import json
import argparse
from functools import partial
import os
import shutil
import time
import numpy as np
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from scipy import interpolate
from detector.detector import KittiDetector as Detector
from eval_kitti import eval_HOTA as eval
from tracker.ucmc import UCMCTrack
from util.run_ucmc import Tracklet
from eval.interpolation import interpolate
import configparser

def make_args():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--seq', type=str, default = "MOT17-02", help='seq name')
    parser.add_argument('--fps', type=float, default=30.0, help='fps')
    parser.add_argument('--wx', type=float, default=5, help='wx')
    parser.add_argument('--wy', type=float, default=5, help='wy')
    parser.add_argument('--vmax', type=float, default=0.5, help='vmax')
    parser.add_argument('--a1', type=float, default=0.99, help='assignment threshold')
    parser.add_argument('--a2', type=float, default=0.99, help='assignment threshold')
    parser.add_argument('--a3', type=float, default=0.99, help='assignment threshold')
    parser.add_argument('--cdt', type=float, default=30.0, help='coasted deletion time')
    parser.add_argument('--high_score', type=float, default=0.6, help='high score threshold')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='detection confidence threshold')
    parser.add_argument("--cmc", action="store_true", help="use cmc or not.")
    parser.add_argument("--hp", action="store_true", help="use head padding or not.")
    parser.add_argument('--u_ratio', type=float, default=0.05, help='assignment threshold')
    parser.add_argument('--v_ratio', type=float, default=0.05, help='assignment threshold')
    parser.add_argument('--u_max', type=float, default=13, help='assignment threshold')
    parser.add_argument('--v_max', type=float, default=10, help='assignment threshold')
    parser.add_argument("--add_cam_noise", type=float, default=0, help="add noise to camera parameter.")
    parser.add_argument("--P", type=float, default=0)
    parser.add_argument("--t_m", type=float, default=100)
    parser.add_argument("--t1", type=float, default=0.9)
    parser.add_argument("--t2", type=float, default=0.9)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--sigma_m", type=float, default=0.05)
    parser.add_argument("--frame_width", type=float, default=1920)
    parser.add_argument("--frame_height", type=float, default=1080)
    args = parser.parse_args()


    g_u_ratio = args.u_ratio
    g_v_ratio = args.v_ratio
    g_u_max = args.u_max
    g_v_max = args.v_max

    print(f"u_ratio = {g_u_ratio}, v_ratio = {g_v_ratio}, u_max = {g_u_max}, v_max = {g_v_max}")


    return args



def run_param_search(x, sequences,
                     args, det_path = "det_results/mot17/yolox_x_ablation",
                        cam_path = "cam_para/mot17",
                        gmc_path = "gmc/mot17",
                        out_path = "output/mot17",
                        exp_name = "val",
                        dataset = "MOT17"):
    
    print(f"params: {x}")
    wx, wy, a1, a2, vmax, conf_thresh, high_score, cdt, b1, a3, t1, t2, ct1, ct2 = x

    config_parser = configparser.ConfigParser()

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    for seq in sequences:
        args.seq = seq
        seq_name = args.seq

        eval_path = os.path.join(out_path,exp_name)
        orig_save_path = os.path.join(eval_path,seq_name)
        if not os.path.exists(orig_save_path):
            os.makedirs(orig_save_path)

        det_file = os.path.join(det_path, f"{seq_name}.txt")
        cam_para = os.path.join(cam_path, f"{seq_name}.txt")
        result_file = os.path.join(orig_save_path,f"{seq_name}.txt")

        args.fps = 10
        args.frame_width = 1392
        args.frame_height = 512

        gmc_file = os.path.join(gmc_path, f"GMC-{seq_name}.txt")

        print(det_file)
        print(cam_para)

        detector = Detector(args.add_cam_noise, 1/args.fps)
        detector.load(cam_para, det_file,gmc_file,args.P)
        print(f"seq_length = {detector.seq_length}")

        a1 = a1
        a2 = a2
        fps = args.fps
        window = args.window

        tracker = UCMCTrack(a1, a2, wx,wy,vmax, cdt, fps, dataset, high_score,args.cmc,detector, args.t_m, b1=b1, t1=t1, t2=t2, window_len=window, a3=a3, alpha=0, ct1=ct1, ct2=ct2)

        timer = time.time()

        tracklets = dict()

        with open(result_file,"w") as f:
            for frame_id in range(1, detector.seq_length + 1):
                frame_affine = detector.gmc.get_affine(frame_id)
                dets = detector.get_dets(frame_id, conf_thresh, [0,1,2,3,4])
                tracker.update(dets,frame_affine)
                if args.hp:
                    for i in tracker.tentative_idx:
                        t = tracker.trackers[i]
                        if(t.detidx < 0 or t.detidx >= len(dets)):
                            continue
                        if t.id not in tracklets:
                            tracklets[t.id] = Tracklet(frame_id, dets[t.detidx].get_box(), dets[t.detidx].conf, dets[t.detidx].det_class)
                        else:
                            tracklets[t.id].add_box(frame_id, dets[t.detidx].get_box(), dets[t.detidx].conf)
                    for i in tracker.confirmed_idx:
                        t = tracker.trackers[i]
                        if(t.detidx < 0 or t.detidx >= len(dets)):
                            continue
                        if t.id not in tracklets:
                            tracklets[t.id] = Tracklet(frame_id, dets[t.detidx].get_box(), dets[t.detidx].conf, dets[t.detidx].det_class)
                        else:
                            tracklets[t.id].add_box(frame_id, dets[t.detidx].get_box(), dets[t.detidx].conf)
                        tracklets[t.id].activate()
                else:
                    for i in tracker.confirmed_idx:
                        t = tracker.trackers[i] 
                        if(t.detidx < 0 or t.detidx >= len(dets)):
                            continue
                        d = dets[t.detidx]
                        f.write(f"{frame_id},{t.id},{d.bb_left:.1f},{d.bb_top:.1f},{d.bb_width:.1f},{d.bb_height:.1f},{d.conf:.2f},{d.det_class},-1,-1\n")

            if args.hp:
                for frame_id in range(1, detector.seq_length + 1):
                    for id in tracklets:
                        if tracklets[id].is_active:
                            if frame_id in tracklets[id].boxes:
                                box = tracklets[id].boxes[frame_id]
                                f.write(f"{frame_id},{id},{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f},-1,{tracklets[id].det_class},-1,-1\n")
        interpolate(orig_save_path, eval_path, n_min=3, n_dti=cdt, is_enable = True, kitti=True)
        print(f"Time cost: {time.time() - timer:.2f}s")

    return eval(wx, wy, a1, vmax, out_path, exp_name)

def run_pattern_search(sequences, seq_params, det_path, cam_path, gmc_path, out_path, exp_name, dataset):
    args = make_args()

    print(seq_params)

    args.hp = True
    args.wx = seq_params["wx"]
    args.wy = seq_params["wy"]
    args.a1 = seq_params["a1"]
    args.a2 = seq_params["a2"]
    args.a3 = seq_params["a3"]
    args.vmax = seq_params["vmax"]
    args.fps = seq_params["fps"]
    args.cdt = seq_params["cdt"]
    args.P = seq_params["P"]
    args.b1 = seq_params["b1"]
    args.t_m = seq_params["t_m"]
    args.t1 = seq_params["t1"]
    args.t2 = seq_params["t2"]
    args.ct1 = seq_params["ct1"]
    args.ct2 = seq_params["ct2"]
    args.conf_thresh = seq_params["conf_thresh"]
    args.high_score = seq_params["high_score"]

    obj_func = partial(
        run_param_search,
        sequences=sequences,
        args=args,
        det_path=det_path, 
        cam_path=cam_path,
        gmc_path=gmc_path, 
        out_path=out_path, 
        exp_name=exp_name,
        dataset=dataset
    )

    obj = [
        obj_func
    ]
    n_var = 14

    problem = FunctionalProblem(
        n_var,
        obj,
        xl=np.array([0.001, 0.001, 0.1, 0.1,  0.001, 0.1, 0.1, 1, 0, 0.1, 0.01, 0.01, 0.01, 0.01]),
        xu=np.array([30,    30,    0.99,0.99,3,     0.9, 0.9, 100, 1, 0.99,0.99, 0.99,0.99, 0.99])
    )

    algorithm = PatternSearch(x0=np.array([args.wx, args.wy, args.a1, args.a2, args.vmax, args.conf_thresh, args.high_score, args.cdt, args.b1, args.a3, args.t1, args.t2, args.ct1, args.ct2]),
                              init_delta=0.75)

    class MyOutput(Output):

        def __init__(self):
            super().__init__()
            self.x = Column("x", width=13)
            self.F = Column("F", width=13)
            self.columns += [self.x, self.F]

        def update(self, algorithm):
            super().update(algorithm)
            self.x.set(algorithm.pop.get("X"))
            self.F.set(algorithm.pop.get("F"))

    res = minimize(problem, algorithm, 
                   get_termination("n_eval", 1),
                   output=MyOutput(),
                   verbose=True, seed=1)
    return {
        "wx": res.X[0],
        "wy": res.X[1], 
        "a1": res.X[2],
        "a2": res.X[3],
        "vmax": res.X[4],
        "conf_thresh": res.X[5],
        "high_score": res.X[6],
        "cdt": res.X[7],
        "b1": res.X[8],
        "a3": res.X[9],
        "t1": res.X[10],
        "t2": res.X[11],
        "ct1": res.X[12],
        "ct2": res.X[13],
        "OBJ": res.F[0]
    }

if __name__ == '__main__':
    # We run this file for one iteration with the parameters obtained on the KITTI training set. This is just a quick and dirty way of getting results files.
    det_path = "det_results/permatrack_kitti_test"
    cam_path = "cam_para/Kitti/testing/calib"
    gmc_path = "gmc/kitti/test"
    out_path = "output_overall_hota_test/kitti"
    exp_name = "test"
    dataset = "Kitti"

    sequences = os.listdir(det_path)
    sequences = sorted([seq.split('.')[0] for seq in sequences])

    default_params = {
      "wx": 14.843421875000002,
      "wy": 16.249625,
      "a1": 0.83375,
      "a2": 0.5,
      "vmax": 0.28215625,
      "conf_thresh": 0.5,
      "high_score": 0.29999999999999993,
      "cdt": 100.0,
      "b1": 0.0,
      "a3": 0.9065624999999999,
      "t1": 0.9,
      "t2": 0.9,
      "ct1": 0.71625,
      "ct2": 0.71625,
        "t_m": 2,
        "P": -6,
        "fps": 30
    }

    results = run_pattern_search(sequences, default_params, det_path, cam_path, gmc_path, out_path, exp_name, dataset)

    print(results)
    out_file = open(f"cv_overall_hota_test_param_search_results_{dataset}_{exp_name}.json", "w")   
    json.dump(results, out_file, indent = 6) 
    out_file.close() 
