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
from detector.detector import Detector
from eval_mot17 import eval_AssA as eval_MOT17
from eval_dance import eval_AssA as eval_Dance
from tracker.ucmc import UCMCTrack
from util.run_ucmc import Tracklet, make_args
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
    parser.add_argument('--a2', type=float, default=0.7, help='assignment threshold')
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
    parser.add_argument("--P", type=float, default=-32)
    parser.add_argument("--t_m", type=float, default=2)
    parser.add_argument("--t1", type=float, default=0.9)
    parser.add_argument("--t2", type=float, default=0.9)
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



def run_param_search(x,
                     args, det_path = "det_results/mot17/yolox_x_ablation",
                        cam_path = "cam_para/mot17",
                        gmc_path = "gmc/mot17",
                        out_path = "output/mot17",
                        exp_name = "val",
                        dataset = "MOT17"):
    
    print(f"params: {x}")
    wx, wy, a1, a2, vmax, t_m, conf_thresh, high_score, t1, t2 = x

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    seq_name = args.seq
    config_parser = configparser.ConfigParser()

    eval_path = os.path.join(out_path,exp_name)
    orig_save_path = os.path.join(eval_path,seq_name)
    if not os.path.exists(orig_save_path):
        os.makedirs(orig_save_path)

    if dataset == "MOT17":
        det_file = os.path.join(det_path, f"{seq_name}-SDP.txt")
        cam_para = os.path.join(cam_path, f"{seq_name}-SDP.txt")
        result_file = os.path.join(orig_save_path,f"{seq_name}-SDP.txt")
        config_parser.read(f"data/MOT17/train/{seq_name}-SDP/seqinfo.ini")
    else:
        det_file = os.path.join(det_path, f"{seq_name}.txt")
        cam_para = os.path.join(cam_path, f"{seq_name}.txt")
        result_file = os.path.join(orig_save_path,f"{seq_name}.txt")
        config_parser.read(f"data/{dataset}/{exp_name}/{seq_name}/seqinfo.ini")
    
    args.fps = float(config_parser['Sequence']['frameRate'])
    args.frame_width = float(config_parser['Sequence']['imWidth'])
    args.frame_height = float(config_parser['Sequence']['imHeight'])
    print(args.fps)

    gmc_file = os.path.join(gmc_path, f"GMC-{seq_name}.txt")

    print(det_file)
    print(cam_para)

    detector = Detector(args.add_cam_noise, args.frame_width, args.frame_height, 1/args.fps)
    detector.load(cam_para, det_file,gmc_file,args.P)
    print(f"seq_length = {detector.seq_length}")

    a1 = a1
    a2 = a2
    # high_score = args.high_score
    fps = args.fps
    # vmax = args.vmax
    cdt = args.cdt
    # conf_thresh = args.conf_thresh

    tracker = UCMCTrack(a1, a2, wx,wy,vmax, cdt, fps, dataset, high_score,args.cmc,detector, t_m, t1=t1, t2=t2)

    t1 = time.time()

    tracklets = dict()

    with open(result_file,"w") as f:
        for frame_id in range(1, detector.seq_length + 1):
            frame_affine = detector.gmc.get_affine(frame_id)
            dets = detector.get_dets(frame_id, conf_thresh, 1 if "oracle" in det_path else 0)
            frametime = time.time()
            try:
                tracker.update(dets,frame_id,frame_affine)
            except Exception as e:
                print(e)
                if os.path.exists(out_path):
                    shutil.rmtree(out_path)
                return 0
            if time.time() - frametime >= 0.2:
                print("Aborting optimistation iteration")
                return 0
            if args.hp:
                for i in tracker.tentative_idx:
                    t = tracker.trackers[i]
                    if(t.detidx < 0 or t.detidx >= len(dets)):
                        continue
                    if t.id not in tracklets:
                        tracklets[t.id] = Tracklet(frame_id, dets[t.detidx].get_box())
                    else:
                        tracklets[t.id].add_box(frame_id, dets[t.detidx].get_box())
                for i in tracker.confirmed_idx:
                    t = tracker.trackers[i]
                    if(t.detidx < 0 or t.detidx >= len(dets)):
                        continue
                    if t.id not in tracklets:
                        tracklets[t.id] = Tracklet(frame_id, dets[t.detidx].get_box())
                    else:
                        tracklets[t.id].add_box(frame_id, dets[t.detidx].get_box())
                    tracklets[t.id].activate()
            else:
                for i in tracker.confirmed_idx:
                    t = tracker.trackers[i] 
                    if(t.detidx < 0 or t.detidx >= len(dets)):
                        continue
                    d = dets[t.detidx]
                    f.write(f"{frame_id},{t.id},{d.bb_left:.1f},{d.bb_top:.1f},{d.bb_width:.1f},{d.bb_height:.1f},{d.conf:.2f},-1,-1,-1\n")

        if args.hp:
            for frame_id in range(1, detector.seq_length + 1):
                for id in tracklets:
                    if tracklets[id].is_active:
                        if frame_id in tracklets[id].boxes:
                            box = tracklets[id].boxes[frame_id]
                            f.write(f"{frame_id},{id},{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f},-1,-1,-1,-1\n")
    try:
        interpolate(orig_save_path, eval_path, n_min=3, n_dti=cdt, is_enable = True)
    except Exception as e:
        print(e)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        return 0
    print(f"Time cost: {time.time() - t1:.2f}s")

    return eval_MOT17(wx, wy, a1, vmax) if "MOT" in dataset else eval_Dance(wx, wy, a1, vmax, out_path, exp_name)

def run_pattern_search(seq, seq_params, det_path, cam_path, gmc_path, out_path, exp_name, dataset):
    args = make_args()

    print(seq_params)

    args.seq = seq
    args.hp = True
    args.wx = seq_params["wx"]
    args.wy = seq_params["wy"]
    args.a1 = seq_params["a1"]
    args.a2 = seq_params["a2"]
    args.vmax = seq_params["vmax"]
    args.fps = seq_params["fps"]
    args.cdt = seq_params["cdt"]
    args.P = seq_params["P"]

    obj_func = partial(
        run_param_search,
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
    n_var = 10

    # vars
    # wx, wy, a1, a2, vmax, t_m, conf_thresh, high_score, t1, t2
    problem = FunctionalProblem(
        n_var,
        obj,
        xl=np.array([0.001, 0.001, 0.1, 0.1, 0.001, 1/30, 0.1, 0.1, 0.01, 0.01]),
        xu=np.array([30,    30,    0.999,0.999,   3, 100, 0.9, 0.9, 0.99, 0.99])
    )
    # problem = FunctionalProblem(
    #     n_var,
    #     obj,
    #     xl=np.array([1,  0]),
    #     xu=np.array([100,1])
    # )

    algorithm = PatternSearch(x0=np.array([args.wx, args.wy, args.a1, args.a2, args.vmax, args.t_m, args.conf_thresh, args.high_score, args.t1, args.t2]),
                              init_delta=0.75)
    # algorithm = PatternSearch(x0=np.array([args.a, args.P]),
    #                           init_delta=0.5)

    # algorithm = RNSGA3(
    #                     ref_points=np.array([[args.wx, args.wy, args.vmax, args.a, args.cdt, args.conf_thresh, 0.5]]).T,
    #                     pop_per_ref_point=1,
    #                     mu=0.1
    #                     )

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
                   get_termination("n_eval", 10 * n_var),
                   output=MyOutput(),
                   verbose=True, seed=1)
    # print(f"Best solution: \nwx={res.X[0]}\nwy={res.X[1]}\na={res.X[2]}\nvmax={res.X[3]}\P={res.X[4]}\nOBJ={res.F}")
    return {
        "wx": res.X[0],
        "wy": res.X[1], 
        "a1": res.X[2],
        "a2": res.X[3],
        "vmax": res.X[4],
        "t_m": res.X[5],
        "conf_thresh": res.X[6],
        "high_score": res.X[7],
        "t1": res.X[8],
        "t2": res.X[9],
        "OBJ": res.F[0]
    }

if __name__ == '__main__':
    det_path = "det_results/dance/val"#"det_results/mot17/yolox_x_ablation"
    cam_path = "cam_para/DanceTrack"#"cam_para/MOT17"
    gmc_path = "gmc/dance"#"gmc/mot17"
    out_path = "output_per/dance"#"output/mot17"
    exp_name = "val"
    dataset = "DanceTrack"#"MOT17"

    # sequences = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
    sequences = os.listdir(det_path)
    sequences = [seq.split('.')[0] for seq in sequences]

    default_params = {
        seq: {
            "wx": 5,
            "wy": 5,
            "a1": 0.99,
            "a2": 0.7,
            "P": -32,
            "vmax": 0.5,
            "cdt": 30,
            "fps": 30
        }
        for seq in sequences
    }
    results = {}
    for seq in sequences:
        results[seq] = run_pattern_search(seq, default_params[seq], det_path, cam_path, gmc_path, out_path, exp_name, dataset)

    print(results)
    out_file = open(f"param_search_results_{dataset}_{exp_name}.json", "w") 
  
    json.dump(results, out_file, indent = 6) 
    
    out_file.close() 