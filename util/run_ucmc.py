import numpy as np
from detector.detector import Detector
from tracker.ucmc import UCMCTrack
from eval.interpolation import interpolate
import os,time
import cv2
import argparse
import configparser
import json

class Tracklet():
    def __init__(self,frame_id,box,conf=None,det_class=None):
        self.is_active = False
        self.boxes = dict()
        self.confs = dict()
        self.boxes[frame_id] = box
        if conf is not None:
            self.confs[frame_id] = conf
            self.det_class = det_class

    def add_box(self, frame_id, box, conf=None):
        self.boxes[frame_id] = box
        if conf is not None:
            self.confs[frame_id] = conf

    def activate(self):
        self.is_active = True


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
    parser.add_argument("--ct1", type=float, default=0.9)
    parser.add_argument("--ct2", type=float, default=0.9)
    parser.add_argument("--b1", type=float, default=0.3)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--sigma_m", type=float, default=0.05)
    parser.add_argument("--frame_width", type=float, default=1920)
    parser.add_argument("--frame_height", type=float, default=1080)
    parser.add_argument("--param_file", type=str, default='')
    parser.add_argument("--video", action="store_true")
    
    args = parser.parse_args()


    g_u_ratio = args.u_ratio
    g_v_ratio = args.v_ratio
    g_u_max = args.u_max
    g_v_max = args.v_max

    print(f"u_ratio = {g_u_ratio}, v_ratio = {g_v_ratio}, u_max = {g_u_max}, v_max = {g_v_max}")


    return args

def run_ucmc(args, det_path = "det_results/mot17/yolox_x_ablation",
                   cam_path = "cam_para/mot17",
                   gmc_path = "gmc/mot17",
                   out_path = "output/mot17",
                   exp_name = "val",
                   dataset = "MOT17"):
    
    if args.seq == 'all':
        args.seq = os.listdir(det_path)
        args.seq = [seq.split('.')[0] for seq in args.seq]
        print(args.seq)
        if dataset == "MOT17":
            args.seq = [seq.split('-SDP')[0] for seq in args.seq]
    else:
        args.seq = [args.seq]

    params = None
    if args.param_file != '':
        params = json.load(open(args.param_file, 'r'))

    for seq in args.seq:
        seq_name = seq

        if params is not None:
            if seq in params:
                print(params[seq])
                args.wx = params[seq]['wx']
                args.wy = params[seq]['wy']
                args.a1 = params[seq]['a1']
                args.a2 = params[seq]['a2']
                args.vmax = params[seq]['vmax']
                args.t_m = params[seq].get('t_m', args.t_m)
                args.conf_thresh = params[seq]['conf_thresh']
                args.high_score = params[seq]['high_score']
                args.cdt = params[seq]["cdt"]
                args.P = params[seq].get("P", args.P)
                args.b1 = params[seq].get("b1", args.b1)
                args.a3 = params[seq].get("a3", args.a3)
                args.t1 = params[seq].get('t1', args.t1)
                args.t2 = params[seq].get('t2', args.t2)
                args.ct1 = params[seq].get('ct1', args.t1)
                args.ct2 = params[seq].get('ct2', args.t2)
            else:
                print(params)
                args.wx = params['wx']
                args.wy = params['wy']
                args.a1 = params['a1']
                args.a2 = params['a2']
                args.vmax = params['vmax']
                args.t_m = params.get('t_m', args.t_m)
                args.conf_thresh = params['conf_thresh']
                args.high_score = params['high_score']
                args.cdt = params["cdt"]
                args.P = params.get("P", args.P)
                args.b1 = params.get("b1", args.b1)
                args.a3 = params.get("a3", args.a3)
                args.t1 = params.get('t1', args.t1)
                args.t2 = params.get('t2', args.t2)
                args.ct1 = params.get('ct1', args.t1)
                args.ct2 = params.get('ct2', args.t2)

        eval_path = os.path.join(out_path,exp_name)
        orig_save_path = os.path.join(eval_path,seq_name)
        if not os.path.exists(orig_save_path):
            os.makedirs(orig_save_path)


        if dataset == "MOT17":
            det_file = os.path.join(det_path, f"{seq_name}-SDP.txt")
            cam_para = os.path.join(cam_path, f"{seq_name}-SDP.txt")
            result_file = os.path.join(orig_save_path,f"{seq_name}-SDP.txt")
        else:
            det_file = os.path.join(det_path, f"{seq_name}.txt")
            cam_para = os.path.join(cam_path, f"{seq_name}.txt")
            result_file = os.path.join(orig_save_path,f"{seq_name}.txt")

        gmc_file = os.path.join(gmc_path, f"GMC-{seq_name}.txt")

        config = configparser.ConfigParser()
        config.read(f"data/{dataset}/{'train' if exp_name == 'val' and dataset == 'MOT17' else exp_name}/{seq_name}{'-SDP' if 'MOT17' in dataset else ''}/seqinfo.ini")
        args.fps = float(config['Sequence']['frameRate'])
        args.frame_width = float(config['Sequence']['imWidth'])
        args.frame_height = float(config['Sequence']['imHeight'])
        args.hp = True

        print(det_file)
        print(cam_para)

        detector = Detector(args.add_cam_noise, 1/args.fps)
        detector.load(cam_para, det_file,gmc_file,args.P,sigma_m=args.sigma_m)
        print(f"seq_length = {detector.seq_length}")

        a1 = args.a1
        a2 = args.a2
        high_score = args.high_score
        conf_thresh = args.conf_thresh
        fps = args.fps
        cdt = args.cdt
        wx = args.wx
        wy = args.wy
        vmax = args.vmax
        
        tracker = UCMCTrack(a1, a2, wx,wy,vmax, cdt, fps, dataset, high_score,args.cmc,detector, t_m=args.t_m, b1=args.b1, t1=args.t1, t2=args.t2, a3=args.a3, ct1=args.ct1, ct2=args.ct2)

        t1 = time.time()

        tracklets = dict()

        if args.video:
            video_out = cv2.VideoWriter(f'{orig_save_path}/{seq_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(args.frame_width), int(args.frame_height)))

        with open(result_file,"w") as f:
            for frame_id in range(1, detector.seq_length + 1):
                if args.video:
                    frame_img = cv2.imread(f"data/{dataset}/{'train' if exp_name == 'val' and dataset == 'MOT17' else exp_name}/{seq_name}{'-SDP' if 'MOT' in dataset else ''}/img1/{str(frame_id).zfill(6 if 'MOT' in dataset else 8)}.jpg")
                frame_affine = detector.gmc.get_affine(frame_id)
                dets = detector.get_dets(frame_id, conf_thresh, 1 if "oracle" in det_path else 0)
                tracker.update(dets,frame_affine)
                if args.hp:
                    for i in tracker.tentative_idx:
                        t = tracker.trackers[i]
                        if(t.detidx < 0 or t.detidx >= len(dets)):
                            continue
                        if t.id not in tracklets:
                            tracklets[t.id] = Tracklet(frame_id, dets[t.detidx].get_box())
                        else:
                            tracklets[t.id].add_box(frame_id, dets[t.detidx].get_box())
                        
                        if args.video:
                            cv2.rectangle(frame_img, (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top)), (int(dets[t.detidx].bb_left+dets[t.detidx].bb_width), int(dets[t.detidx].bb_top+dets[t.detidx].bb_height)), (0, 255, 0), 2)
                            cv2.putText(frame_img, str(np.round(t.boxmu, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            # cv2.putText(frame_img, str(np.round(t.relative_iou, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            # cv2.putText(frame_img, str(np.round(t.g_mahala, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(t.id), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.circle(frame_img, tuple(t.uv), 5, (255, 0, 0), -1)

                    for i in tracker.coasted_idx:
                        t = tracker.trackers[i]
                        try:
                            if args.video:
                                cv2.rectangle(frame_img, (int(t.box_pred[0]), int(t.box_pred[1])), (int(t.box_pred[2]), int(t.box_pred[3])), (0, 0, 255), 2)
                                cv2.putText(frame_img, str(np.round(t.boxmu, 2)), (int(t.box_pred[0]), int(t.box_pred[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                # cv2.putText(frame_img, str(np.round(t.relative_iou, 2)), (int(t.box_pred[0]), int(t.box_pred[1]) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                # cv2.putText(frame_img, str(np.round(t.g_mahala, 2)), (int(t.box_pred[0]), int(t.box_pred[1]) + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.putText(frame_img, str(t.id), (int(t.box_pred[0]), int(t.box_pred[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.circle(frame_img, tuple(t.uv), 5, (0, 0, 255), -1)
                        except:
                            pass

                    for i in tracker.confirmed_idx:
                        t = tracker.trackers[i]
                        if(t.detidx < 0 or t.detidx >= len(dets)):
                            continue
                        if t.id not in tracklets:
                            tracklets[t.id] = Tracklet(frame_id, dets[t.detidx].get_box())
                        else:
                            tracklets[t.id].add_box(frame_id, dets[t.detidx].get_box())
                        tracklets[t.id].activate()

                        if args.video:
                            cv2.rectangle(frame_img, (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top)), (int(dets[t.detidx].bb_left+dets[t.detidx].bb_width), int(dets[t.detidx].bb_top+dets[t.detidx].bb_height)), (0, 255, 0), 2)
                            cv2.putText(frame_img, str(np.round(t.boxmu, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            # cv2.putText(frame_img, str(np.round(t.relative_iou, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            # cv2.putText(frame_img, str(np.round(t.g_mahala, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(t.id), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.circle(frame_img, tuple(t.uv), 5, (255, 0, 0), -1)
                            major_axis = int(t.eigenvalues[0] * 2)
                            minor_axis = int(t.eigenvalues[1] * 2)
                            angle = np.rad2deg(np.arccos(t.eigenvectors[0, 0]))
                            cv2.ellipse(frame_img, tuple(t.uv), (major_axis, minor_axis), angle, 0, 360, (255, 255, 255), 2)

                            cv2.putText(frame_img, str(round(tracker.track_position_bias[i], 2)), (int(dets[t.detidx].bb_left+dets[t.detidx].bb_width), int(dets[t.detidx].bb_top+dets[t.detidx].bb_height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                else:
                    for i in tracker.confirmed_idx:
                        t = tracker.trackers[i] 
                        if(t.detidx < 0 or t.detidx >= len(dets)):
                            continue
                        d = dets[t.detidx]
                        f.write(f"{frame_id},{t.id},{d.bb_left:.1f},{d.bb_top:.1f},{d.bb_width:.1f},{d.bb_height:.1f},{d.conf:.2f},-1,-1,-1\n")
                        
                        if args.video:
                            cv2.rectangle(frame_img, (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top)), (int(dets[t.detidx].bb_left+dets[t.detidx].bb_width), int(dets[t.detidx].bb_top+dets[t.detidx].bb_height)), (0, 255, 0), 2)
                            cv2.putText(frame_img, str(np.round(t.boxmu, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(np.round(t.relative_iou, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(np.round(t.g_mahala, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(t.id), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            cv2.circle(frame_img, tuple(t.uv), 5, (255, 0, 0), -1)
        
                # frame_img[:homog_resize_dim, -homog_resize_dim:] = cv2.resize(homog_img, (homog_resize_dim, homog_resize_dim))
                # if frame_id % 69 == 0:
                #     print()
                if args.video:
                    video_out.write(frame_img)
                # homog_out.write(homog_img)
            if args.hp:
                for frame_id in range(1, detector.seq_length + 1):
                    for id in tracklets:
                        if tracklets[id].is_active:
                            if frame_id in tracklets[id].boxes:
                                box = tracklets[id].boxes[frame_id]
                                f.write(f"{frame_id},{id},{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f},-1,-1,-1,-1\n")

        interpolate(orig_save_path, eval_path, n_min=3, n_dti=cdt, is_enable = True)
        if args.video:
            video_out.release()
        # homog_out.release()
        print(f"Time cost: {time.time() - t1:.2f}s")

