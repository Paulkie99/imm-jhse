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
    def __init__(self,frame_id,box):
        self.is_active = False
        self.boxes = dict()
        self.boxes[frame_id] = box

    def add_box(self, frame_id, box):
        self.boxes[frame_id] = box

    def activate(self):
        self.is_active = True


def make_args():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--seq', type=str, default = "MOT17-02", help='seq name')
    parser.add_argument('--fps', type=float, default=30.0, help='fps')
    parser.add_argument('--wx', type=float, default=0.001, help='wx')
    parser.add_argument('--wy', type=float, default=5, help='wy')
    parser.add_argument('--vmax', type=float, default=2.5, help='vmax')
    parser.add_argument('--a1', type=float, default=0.99, help='assignment threshold')
    parser.add_argument('--a2', type=float, default=0.7, help='assignment threshold')
    parser.add_argument('--cdt', type=float, default=30.0, help='coasted deletion time')
    parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='detection confidence threshold')
    parser.add_argument("--cmc", action="store_true", help="use cmc or not.")
    parser.add_argument("--hp", action="store_true", help="use head padding or not.")
    parser.add_argument('--u_ratio', type=float, default=0.05, help='assignment threshold')
    parser.add_argument('--v_ratio', type=float, default=0.05, help='assignment threshold')
    parser.add_argument('--u_max', type=float, default=13, help='assignment threshold')
    parser.add_argument('--v_max', type=float, default=10, help='assignment threshold')
    parser.add_argument("--add_cam_noise", type=float, default=0, help="add noise to camera parameter.")
    parser.add_argument("--axis", type=str, default="z", help="add noise to camera parameter.")
    parser.add_argument("--P", type=float, default=-32)
    parser.add_argument("--t_m", type=float, default=2)
    parser.add_argument("--t1", type=float, default=0.9)
    parser.add_argument("--t2", type=float, default=0.5)
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
    else:
        args.seq = [args.seq]

    params = None
    if args.param_file != '':
        params = json.load(open(args.param_file, 'r'))

    for seq in args.seq:
        seq_name = seq

        if params is not None:
            args.wx = params[seq]['wx']
            args.wy = params[seq]['wy']
            args.a1 = params[seq]['a1']
            args.a2 = params[seq]['a2']
            args.vmax = params[seq]['vmax']
            args.t_m = params[seq]['t_m']
            args.conf_thresh = params[seq]['conf_thresh']
            args.high_score = params[seq]['high_score']
            args.t1 = params[seq]['t1']
            args.t2 = params[seq]['t2']

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
        config.read(f"data/{dataset}/{'train' if exp_name == 'val' and dataset == 'MOT17' else exp_name}/{seq_name}{'-SDP' if 'MOT' in dataset else ''}/seqinfo.ini")
        args.fps = float(config['Sequence']['frameRate'])
        args.frame_width = float(config['Sequence']['imWidth'])
        args.frame_height = float(config['Sequence']['imHeight'])
        args.hp = True

        print(det_file)
        print(cam_para)

        detector = Detector(args.add_cam_noise, args.frame_width, args.frame_height, 1/args.fps, args.axis)
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
        
        tracker = UCMCTrack(a1, a2, wx,wy,vmax, cdt, fps, dataset, high_score,args.cmc,detector,t_m=args.t_m, t1=args.t1, t2=args.t2)

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
                tracker.update(dets,frame_id,frame_affine)
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
                            cv2.putText(frame_img, str(np.round(t.mu, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(np.round(t.relative_iou, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(np.round(t.g_mahala, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(dets[t.detidx].track_id), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.circle(frame_img, tuple(t.uv), 5, (0, 0, 255), -1)

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
                            cv2.putText(frame_img, str(np.round(t.mu, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(np.round(t.relative_iou, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(np.round(t.g_mahala, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(dets[t.detidx].track_id), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.circle(frame_img, tuple(t.uv), 5, (0, 0, 255), -1)
                else:
                    for i in tracker.confirmed_idx:
                        t = tracker.trackers[i] 
                        if(t.detidx < 0 or t.detidx >= len(dets)):
                            continue
                        d = dets[t.detidx]
                        f.write(f"{frame_id},{t.id},{d.bb_left:.1f},{d.bb_top:.1f},{d.bb_width:.1f},{d.bb_height:.1f},{d.conf:.2f},-1,-1,-1\n")
                        
                        if args.video:
                            cv2.rectangle(frame_img, (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top)), (int(dets[t.detidx].bb_left+dets[t.detidx].bb_width), int(dets[t.detidx].bb_top+dets[t.detidx].bb_height)), (0, 255, 0), 2)
                            cv2.putText(frame_img, str(np.round(t.mu, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(np.round(t.relative_iou, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(np.round(t.g_mahala, 2)), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top) + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                            cv2.putText(frame_img, str(dets[t.detidx].track_id), (int(dets[t.detidx].bb_left), int(dets[t.detidx].bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.circle(frame_img, tuple(t.uv), 5, (0, 0, 255), -1)
                # frame_img[:homog_resize_dim, -homog_resize_dim:] = cv2.resize(homog_img, (homog_resize_dim, homog_resize_dim))
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

