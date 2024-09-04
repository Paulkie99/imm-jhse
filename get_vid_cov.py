import os
import numpy as np
from detector.detector import Detector

dataset = "data/DanceTrack/val"
det_path = "det_results/dance/val"
cam_path = "cam_para/DanceTrack"
gmc_path = "gmc/dance"

# videos = os.listdir(dataset)
videos = ["dancetrack0007"]
print(videos)

homogs = []

for vid_name in videos:

    save_file = open(f"gmc/GMC-{vid_name}.txt", 'w')

    det_file = os.path.join(det_path, f"{vid_name}.txt")
    cam_para = os.path.join(cam_path, f"{vid_name}.txt")
    gmc_file = os.path.join(gmc_path, f"GMC-{vid_name}.txt")

    detector = Detector(dt = 1/20)
    detector.load(cam_para, det_file, gmc_file)

    orig_homog = detector.mapper.A

    for frame_num in range(1, detector.seq_length + 1):
        frame_affine = detector.gmc.get_affine(frame_num)
        augmented_affine = np.r_[frame_affine, [[0, 0, 1]]]

        pred_h = augmented_affine @ orig_homog
        diff = pred_h - orig_homog

        homogs.append(diff.T.flatten()[:-1])

    homogs = np.array(homogs)
    cov = homogs.T @ homogs / homogs.shape[0]

    print(cov)
    np.save(f"pnoise/{vid_name}", cov)
