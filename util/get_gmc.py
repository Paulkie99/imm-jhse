import os
import cv2
import gmc

dataset = "data/DanceTrack/val"
videos = os.listdir(dataset)
print(videos)

for vid_name in videos:
    local_gmc = gmc.GMC()

    save_file = open(f"gmc/GMC-{vid_name}.txt", 'w')

    for frame_num, frame_name in enumerate(sorted(os.listdir(f"{dataset}/{vid_name}/img1"))):
        print(frame_name)
        img = cv2.imread(f"{dataset}/{vid_name}/img1/{frame_name}")
        affine = local_gmc.apply(img)

        save_file.write(f"{frame_num} {affine[0, 0]} {affine[0, 1]} {affine[0, 2]} {affine[1, 0]} {affine[1, 1]} {affine[1, 2]}\n")

    save_file.close()
