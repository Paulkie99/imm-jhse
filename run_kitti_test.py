from util.run_ucmc_kitti import run_ucmc, make_args

if __name__ == '__main__':

    det_path = "det_results/permatrack_kitti_test"
    cam_path = "cam_para/Kitti/testing/calib"
    gmc_path = "gmc/kitti/test"
    out_path = "kitti_test/kitti"
    exp_name = "test"
    dataset = "Kitti"
    args = make_args()

    run_ucmc(args, det_path, cam_path, gmc_path, out_path, exp_name,dataset)
