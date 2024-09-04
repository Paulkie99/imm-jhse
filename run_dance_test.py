from util.run_ucmc import run_ucmc, make_args

if __name__ == '__main__':

    det_path = "det_results/dance/test"
    cam_path = "cam_para/DanceTrack"
    gmc_path = "gmc/dance"
    out_path = "test_output/dance"
    exp_name = "test"
    dataset = "DanceTrack"
    args = make_args()

    run_ucmc(args, det_path, cam_path, gmc_path, out_path, exp_name,dataset)
