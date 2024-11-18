# One Homography is All You Need: IMM-JHSE

> **[Information Fusion 2024] One Homography is All You Need: IMM-based Joint Homography and Multiple Object State Estimation**.

[![arXiv](https://img.shields.io/badge/arXiv-2409.02562-<COLOR>.svg)](https://arxiv.org/abs/2409.02562)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/interacting-multiple-model-based-joint/multiple-object-tracking-on-kitti-test-online)](https://paperswithcode.com/sota/multiple-object-tracking-on-kitti-test-online?p=interacting-multiple-model-based-joint)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/interacting-multiple-model-based-joint/multi-object-tracking-on-dancetrack)](https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=interacting-multiple-model-based-joint)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/interacting-multiple-model-based-joint/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=interacting-multiple-model-based-joint)

This repo was adapted from [UCMCTrack](https://github.com/corfyi/UCMCTrack). It is still under construction!

DanceTrack example:
![Loading DanceTrack example...](dance_example.gif)

## Setting Up the Conda Environment

To create a new conda environment with the required dependencies, follow these steps:

1. Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

2. Open a terminal or command prompt.

3. Navigate to the directory containing the `environment.yml` file.

4. Run the following command to create a new conda environment:

    ```sh
    conda env create -f environment.yml
    ```

5. Activate the newly created environment:

    ```sh
    conda activate <environment_name>
    ```

    Replace `<environment_name>` with the name specified in the `environment.yml` file.

## Run IMM-JHSE on the DanceTrack test set

1. Download the [DanceTrack testing and validation sets](https://huggingface.co/datasets/noahcao/dancetrack/tree/main).

2. Extract the contents to maintain the following folder structure:
      ~~~
      {IMM-JHSE ROOT}
      |-- data
            |-- DanceTrack
            |   |-- val
            |   |   |-- dancetrack0004
            |   |   |   |-- img1
            |   |   |   |   |-- 00000001.jpg
            |   |   |   |   |-- ...
            |   |   |   |-- gt
            |   |   |   |   |-- gt.txt            
            |   |   |   |-- seqinfo.ini
            |   |   |-- ...
            |   |-- test
            |   |   |-- ...
      ~~~

3.  Run the following to get results with video in the `test_output/dance/test` folder:
    ```sh
    run_dance_test.py --seq all --param_file dancetrack_params.json --video
    ```
    You can omit the --video flag to perform inference without video output.

## Test results

The result files used to obtain the results reported in the paper for the test sets are given in the `result_files` folder.
Submit these to the evaluation servers for the various test datasets.

## Citation

Please cite our article if you use this repo for further research:
```json
@misc{claasen2024homographyneedimmbasedjoint,
      title={One Homography is All You Need: IMM-based Joint Homography and Multiple Object State Estimation}, 
      author={Paul Johannes Claasen and Johan Pieter de Villiers},
      year={2024},
      eprint={2409.02562},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.02562}, 
}
```
