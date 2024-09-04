
import argparse
import sys
import os
from multiprocessing import freeze_support
import numpy as np

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .TrackEval import trackeval  # noqa: E402


def eval(dataset,eval_dir,seqmap,exp_name,fps_div,half_eval = False):
    freeze_support()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

    # TRACKERS_FOLDER = os.getcwd()+"\\eval\\"

    eval_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 1,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': True,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': 'None',  # if not None, save any errors into a log file.
            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': False,
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': False,
            'DISPLAY_LESS_PROGRESS': True,
            'OUTPUT_SUMMARY': True,
            'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
            'OUTPUT_DETAILED': False,
            'PLOT_CURVES': False,
        }
    
    if half_eval == False:
        gt_format = '{gt_folder}/{seq}/gt/gt.txt'
    elif fps_div == 1:
        gt_format = '{gt_folder}/{seq}/gt/gt_val_half.txt'
    else:
        gt_format = '{gt_folder}/{seq}/gt/gt_1_'+f'{fps_div}.txt'

    dataset_config = {
            'GT_FOLDER': dataset,  # Location of GT data
            'TRACKERS_FOLDER': eval_dir,  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': [exp_name],  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
            'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
            'SPLIT_TO_EVAL': 'val',  # Valid: 'train', 'test', 'all'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': False,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'TRACKER_SUB_FOLDER': '',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': seqmap,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            'GT_LOC_FORMAT': gt_format,  
            'SKIP_SPLIT_FOL': True
        }


    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    
    return output_res['summary'][0]['HOTA'], output_res['summary'][2]['IDF1'], output_res['summary'][1]['MOTA'], output_res['summary'][0]['AssA']

def eval_kitti(dataset,eval_dir,seqmap,exp_name):
    freeze_support()
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    default_dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    dataset_config['GT_FOLDER'] = dataset
    dataset_config['TRACKERS_FOLDER'] = eval_dir
    dataset_config['TRACKERS_TO_EVAL'] = [exp_name]
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    dataset_config['OUTPUT_SUB_FOLDER'] = ''
    # dataset_config['SPLIT_TO_EVAL'] = 'val'
    # dataset_config['TRACKER_DISPLAY_NAMES'] = None
    # dataset_config['SEQMAP_FOLDER'] = None
    # dataset_config['SEQMAP_FILE'] = seqmap
    # dataset_config['SEQ_INFO'] = None

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.Kitti2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    return output_res['summary'][0]['HOTA'], output_res['summary'][2]['IDF1'], output_res['summary'][1]['MOTA'], output_res['summary'][0]['AssA']
    
    return (
        output_res['Kitti2DBox']['val']['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA'].mean() * 50 + output_res['Kitti2DBox']['val']['COMBINED_SEQ']['car']['HOTA']['HOTA'].mean() * 50,
        50 * output_res['Kitti2DBox']['val']['COMBINED_SEQ']['pedestrian']['Identity']['IDF1'] + 50 * output_res['Kitti2DBox']['val']['COMBINED_SEQ']['car']['Identity']['IDF1'], 
        50 * output_res['Kitti2DBox']['val']['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA'] + 50 * output_res['Kitti2DBox']['val']['COMBINED_SEQ']['car']['CLEAR']['MOTA'], 
        output_res['Kitti2DBox']['val']['COMBINED_SEQ']['pedestrian']['HOTA']['AssA'].mean() * 50 + output_res['Kitti2DBox']['val']['COMBINED_SEQ']['car']['HOTA']['AssA'].mean() * 50,
    )


def eval_bdd(gt_folder,exp_name, eval_classes):

    freeze_support()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR','Identity']}

    TRACKERS_FOLDER = os.getcwd()+"\\eval\\"

    eval_config = {
            'USE_PARALLEL': True,
            'NUM_PARALLEL_CORES': 16,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': 'None',  # if not None, save any errors into a log file.
            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': True,
            'PRINT_CONFIG': False,
            'TIME_PROGRESS': False,
            'DISPLAY_LESS_PROGRESS': True,
            'OUTPUT_SUMMARY': True,
            'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
            'OUTPUT_DETAILED': False,
            'PLOT_CURVES': False,
        }

    dataset_config = {
        'GT_FOLDER': gt_folder,  # Location of GT data
        'TRACKERS_FOLDER': TRACKERS_FOLDER,  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': [exp_name],  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': eval_classes,
        # Valid: ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle']
        'SPLIT_TO_EVAL': 'val',  # Valid: 'training', 'val',
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
        'PRINT_CONFIG': False,  # Whether to print current config
        'TRACKER_SUB_FOLDER': '',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
    }

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.BDD100K(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    # HOTA,IDF1,MOTA,AssA = output_res['summary'][0]['HOTA'], output_res['summary'][2]['IDF1'], output_res['summary'][1]['MOTA'], output_res['summary'][0]['AssA']

    res = next(iter(output_res['BDD100K'].values()))
    HOTA = res['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['HOTA']
    AssA = res['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['AssA']
    MOTA = res['COMBINED_SEQ']['cls_comb_cls_av']['CLEAR']['MOTA']*100
    IDF1 = res['COMBINED_SEQ']['cls_comb_cls_av']['Identity']['IDF1']*100
    HOTA = np.mean(HOTA)*100
    AssA = np.mean(AssA)*100

    #保留小数点后3位
    HOTA = round(HOTA,3)
    AssA = round(AssA,3)
    MOTA = round(MOTA,3)
    IDF1 = round(IDF1,3)

    return HOTA,IDF1,MOTA,AssA