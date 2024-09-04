
import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from ..utils import TrackEvalException
from .. import _timing

import csv
import io
import zipfile
import os
import traceback
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from .. import _timing
from ..utils import TrackEvalException


class _BaseKitti(ABC):
    @abstractmethod
    def __init__(self):
        self.tracker_list = None
        self.seq_list = None
        self.class_list = None
        self.output_fol = None
        self.output_sub_fol = None
        self.should_classes_combine = True
        self.use_super_categories = False

    # Functions to implement:

    @staticmethod
    @abstractmethod
    def get_default_dataset_config():
        ...

    @abstractmethod
    def _load_raw_file(self, tracker, seq, is_gt):
        ...

    @_timing.time
    @abstractmethod
    def get_preprocessed_seq_data(self, raw_data, cls):
        ...

    @abstractmethod
    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        ...

    # Helper functions for all datasets:

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    def get_name(self):
        return self.get_class_name()

    def get_output_fol(self, tracker):
        return os.path.join(self.output_fol, tracker, self.output_sub_fol)

    def get_display_name(self, tracker):
        """ Can be overwritten if the trackers name (in files) is different to how it should be displayed.
        By default this method just returns the trackers name as is.
        """
        return tracker

    def get_eval_info(self):
        """Return info about the dataset needed for the Evaluator"""
        return self.tracker_list, self.seq_list, self.class_list

    @_timing.time
    def get_raw_seq_data(self, tracker, seq):
        """ Loads raw data (tracker and ground-truth) for a single tracker on a single sequence.
        Raw data includes all of the information needed for both preprocessing and evaluation, for all classes.
        A later function (get_processed_seq_data) will perform such preprocessing and extract relevant information for
        the evaluation of each class.

        This returns a dict which contains the fields:
        [num_timesteps]: integer
        [gt_ids, tracker_ids, gt_classes, tracker_classes, tracker_confidences]:
                                                                list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, tracker_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [similarity_scores]: list (for each timestep) of 2D NDArrays.
        [gt_extras]: dict (for each extra) of lists (for each timestep) of 1D NDArrays (for each det).

        gt_extras contains dataset specific information used for preprocessing such as occlusion and truncation levels.

        Note that similarities are extracted as part of the dataset and not the metric, because almost all metrics are
        independent of the exact method of calculating the similarity. However datasets are not (e.g. segmentation
        masks vs 2D boxes vs 3D boxes).
        We calculate the similarity before preprocessing because often both preprocessing and evaluation require it and
        we don't wish to calculate this twice.
        We calculate similarity between all gt and tracker classes (not just each class individually) to allow for
        calculation of metrics such as class confusion matrices. Typically the impact of this on performance is low.
        """
        # Load raw data.
        raw_gt_data = self._load_raw_file(tracker, seq, is_gt=True)
        raw_tracker_data = self._load_raw_file(tracker, seq, is_gt=False)
        raw_data = {**raw_tracker_data, **raw_gt_data}  # Merges dictionaries

        # Calculate similarities for each timestep.
        similarity_scores = []
        for t, (gt_dets_t, tracker_dets_t) in enumerate(zip(raw_data['gt_dets'], raw_data['tracker_dets'])):
            ious = self._calculate_similarities(gt_dets_t, tracker_dets_t)
            similarity_scores.append(ious)
        raw_data['similarity_scores'] = similarity_scores
        return raw_data

    @staticmethod
    def _load_simple_text_file(file, time_col=0, id_col=None, remove_negative_ids=False, valid_filter=None,
                               crowd_ignore_filter=None, convert_filter=None, is_zipped=False, zip_file=None,
                               force_delimiters=None):
        """ Function that loads data which is in a commonly used text file format.
        Assumes each det is given by one row of a text file.
        There is no limit to the number or meaning of each column,
        however one column needs to give the timestep of each det (time_col) which is default col 0.

        The file dialect (deliminator, num cols, etc) is determined automatically.
        This function automatically separates dets by timestep,
        and is much faster than alternatives such as np.loadtext or pandas.

        If remove_negative_ids is True and id_col is not None, dets with negative values in id_col are excluded.
        These are not excluded from ignore data.

        valid_filter can be used to only include certain classes.
        It is a dict with ints as keys, and lists as values,
        such that a row is included if "row[key].lower() is in value" for all key/value pairs in the dict.
        If None, all classes are included.

        crowd_ignore_filter can be used to read crowd_ignore regions separately. It has the same format as valid filter.

        convert_filter can be used to convert value read to another format.
        This is used most commonly to convert classes given as string to a class id.
        This is a dict such that the key is the column to convert, and the value is another dict giving the mapping.

        Optionally, input files could be a zip of multiple text files for storage efficiency.

        Returns read_data and ignore_data.
        Each is a dict (with keys as timesteps as strings) of lists (over dets) of lists (over column values).
        Note that all data is returned as strings, and must be converted to float/int later if needed.
        Note that timesteps will not be present in the returned dict keys if there are no dets for them
        """

        if remove_negative_ids and id_col is None:
            raise TrackEvalException('remove_negative_ids is True, but id_col is not given.')
        if crowd_ignore_filter is None:
            crowd_ignore_filter = {}
        if convert_filter is None:
            convert_filter = {}
        try:
            if is_zipped:  # Either open file directly or within a zip.
                if zip_file is None:
                    raise TrackEvalException('is_zipped set to True, but no zip_file is given.')
                archive = zipfile.ZipFile(os.path.join(zip_file), 'r')
                fp = io.TextIOWrapper(archive.open(file, 'r'))
            else:
                fp = open(file)
            read_data = {}
            crowd_ignore_data = {}
            fp.seek(0, os.SEEK_END)
            # check if file is empty
            if fp.tell():
                fp.seek(0)
                dialect = csv.Sniffer().sniff(fp.readline(), delimiters=force_delimiters)  # Auto determine structure.
                dialect.skipinitialspace = True  # Deal with extra spaces between columns
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    try:
                        # Deal with extra trailing spaces at the end of rows
                        if row[-1] in '':
                            row = row[:-1]
                        timestep = str(int(float(row[time_col])))
                        # Read ignore regions separately.
                        is_ignored = False
                        for ignore_key, ignore_value in crowd_ignore_filter.items():
                            if row[ignore_key].lower() in ignore_value:
                                # Convert values in one column (e.g. string to id)
                                for convert_key, convert_value in convert_filter.items():
                                    row[convert_key] = convert_value[row[convert_key].lower()]
                                # Save data separated by timestep.
                                if timestep in crowd_ignore_data.keys():
                                    crowd_ignore_data[timestep].append(row)
                                else:
                                    crowd_ignore_data[timestep] = [row]
                                is_ignored = True
                        if is_ignored:  # if det is an ignore region, it cannot be a normal det.
                            continue
                        # Exclude some dets if not valid.
                        if valid_filter is not None:
                            for key, value in valid_filter.items():
                                if row[key].lower() not in value:
                                    continue
                        if remove_negative_ids:
                            if int(float(row[id_col])) < 0:
                                continue
                        # Convert values in one column (e.g. string to id)
                        for convert_key, convert_value in convert_filter.items():
                            row[convert_key] = convert_value[row[convert_key].lower()]
                        # Save data separated by timestep.
                        if timestep in read_data.keys():
                            read_data[timestep].append(row)
                        else:
                            read_data[timestep] = [row]
                    except Exception:
                        exc_str_init = 'In file %s the following line cannot be read correctly: \n' % os.path.basename(
                            file)
                        exc_str = ' '.join([exc_str_init]+row)
                        raise TrackEvalException(exc_str)
            fp.close()
        except Exception:
            print('Error loading file: %s, printing traceback.' % file)
            traceback.print_exc()
            raise TrackEvalException(
                'File %s cannot be read because it is either not present or invalidly formatted' % os.path.basename(
                    file))
        return read_data, crowd_ignore_data

    @staticmethod
    def _calculate_mask_ious(masks1, masks2, is_encoded=False, do_ioa=False):
        """ Calculates the IOU (intersection over union) between two arrays of segmentation masks.
        If is_encoded a run length encoding with pycocotools is assumed as input format, otherwise an input of numpy
        arrays of the shape (num_masks, height, width) is assumed and the encoding is performed.
        If do_ioa (intersection over area) , then calculates the intersection over the area of masks1 - this is commonly
        used to determine if detections are within crowd ignore region.
        :param masks1:  first set of masks (numpy array of shape (num_masks, height, width) if not encoded,
                        else pycocotools rle encoded format)
        :param masks2:  second set of masks (numpy array of shape (num_masks, height, width) if not encoded,
                        else pycocotools rle encoded format)
        :param is_encoded: whether the input is in pycocotools rle encoded format
        :param do_ioa: whether to perform IoA computation
        :return: the IoU/IoA scores
        """

        # Only loaded when run to reduce minimum requirements
        from pycocotools import mask as mask_utils

        # use pycocotools for run length encoding of masks
        if not is_encoded:
            masks1 = mask_utils.encode(np.array(np.transpose(masks1, (1, 2, 0)), order='F'))
            masks2 = mask_utils.encode(np.array(np.transpose(masks2, (1, 2, 0)), order='F'))

        # use pycocotools for iou computation of rle encoded masks
        ious = mask_utils.iou(masks1, masks2, [do_ioa]*len(masks2))
        if len(masks1) == 0 or len(masks2) == 0:
            ious = np.asarray(ious).reshape(len(masks1), len(masks2))
        assert (ious >= 0 - np.finfo('float').eps).all()
        assert (ious <= 1 + np.finfo('float').eps).all()

        return ious

    @staticmethod
    def _calculate_box_ious(bboxes1, bboxes2, box_format='xywh', do_ioa=False):
        """ Calculates the IOU (intersection over union) between two arrays of boxes.
        Allows variable box formats ('xywh' and 'x0y0x1y1').
        If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
        used to determine if detections are within crowd ignore region.
        """
        if box_format in 'xywh':
            # layout: (x0, y0, w, h)
            bboxes1 = deepcopy(bboxes1)
            bboxes2 = deepcopy(bboxes2)

            bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
            bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
            bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
            bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
        elif box_format not in 'x0y0x1y1':
            raise (TrackEvalException('box_format %s is not implemented' % box_format))

        # layout: (x0, y0, x1, y1)
        min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

        if do_ioa:
            ioas = np.zeros_like(intersection)
            valid_mask = area1 > 0 + np.finfo('float').eps
            ioas[valid_mask, :] = intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]

            return ioas
        else:
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
            union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
            intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
            intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
            intersection[union <= 0 + np.finfo('float').eps] = 0
            union[union <= 0 + np.finfo('float').eps] = 1
            ious = intersection / union
            return ious

    @staticmethod
    def _calculate_euclidean_similarity(dets1, dets2, zero_distance=2.0):
        """ Calculates the euclidean distance between two sets of detections, and then converts this into a similarity
        measure with values between 0 and 1 using the following formula: sim = max(0, 1 - dist/zero_distance).
        The default zero_distance of 2.0, corresponds to the default used in MOT15_3D, such that a 0.5 similarity
        threshold corresponds to a 1m distance threshold for TPs.
        """
        dist = np.linalg.norm(dets1[:, np.newaxis]-dets2[np.newaxis, :], axis=2)
        sim = np.maximum(0, 1 - dist/zero_distance)
        return sim

    @staticmethod
    def _check_unique_ids(data, after_preproc=False):
        """Check the requirement that the tracker_ids and gt_ids are unique per timestep"""
        gt_ids = data['gt_ids']
        tracker_ids = data['tracker_ids']
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
            if len(tracker_ids_t) > 0:
                unique_ids, counts = np.unique(tracker_ids_t, return_counts=True)
                if np.max(counts) != 1:
                    duplicate_ids = unique_ids[counts > 1]
                    exc_str_init = 'Tracker predicts the same ID more than once in a single timestep ' \
                                   '(seq: %s, frame: %i, ids:' % (data['seq'], t+1)
                    exc_str = ' '.join([exc_str_init] + [str(d) for d in duplicate_ids]) + ')'
                    if after_preproc:
                        exc_str_init += '\n Note that this error occurred after preprocessing (but not before), ' \
                                        'so ids may not be as in file, and something seems wrong with preproc.'
                    raise TrackEvalException(exc_str)
            if len(gt_ids_t) > 0:
                unique_ids, counts = np.unique(gt_ids_t, return_counts=True)
                if np.max(counts) != 1:
                    duplicate_ids = unique_ids[counts > 1]
                    exc_str_init = 'Ground-truth has the same ID more than once in a single timestep ' \
                                   '(seq: %s, frame: %i, ids:' % (data['seq'], t+1)
                    exc_str = ' '.join([exc_str_init] + [str(d) for d in duplicate_ids]) + ')'
                    if after_preproc:
                        exc_str_init += '\n Note that this error occurred after preprocessing (but not before), ' \
                                        'so ids may not be as in file, and something seems wrong with preproc.'
                    raise TrackEvalException(exc_str)


class Kitti2DBox(_BaseKitti):
    """Dataset class for KITTI 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/kitti/kitti_2d_box_train'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/kitti/kitti_2d_box_train/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['car', 'pedestrian'],  # Valid: ['car', 'pedestrian']
            'SPLIT_TO_EVAL': 'training',  # Valid: 'training', 'val', 'training_minus_val', 'test'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.gt_fol = self.config['GT_FOLDER']
        self.tracker_fol = self.config['TRACKERS_FOLDER']
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config['INPUT_AS_ZIP']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        self.max_occlusion = 2
        self.max_truncation = 0
        self.min_height = 25

        # Get classes to eval
        self.valid_classes = ['car', 'pedestrian']
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise TrackEvalException('Attempted to evaluate an invalid class. Only classes [car, pedestrian] are valid.')
        self.class_name_to_class_id = {'car': 1, 'van': 2, 'truck': 3, 'pedestrian': 4, 'person': 5,  # person sitting
                                       'cyclist': 6, 'tram': 7, 'misc': 8, 'dontcare': 9, 'car_2': 1}

        # Get sequences to eval and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}
        seqmap_name = 'evaluate_tracking.seqmap.' + self.config['SPLIT_TO_EVAL']
        seqmap_file = os.path.join(self.gt_fol, seqmap_name)
        if not os.path.isfile(seqmap_file):
            raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
        with open(seqmap_file) as fp:
            dialect = csv.Sniffer().sniff(fp.read(1024))
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for row in reader:
                if len(row) >= 4:
                    seq = row[0]
                    self.seq_list.append(seq)
                    self.seq_lengths[seq] = int(row[3])
                    if not self.data_is_zipped:
                        curr_file = os.path.join(self.gt_fol, 'label_02', seq + '.txt')
                        if not os.path.isfile(curr_file):
                            raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))
            if self.data_is_zipped:
                curr_file = os.path.join(self.gt_fol, 'data.zip')
                if not os.path.isfile(curr_file):
                    raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
                if not os.path.isfile(curr_file):
                    raise TrackEvalException('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                    if not os.path.isfile(curr_file):
                        raise TrackEvalException(
                            'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
                                curr_file))

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the kitti 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(self.gt_fol, 'data.zip')
            else:
                zip_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
            file = seq + '.txt'
        else:
            zip_file = None
            if is_gt:
                file = os.path.join(self.gt_fol, 'label_02', seq + '.txt')
            else:
                file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')

        # Ignore regions
        if is_gt:
            crowd_ignore_filter = {2: ['dontcare']}
        else:
            crowd_ignore_filter = None

        # Valid classes
        valid_filter = {2: [x for x in self.class_list]}
        if is_gt:
            if 'car' in self.class_list:
                valid_filter[2].append('van')
            if 'pedestrian' in self.class_list:
                valid_filter[2] += ['person']

        # Convert kitti class strings to class ids
        convert_filter = {2: self.class_name_to_class_id}

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, time_col=0, id_col=1, remove_negative_ids=True,
                                                             valid_filter=valid_filter,
                                                             crowd_ignore_filter=crowd_ignore_filter,
                                                             convert_filter=convert_filter,
                                                             is_zipped=self.data_is_zipped, zip_file=zip_file)
        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str(t) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = 'Ground-truth'
            else:
                text = 'Tracking'
            raise TrackEvalException(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys]))

        for t in range(num_timesteps):
            time_key = str(t)
            if time_key in read_data.keys():
                time_data = np.asarray(read_data[time_key], dtype=float)
                raw_data['dets'][t] = np.atleast_2d(time_data[:, 6:10])
                raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                raw_data['classes'][t] = np.atleast_1d(time_data[:, 2]).astype(int)
                if is_gt:
                    gt_extras_dict = {'truncation': np.atleast_1d(time_data[:, 3].astype(int)),
                                      'occlusion': np.atleast_1d(time_data[:, 4].astype(int))}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    if time_data.shape[1] > 17:
                        raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 17])
                    else:
                        raw_data['tracker_confidences'][t] = np.ones(time_data.shape[0])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'truncation': np.empty(0),
                                      'occlusion': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                if time_key in ignore_data.keys():
                    time_ignore = np.asarray(ignore_data[time_key], dtype=float)
                    raw_data['gt_crowd_ignore_regions'][t] = np.atleast_2d(time_ignore[:, 6:10])
                else:
                    raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        KITTI:
            In KITTI, the 4 preproc steps are as follow:
                1) There are two classes (pedestrian and car) which are evaluated separately.
                2) For the pedestrian class, the 'person' class is distractor objects (people sitting).
                    For the car class, the 'van' class are distractor objects.
                    GT boxes marked as having occlusion level > 2 or truncation level > 0 are also treated as
                        distractors.
                3) Crowd ignore regions are used to remove unmatched detections. Also unmatched detections with
                    height <= 25 pixels are removed.
                4) Distractor gt dets (including truncated and occluded) are removed.
        """
        if cls == 'pedestrian':
            distractor_classes = [self.class_name_to_class_id['person']]
        elif cls == 'car':
            distractor_classes = [self.class_name_to_class_id['van']]
        else:
            raise (TrackEvalException('Class %s is not evaluatable' % cls))
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Only extract relevant dets for this class for preproc and eval (cls + distractor classes)
            gt_class_mask = np.sum([raw_data['gt_classes'][t] == c for c in [cls_id] + distractor_classes], axis=0)
            gt_class_mask = gt_class_mask.astype(bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]
            gt_classes = raw_data['gt_classes'][t][gt_class_mask]
            gt_occlusion = raw_data['gt_extras'][t]['occlusion'][gt_class_mask]
            gt_truncation = raw_data['gt_extras'][t]['truncation'][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            tracker_confidences = raw_data['tracker_confidences'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
            # which are labeled as truncated, occluded, or belonging to a distractor class.
            to_remove_matched = np.array([], int)
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                is_occluded_or_truncated = np.logical_or(
                    gt_occlusion[match_rows] > self.max_occlusion + np.finfo('float').eps,
                    gt_truncation[match_rows] > self.max_truncation + np.finfo('float').eps)
                to_remove_matched = np.logical_or(is_distractor_class, is_occluded_or_truncated)
                to_remove_matched = match_cols[to_remove_matched]
                unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

            # For unmatched tracker dets, also remove those smaller than a minimum height.
            unmatched_tracker_dets = tracker_dets[unmatched_indices, :]
            unmatched_heights = unmatched_tracker_dets[:, 3] - unmatched_tracker_dets[:, 1]
            is_too_small = unmatched_heights <= self.min_height + np.finfo('float').eps

            # For unmatched tracker dets, also remove those that are greater than 50% within a crowd ignore region.
            crowd_ignore_regions = raw_data['gt_crowd_ignore_regions'][t]
            intersection_with_ignore_region = self._calculate_box_ious(unmatched_tracker_dets, crowd_ignore_regions,
                                                                       box_format='x0y0x1y1', do_ioa=True)
            is_within_crowd_ignore_region = np.any(intersection_with_ignore_region > 0.5 + np.finfo('float').eps, axis=1)

            # Apply preprocessing to remove all unwanted tracker dets.
            to_remove_unmatched = unmatched_indices[np.logical_or(is_too_small, is_within_crowd_ignore_region)]
            to_remove_tracker = np.concatenate((to_remove_matched, to_remove_unmatched), axis=0)
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Also remove gt dets that were only useful for preprocessing and are not needed for evaluation.
            # These are those that are occluded, truncated and from distractor objects.
            gt_to_keep_mask = (np.less_equal(gt_occlusion, self.max_occlusion)) & \
                              (np.less_equal(gt_truncation, self.max_truncation)) & \
                              (np.equal(gt_classes, cls_id))
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')
        return similarity_scores