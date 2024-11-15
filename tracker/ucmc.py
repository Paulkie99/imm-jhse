
from __future__ import print_function

import numpy as np
from lap import lapjv
import scipy


from .kalman import CVHIMM, KalmanTrackerBox,TrackStatus


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

class UCMCTrack(object):
    def __init__(self,a1,a2,wx, wy,vmax, max_age, fps, dataset, high_score, use_cmc,detector = None,t_m=5.7470703125, b1=0.3, b2=0.4, t1=0.9, t2=0.9, window_len=5, a3=0.5, alpha=0, ct1=0.9, ct2=0.9):
        CVHIMM.count = 1

        self.wx = wx
        self.wy = wy
        self.vmax = vmax
        self.dataset = dataset
        self.high_score = high_score
        self.max_age = max_age
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.dt = 1.0/fps
        self.t_m = t_m
        self.b1 = b1
        self.b2 = b2
        self.t1 = t1
        self.t2 = t2
        self.ct1 = ct1
        self.ct2 = ct2
        self.window_len = window_len

        self.use_cmc = use_cmc

        self.trackers = []
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []
        self.detector = detector

        self.alpha = alpha

    def update(self, dets,frame_affine): 
        camdists = []
        for track in self.trackers:
            track.predict(frame_affine)
            camdists.append(track.camdist)
        if len(camdists):
            min_dist = min(camdists)
            max_dist = max(camdists)
            camdists = np.array(camdists)
            self.track_position_bias = np.exp(-self.alpha*(camdists - min_dist)/max_dist)

        self.data_association(dets)
        
        self.associate_tentative(dets)

        self.initial_tentative(dets, self.detector.mapper.A.T.flatten()[:-1], self.detector.mapper.covariance,
        self.detector.mapper.process_covariance)
        
        self.delete_old_trackers()

        self.update_status(dets)
    
    def data_association(self, dets):
        # Separate detections into high score and low score
        detidx_high = []
        detidx_low = []
        for i in range(len(dets)):
            if dets[i].conf >= self.high_score:
                detidx_high.append(i)
            else:
                detidx_low.append(i)
        # detidx_high = list(np.arange(len(dets)))
        trackidx_remain = []
        self.detidx_remain = []
        detidx_remain = []

        # Associate high score detections with tracks
        trackidx = self.confirmed_idx + self.coasted_idx
        num_det = len(detidx_high)
        num_trk = len(trackidx)

        if num_det*num_trk > 0:
            cost_matrix = np.zeros((num_det, num_trk))
            relative_ious = np.zeros((num_det, num_trk))
            relative_ps = np.zeros((num_det, num_trk))

            det_ys = [dets[det_idx].y for det_idx in detidx_high]
            det_covs = [dets[det_idx].R for det_idx in detidx_high]
            det_confs = np.array([dets[det_idx].conf for det_idx in detidx_high])
            det_classes = np.array([dets[det_idx].det_class for det_idx in self.detidx_remain])

            for j in range(num_trk):
                trk_idx = trackidx[j]
                _, relative_ious[:,j], relative_ps[:,j] = self.trackers[trk_idx].distance(det_ys, det_covs, self.b1, det_classes)
                # cost_matrix[:, j] = _
                # cbar = np.dot(self.trackers[trk_idx].boxlikelihood, self.trackers[trk_idx].boxM)
                cost_matrix[:, j] = 1 - relative_ps[:, j] * relative_ious[:, j] * self.track_position_bias[trk_idx] * det_confs
                # cost_matrix[:, j] = 1 - (relative_ious[:, j] * relative_ps[:, j]) * self.track_position_bias[trk_idx] * det_confs
            
            matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix, self.a1)

            # relative_ious = relative_ious / relative_ious.sum(axis=1)[:, None]
            # relative_ps = relative_ps / relative_ps.sum(axis=1)[:, None]
            
            for i in unmatched_a:
                detidx_remain.append(detidx_high[i])
            for i in unmatched_b:
                trackidx_remain.append(trackidx[i])
            
            for i,j in matched_indices:
                det_idx = detidx_high[i]
                trk_idx = trackidx[j]
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R, relative_ious[i,j],
                                              relative_ps[i,j])
                self.trackers[trk_idx].death_count = 0
                self.trackers[trk_idx].detidx = det_idx
                # self.trackers[trk_idx].R = dets[det_idx].R
                self.trackers[trk_idx].status = TrackStatus.Confirmed
                dets[det_idx].track_id = self.trackers[trk_idx].id

        else:
            detidx_remain = detidx_high
            trackidx_remain = trackidx 
        
        # Associate low score detections with remain tracks
        detidx_low += detidx_remain
        num_det = len(detidx_low)
        num_trk = len(trackidx_remain)
        if num_det*num_trk > 0:
            cost_matrix = np.zeros((num_det, num_trk))
            relative_ious = np.zeros((num_det, num_trk))
            relative_ps = np.zeros((num_det, num_trk))

            det_ys = [dets[det_idx].y for det_idx in detidx_low]
            det_covs = [dets[det_idx].R for det_idx in detidx_low]
            det_confs = np.array([dets[det_idx].conf for det_idx in detidx_low])
            det_classes = np.array([dets[det_idx].det_class for det_idx in self.detidx_remain])
            for j in range(num_trk):
                trk_idx = trackidx_remain[j]
                _, relative_ious[:, j], relative_ps[:, j] = self.trackers[trk_idx].distance(det_ys, det_covs, self.b1, det_classes)
                cbar = self.trackers[trk_idx].boxcbar
                cost_matrix[:, j] = 1 - (cbar[0] * relative_ious[:, j] + cbar[1] * relative_ps[:, j]) * self.track_position_bias[trk_idx] * det_confs
                # cost_matrix[:, j] = 1 - relative_ps[:, j] * relative_ious[:, j] * self.track_position_bias[trk_idx] * det_confs

                # cbar = np.dot(self.trackers[trk_idx].boxlikelihood, self.trackers[trk_idx].boxM)
                # cost_matrix[:, j] = 1 - cbar[0] * relative_ious[:, j] * self.track_position_bias[trk_idx] * det_confs
                # cost_matrix[:, j] = _

            matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix,self.a2)

            # relative_ious = relative_ious / relative_ious.sum(axis=1)[:, None]
            # relative_ps = relative_ps / relative_ps.sum(axis=1)[:, None]

            for i in unmatched_a:
                if dets[detidx_low[i]].conf >= self.high_score:
                    self.detidx_remain.append(detidx_low[i])
            
            for i in unmatched_b:
                trk_idx = trackidx_remain[i]
                self.trackers[trk_idx].status = TrackStatus.Coasted
                # self.trackers[trk_idx].death_count += 1
                self.trackers[trk_idx].detidx = -1

            for i,j in matched_indices:
                det_idx = detidx_low[i]
                trk_idx = trackidx_remain[j]
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R, relative_ious[i,j],
                                relative_ps[i,j])
                      
                self.trackers[trk_idx].death_count = 0
                self.trackers[trk_idx].detidx = det_idx
                # self.trackers[trk_idx].R = dets[det_idx].R
                self.trackers[trk_idx].status = TrackStatus.Confirmed
                dets[det_idx].track_id = self.trackers[trk_idx].id
        else:
            for trk_idx in trackidx_remain:
                self.trackers[trk_idx].detidx = -1
                self.trackers[trk_idx].status = TrackStatus.Coasted
            self.detidx_remain = [idx for idx in detidx_remain if dets[idx].conf >= self.high_score]

    def associate_tentative(self, dets):
        num_det = len(self.detidx_remain)
        num_trk = len(self.tentative_idx)

        cost_matrix = np.zeros((num_det, num_trk))
        relative_ious = np.zeros((num_det, num_trk))
        relative_ps = np.zeros((num_det, num_trk))

        det_ys = [dets[det_idx].y for det_idx in self.detidx_remain]
        det_covs = [dets[det_idx].R for det_idx in self.detidx_remain]
        det_confs = np.array([dets[det_idx].conf for det_idx in self.detidx_remain])
        det_classes = np.array([dets[det_idx].det_class for det_idx in self.detidx_remain])
        if len(det_ys):
            for j in range(num_trk):
                trk_idx = self.tentative_idx[j]
                _, relative_ious[:, j], relative_ps[:,j] = self.trackers[trk_idx].distance(det_ys, det_covs, self.b1, det_classes)
                cbar = self.trackers[trk_idx].boxcbar
                cost_matrix[:, j] = 1 - (cbar[0] * relative_ious[:, j] + cbar[1] * relative_ps[:, j]) * self.track_position_bias[trk_idx] * det_confs
                # cost_matrix[:, j] = 1 - relative_ps[:, j] * relative_ious[:, j] * self.track_position_bias[trk_idx] * det_confs
                # cbar = np.dot(self.trackers[trk_idx].boxlikelihood, self.trackers[trk_idx].boxM)
                # cost_matrix[:, j] = 1 - cbar[1] * relative_ps[:, j] * self.track_position_bias[trk_idx]
                # cost_matrix[:, j] = _

        matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix,self.a3)

        # relative_ious = relative_ious / relative_ious.sum(axis=1)[:, None]
        # relative_ps = relative_ps / relative_ps.sum(axis=1)[:, None]

        for i,j in matched_indices:
            det_idx = self.detidx_remain[i]
            trk_idx = self.tentative_idx[j]
            self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R, relative_ious[i,j],
                                relative_ps[i,j])
            self.trackers[trk_idx].death_count = 0
            self.trackers[trk_idx].birth_count += 1
            self.trackers[trk_idx].detidx = det_idx
            # self.trackers[trk_idx].R = dets[det_idx].R
            dets[det_idx].track_id = self.trackers[trk_idx].id
            if self.trackers[trk_idx].birth_count >= 2:
                self.trackers[trk_idx].birth_count = 0
                self.trackers[trk_idx].status = TrackStatus.Confirmed

        unmatched_trks = []
        for i in unmatched_b:
            trk_idx = self.tentative_idx[i]
            unmatched_trks.append(trk_idx)
            self.trackers[trk_idx].detidx = -1

        unmatched_detidx = []
        for i in unmatched_a:
            unmatched_detidx.append(self.detidx_remain[i])

        # num_det = len(unmatched_detidx)
        # num_trk = len(unmatched_trks)
        # if num_det * num_trk > 0:
        #     cost_matrix = np.zeros((num_det, num_trk))
        #     relative_ious = np.zeros((num_det, num_trk))
        #     relative_ps = np.zeros((num_det, num_trk))
        #     det_ys = [dets[det_idx].y for det_idx in unmatched_detidx]
        #     det_covs = [dets[det_idx].R for det_idx in unmatched_detidx]
        #     det_confs = np.array([dets[det_idx].conf for det_idx in unmatched_detidx])
        #     if len(det_ys):
        #         for j in range(num_trk):
        #             trk_idx = unmatched_trks[j]
        #             _, relative_ious[:, j], relative_ps[:,j] = self.trackers[trk_idx].distance(det_ys, det_covs, self.b1)
        #             # cbar = np.dot(self.trackers[trk_idx].likelihood, self.trackers[trk_idx].M)
        #             cost_matrix[:, j] = 1 - self.trackers[trk_idx].cbar[1] * relative_ious[:, j] #* det_confs
                
        #     matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix,self.a2)

        #     for i,j in matched_indices:
        #         det_idx = unmatched_detidx[i]
        #         trk_idx = unmatched_trks[j]
        #         self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R, relative_ious[i,j],
        #                             relative_ps[i,j])
        #         self.trackers[trk_idx].death_count = 0
        #         self.trackers[trk_idx].birth_count += 1
        #         self.trackers[trk_idx].detidx = det_idx
        #         # self.trackers[trk_idx].R = dets[det_idx].R
        #         dets[det_idx].track_id = self.trackers[trk_idx].id
        #         if self.trackers[trk_idx].birth_count >= 2:
        #             self.trackers[trk_idx].birth_count = 0
        #             self.trackers[trk_idx].status = TrackStatus.Confirmed

        #     for i in unmatched_b:
        #         trk_idx = unmatched_trks[i]
        #         # self.trackers[trk_idx].death_count += 1
        #         self.trackers[trk_idx].detidx = -1

        #     temp_unmatched_detidx = []
        #     for i in unmatched_a:
        #         temp_unmatched_detidx.append(unmatched_detidx[i])
        #     unmatched_detidx = temp_unmatched_detidx

        self.detidx_remain = unmatched_detidx

    def initial_tentative(self,dets,H,H_P,H_Q):
        for i in self.detidx_remain: 
            # self.trackers.append(SingerKalmanTracker(dets[i],self.wx,self.wy,self.vmax,self.dt,H,H_P,H_Q, self.t_m, self.t1, self.t2, self.window_len))
            self.trackers.append(CVHIMM(dets[i],self.wx,self.wy,self.vmax,self.dt,H,H_P,H_Q, window=self.window_len, t1=self.t1, t2=self.t2, ct1=self.ct1, ct2=self.ct2))
            # self.trackers.append(KalmanTrackerBox(dets[i],self.wx,self.wy,self.vmax,self.dt,H,H_P,H_Q, window=self.window_len, t1=self.t1, t2=self.t2))
            # self.trackers.append(OGKalmanTracker(dets[i].y[:2, :2], dets[i].R[:2, :2],self.wx,self.wy,self.vmax,dets[i].bb_width, dets[i].bb_height,self.dt))
            self.trackers[-1].status = TrackStatus.Tentative
            self.trackers[-1].detidx = i
            # self.trackers[-1].R = dets[i].R
        self.detidx_remain = []

    def delete_old_trackers(self):
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            trk.death_count += 1
            i -= 1 
            if ( trk.status == TrackStatus.Coasted and trk.death_count >= self.max_age) or ( trk.status == TrackStatus.Tentative and trk.death_count >= 2):
                  self.trackers.pop(i)

    def update_status(self,dets):
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []
        detidxs = []
        for i in range(len(self.trackers)):

            detidx = self.trackers[i].detidx
            if detidx >= 0 and detidx < len(dets):
                detidxs.append(detidx)
                self.trackers[i].h = dets[detidx].bb_height
                self.trackers[i].w = dets[detidx].bb_width

            if self.trackers[i].status == TrackStatus.Confirmed:
                self.confirmed_idx.append(i)
            elif self.trackers[i].status == TrackStatus.Coasted:
                self.coasted_idx.append(i)
            elif self.trackers[i].status == TrackStatus.Tentative:
                self.tentative_idx.append(i)

        assert len(detidxs) == len(set(detidxs))

        self.track_position_bias = np.ones((len(self.trackers,)))
