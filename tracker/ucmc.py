
from __future__ import print_function

import numpy as np
from lap import lapjv


from .kalman import KalmanTracker,TrackStatus


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
    def __init__(self,a1,a2,wx, wy,vmax, max_age, fps, dataset, high_score, use_cmc,detector = None):
        self.wx = wx
        self.wy = wy
        self.vmax = vmax
        self.dataset = dataset
        self.high_score = high_score
        self.max_age = max_age
        self.a1 = a1
        self.a2 = a2
        self.dt = 1.0/fps

        self.use_cmc = use_cmc

        self.trackers = []
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []

        self.detector = detector


    def update(self, dets,frame_id,frame_affine): 
        for track in self.trackers:
            track.predict(frame_affine)
                   
        self.data_association(dets,frame_id,frame_affine)
        
        self.associate_tentative(dets)

        if frame_id > 1:
            self.detector.mapper.predict(frame_affine)
        updated_tracks = [track for track in self.trackers if track.detidx > -1 and track.status == TrackStatus.Confirmed]
        self.detector.mapper.update(updated_tracks, dets)
        for track in self.trackers:
            track.update_homography(self.detector.mapper.A.T.flatten()[:-1], self.detector.mapper.covariance)

        self.initial_tentative(dets, self.detector.mapper.A.T.flatten()[:-1], self.detector.mapper.covariance,
                               self.detector.mapper.process_covariance)
        
        self.delete_old_trackers()
        
        self.update_status(dets)
    
    def data_association(self, dets,frame_id,frame_affine):
        # Separate detections into high score and low score
        detidx_high = []
        detidx_low = []
        for i in range(len(dets)):
            if dets[i].conf >= self.high_score:
                detidx_high.append(i)
            else:
                detidx_low.append(i)

        # Predcit new locations of tracks
        # for track in self.trackers:
        #     track.predict()
            # if self.use_cmc:
            #     x,y = self.detector.cmc(track.kf.x[0,0],track.kf.x[2,0],track.w,track.h,frame_id)
            #     track.kf.x[0,0] = x
            #     track.kf.x[2,0] = y
        
        trackidx_remain = []
        self.detidx_remain = []

        # Associate high score detections with tracks
        trackidx = self.confirmed_idx + self.coasted_idx
        num_det = len(detidx_high)
        num_trk = len(trackidx)

        for trk in self.trackers:
            trk.detidx = -1

        if num_det*num_trk > 0:
            cost_matrix = np.zeros((num_det, num_trk))
            det_ys = [dets[det_idx].y for det_idx in detidx_high]
            det_covs = [dets[det_idx].R for det_idx in detidx_high]
            for j in range(num_trk):
                trk_idx = trackidx[j]
                # cost_matrix[i,j] = self.trackers[trk_idx].distance(*self.detector.mapper.get_UV_and_error(dets[det_idx].get_box()))
                cost_matrix[:,j] = self.trackers[trk_idx].distance(det_ys, det_covs)
                
            matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix, self.a1)
            
            for i in unmatched_a:
                self.detidx_remain.append(detidx_high[i])
            for i in unmatched_b:
                trackidx_remain.append(trackidx[i])
            
            for i,j in matched_indices:
                det_idx = detidx_high[i]
                trk_idx = trackidx[j]
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R)
                # self.detector.mapper.update(
                #     *self.detector.mapper.get_UV_and_error([dets[det_idx].bb_left,dets[det_idx].bb_top,dets[det_idx].bb_width,dets[det_idx].bb_height]),
                #                              self.trackers[trk_idx].kf.x, self.trackers[trk_idx].kf.P)
                self.trackers[trk_idx].death_count = 0
                self.trackers[trk_idx].detidx = det_idx
                self.trackers[trk_idx].R = dets[det_idx].R
                self.trackers[trk_idx].status = TrackStatus.Confirmed
                dets[det_idx].track_id = self.trackers[trk_idx].id

        else:
            self.detidx_remain = detidx_high
            trackidx_remain = trackidx

        
        # Associate low score detections with remain tracks
        num_det = len(detidx_low)
        num_trk = len(trackidx_remain)
        if num_det*num_trk > 0:
            cost_matrix = np.zeros((num_det, num_trk))
            det_ys = [dets[det_idx].y for det_idx in detidx_low]
            det_covs = [dets[det_idx].R for det_idx in detidx_low]
            for j in range(num_trk):
                trk_idx = trackidx_remain[j]
                # cost_matrix[i,j] = self.trackers[trk_idx].distance(*self.detector.mapper.get_UV_and_error(dets[det_idx].get_box()))
                cost_matrix[:,j] = self.trackers[trk_idx].distance(det_ys, det_covs)
            matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix,self.a2)
            
            for i in unmatched_b:
                trk_idx = trackidx_remain[i]
                self.trackers[trk_idx].status = TrackStatus.Coasted
                # self.trackers[trk_idx].death_count += 1
                self.trackers[trk_idx].detidx = -1

            for i,j in matched_indices:
                det_idx = detidx_low[i]
                trk_idx = trackidx_remain[j]
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R)                
                # self.detector.mapper.update(
                #     *self.detector.mapper.get_UV_and_error([dets[det_idx].bb_left,dets[det_idx].bb_top,dets[det_idx].bb_width,dets[det_idx].bb_height]),
                #                              self.trackers[trk_idx].kf.x, self.trackers[trk_idx].kf.P)
                self.trackers[trk_idx].death_count = 0
                self.trackers[trk_idx].detidx = det_idx
                self.trackers[trk_idx].R = dets[det_idx].R
                self.trackers[trk_idx].status = TrackStatus.Confirmed
                dets[det_idx].track_id = self.trackers[trk_idx].id


    def associate_tentative(self, dets):
        num_det = len(self.detidx_remain)
        num_trk = len(self.tentative_idx)

        cost_matrix = np.zeros((num_det, num_trk))
        det_ys = [dets[det_idx].y for det_idx in self.detidx_remain]
        det_covs = [dets[det_idx].R for det_idx in self.detidx_remain]
        if len(det_ys):
            for j in range(num_trk):
                trk_idx = self.tentative_idx[j]
                # cost_matrix[i,j] = self.trackers[trk_idx].distance(*self.detector.mapper.get_UV_and_error(dets[det_idx].get_box()))
                cost_matrix[:,j] = self.trackers[trk_idx].distance(det_ys, det_covs)
            
        matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix,self.a1)

        for i,j in matched_indices:
            det_idx = self.detidx_remain[i]
            trk_idx = self.tentative_idx[j]
            self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R)
            # self.detector.mapper.update(
            #         *self.detector.mapper.get_UV_and_error([dets[det_idx].bb_left,dets[det_idx].bb_top,dets[det_idx].bb_width,dets[det_idx].bb_height]),
            #                                  self.trackers[trk_idx].kf.x, self.trackers[trk_idx].kf.P)
            self.trackers[trk_idx].death_count = 0
            self.trackers[trk_idx].birth_count += 1
            self.trackers[trk_idx].detidx = det_idx
            self.trackers[trk_idx].R = dets[det_idx].R
            dets[det_idx].track_id = self.trackers[trk_idx].id
            if self.trackers[trk_idx].birth_count >= 2:
                self.trackers[trk_idx].birth_count = 0
                self.trackers[trk_idx].status = TrackStatus.Confirmed

        for i in unmatched_b:
            trk_idx = self.tentative_idx[i]
            # self.trackers[trk_idx].death_count += 1
            self.trackers[trk_idx].detidx = -1

    
        unmatched_detidx = []
        for i in unmatched_a:
            unmatched_detidx.append(self.detidx_remain[i])
        self.detidx_remain = unmatched_detidx

            
    
    def initial_tentative(self,dets,H,H_P,H_Q):
        for i in self.detidx_remain: 
            dets[i].y,dets[i].R = self.detector.mapper.uv2xy(dets[i].y,dets[i].R)
            self.trackers.append(KalmanTracker(dets[i].y,dets[i].R,self.wx,self.wy,self.vmax, dets[i].bb_width,dets[i].bb_height,self.dt,H,H_P,H_Q,self.detector.mapper.process_alpha))
            self.trackers[-1].status = TrackStatus.Tentative
            self.trackers[-1].detidx = i
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
        for i in range(len(self.trackers)):

            detidx = self.trackers[i].detidx
            if detidx >= 0 and detidx < len(dets):
                self.trackers[i].h = dets[detidx].bb_height
                self.trackers[i].w = dets[detidx].bb_width

            if self.trackers[i].status == TrackStatus.Confirmed:
                self.confirmed_idx.append(i)
            elif self.trackers[i].status == TrackStatus.Coasted:
                self.coasted_idx.append(i)
            elif self.trackers[i].status == TrackStatus.Tentative:
                self.tentative_idx.append(i)

        
