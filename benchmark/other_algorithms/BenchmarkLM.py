import warnings

warnings.filterwarnings("ignore")

import numpy as np

import argparse
import os, sys
import datetime

import numpy_indexed as npi
import numpy_groupies as npg
import glob
import laspy
import pandas as pd

import random
from scipy import stats

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx
from skimage.feature import peak_local_max

random.seed(777)
np.random.seed(777)

# pip install scikit-learn laspy[lazrs] scipy numpy_indexed numpy_groupies commentjson timm scikit-image matplotlib einops crfseg plotly
parser = argparse.ArgumentParser(description="Parsing the main command options")
# parser.add_argument("--mode", default="apply", type=str, nargs="+", help="Choose your mode: apply/assess")
parser.add_argument("--mode", default="assess", type=str, nargs="+", help="Choose your mode: apply/assess")
args = parser.parse_args()

def calc_accuracy_by_overlap(prediction_base, reference_base, reference_pts):#reference base must be one 3D point from the point clouds, no interpolated coordinates
    n_matched=0
    itc_unq, itc_idx_groups = npi.group_by(reference_pts[:, -2], np.arange(len(reference_pts[:, -2])))
    for itc_idx in itc_idx_groups:
        itc_pts = reference_pts[itc_idx, :]
        tree = KDTree(itc_pts[:, :3])
        if len(prediction_base)>1:
            matched_count=np.sum(tree.query(prediction_base[:, :3], )[0]<0.01)
            if matched_count==1:
                n_matched=n_matched+matched_count
    # matched=np.array(matched)

    # n_matched = len(matched)
    precision = n_matched / len(prediction_base) * 100.0

    recall = n_matched / len(reference_base) * 100.0

    if (precision + recall)>0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    detection_rate=len(prediction_base)/len(reference_base)*100
    return detection_rate,precision, recall, f1_score



def assess():
    data_dir = r"..\data"
    out_dir = r"output"
    app_list = r'..\server_prev_step_for_refined_training_after_ref_train\config\valid_list_stem.txt'

    base_buffer=2.0


    with open(app_list, 'r') as prep_file:
        app_pcd_fnames = prep_file.read().split('\n')
    app_pcd_fnames = [pcd_fname for pcd_fname in app_pcd_fnames if pcd_fname]
    for app_pcd_fname in app_pcd_fnames:
        fname = glob.glob(os.path.join(data_dir, app_pcd_fname + '.la*'))[0]
        las = laspy.open(fname).read()
        pred_treebase_fname = glob.glob(os.path.join(out_dir, app_pcd_fname + '_treetop_lm.csv'))[0]
        pred_treebase = pd.read_csv(pred_treebase_fname, delimiter=' ', header=None).values

        columns = np.transpose(np.array([las.x, las.y, las.z, las.itc_edit, las.classification]))  # las.point_format.dimension_names
        itc_unq, itc_idx_groups = npi.group_by(columns[:, -2], np.arange(len(columns[:, -2])))

        stem_idx = []
        seg_peak_poses = []

        for itc_idx in itc_idx_groups:
            segs_pos = columns[itc_idx, :]
            stem_ind = segs_pos[:, -1] == 4
            stem_pos = segs_pos[stem_ind]
            if len(stem_pos) == 0:
                seg_peak_pos = segs_pos[np.argmax(segs_pos[:, 2]), :3]
                stem_idx.append(itc_idx)
            else:
                stem_idx.append(itc_idx[stem_ind])
                seg_peak_pos = np.mean(stem_pos[stem_pos[:, 2] - np.min(stem_pos[:, 2]) < base_buffer, :3], axis=0)
            seg_peak_poses.append(seg_peak_pos)
        seg_peak_poses=np.array(seg_peak_poses)
        detection_rate, precision, recall, f1_score = calc_accuracy_by_overlap(pred_treebase, seg_peak_poses,columns)
        print("%s ---> Finish assessing %s (F-score, Precision, Recall, DetectionRate):\t%10.5f\t%10.5f\t%10.5f\t%10.5f" % (datetime.datetime.now(), fname, f1_score, precision, recall, detection_rate))




def create_chm(columns,res):#no ground

    xyz_mins = np.min(columns[:, :3], axis=0)
    xyz_maxs = np.max(columns[:, :3], axis=0)
    pcd_dtm_shape=np.floor((xyz_maxs-xyz_mins)/res)+1
    pcd_dtm_shape=pcd_dtm_shape[: 2].astype(np.int32)
    columns_i = xyz_maxs[1] - columns[:, 1]
    columns_j = columns[:, 0] - xyz_mins[0]

    columns_ij = np.floor(np.concatenate([columns_i[:, np.newaxis], columns_j[:, np.newaxis]], axis=1) / res[[1, 0]]).astype(np.int32)
    column_within_ind = np.all(np.logical_and(columns_ij >= np.array([0, 0]), columns_ij < np.array(pcd_dtm_shape)),axis=1)
    # column_abg_ind=columns[:,-1]>1

    columns_dsm_ij = columns_ij[column_within_ind]
    columns_dsm_idx = np.ravel_multi_index((np.transpose(columns_dsm_ij)).astype(np.int32), pcd_dtm_shape)
    column_dsm_idx_u, column_dsm_idx_uidx, column_dsm_inverse_idx = np.unique(columns_dsm_idx, return_index=True,return_inverse=True)

    columns_dtm_ij = columns_ij[column_within_ind]
    columns_dtm_idx = np.ravel_multi_index((np.transpose(columns_dtm_ij)).astype(np.int32), pcd_dtm_shape)
    column_dtm_idx_u, column_dtm_idx_uidx, column_dtm_inverse_idx = np.unique(columns_dtm_idx, return_index=True,return_inverse=True)


    columns_z_min = npg.aggregate(column_dtm_inverse_idx, columns[column_within_ind, 2], func='min', fill_value=np.nan)
    columns_z_max = npg.aggregate(column_dsm_inverse_idx, columns[column_within_ind, 2], func='max', fill_value=np.nan)

    # pcd_abg_layer = np.full((*raster_tile_shape, 1), np.inf)
    pcd_abg_dtm = np.full(pcd_dtm_shape[0]*pcd_dtm_shape[1], np.inf)
    pcd_abg_dtm[column_dtm_idx_u] = columns_z_min
    pcd_abg_dsm = np.full(pcd_dtm_shape[0]*pcd_dtm_shape[1], np.inf)
    pcd_abg_dsm[column_dsm_idx_u] = columns_z_max

    pcd_abg_dtm=pcd_abg_dtm.reshape(pcd_dtm_shape)
    pcd_abg_dsm=pcd_abg_dsm.reshape(pcd_dtm_shape)
    dtm_mask = np.isinf(pcd_abg_dtm)
    dtm_coord = np.argwhere(~dtm_mask)
    values = pcd_abg_dtm[~dtm_mask]

    grid_indices = np.argwhere(dtm_mask)
    dtm_filled = griddata(dtm_coord, values, grid_indices, method='cubic')
    pcd_abg_dtm[grid_indices[:, 0], grid_indices[:, 1]] = dtm_filled

    # plt.imshow(pcd_abg_dtm)
    # plt.show()

    # dsm_mask = np.isinf(pcd_abg_dsm[:, :, 0])
    # dsm_coord = np.argwhere(~dsm_mask)
    # values = pcd_abg_dsm[~dsm_mask]
    # grid_indices = np.argwhere(dsm_mask)
    # dsm_filled = griddata(dsm_coord, values, grid_indices, method='cubic')
    # pcd_abg_dsm[grid_indices[:, 0], grid_indices[:, 1]] = dsm_filled

    chm = pcd_abg_dsm - pcd_abg_dtm
    chm[chm < 0] = 0
    chm[np.isnan(chm)] = 0
    chm[np.isinf(chm)] = 0
    # write_tif(os.path.join(out_dir, out_name + '_chm.tif'), chm, crs, transform_tile)
    return chm

def apply():
    data_dir = r"..\data"
    out_dir = r"output"
    min_res = np.array([0.3, 0.3, 0.3])
    search_radius=0.3

    app_list = r'..\server_prev_step_for_refined_training_after_ref_train\config\app_list_stem.txt'
    with open(app_list, 'r') as prep_file:
        app_pcd_fnames = prep_file.read().split('\n')
    app_pcd_fnames = [pcd_fname for pcd_fname in app_pcd_fnames if pcd_fname]
    for app_pcd_fname in app_pcd_fnames:
        fname = glob.glob(os.path.join(data_dir, app_pcd_fname + '.la*'))[0]
        las = laspy.open(fname).read()
        columns = np.transpose(np.array([las.x, las.y, las.z, las.itc_edit, las.classification]))  # las.point_format.dimension_names

        chm=create_chm(columns,min_res)
        chm_lm=peak_local_max(chm, min_distance=5,exclude_border=False)
        # chm_lm_pts=chm_lm[:,[1,0]]*min_res[:2]+np.min(columns[:,:2],0)
        chm_lm_pts=np.transpose([chm_lm[:,1]*min_res[0]+np.min(columns[:,0]),np.max(columns[:,1])-chm_lm[:,0]*min_res[1]])
        # chm_lm = peak_local_max(chm, min_distance=5, exclude_border=False)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(chm)
        # ax[0].scatter(chm_lm[:, 1], chm_lm[:, 0], c="orange", s=20)
        # plt.show()
        tree = KDTree(columns[:, :2])
        pred_idxs = tree.query_ball_point(chm_lm_pts[:, :2], search_radius)
        pred_pts = np.array([columns[pred_idx[np.argmax(columns[pred_idx, 2])], :3] for pred_idx in pred_idxs if pred_idx])

        outfname=os.path.join(out_dir, "{}_treetop_lm.csv".format(app_pcd_fname))
        np.savetxt(outfname,pred_pts)

        print(f"Finished processing {app_pcd_fname}")

if __name__ == '__main__':
    if args.mode == "apply":
        apply()

    if args.mode == "assess":
        assess()



