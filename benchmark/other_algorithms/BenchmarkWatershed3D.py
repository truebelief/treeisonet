import warnings

warnings.filterwarnings("ignore")

import numpy as np

import argparse
import os, sys
import datetime


import numpy_indexed as npi
import glob
import laspy

import random
import numpy_groupies as npg
from scipy import stats

from skimage.segmentation import watershed, random_walker,clear_border
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

random.seed(777)
np.random.seed(777)

# pip install scikit-learn laspy[lazrs] scipy numpy_indexed numpy_groupies commentjson timm scikit-image matplotlib einops crfseg plotly

parser = argparse.ArgumentParser(description="Parsing the main command options")
parser.add_argument("--mode", default="apply", type=str, nargs="+", help="Choose your mode: apply/assess")
#parser.add_argument("--mode", default="assess", type=str, nargs="+", help="Choose your mode: apply/assess")
args = parser.parse_args()


def calc_seg_iou(prediction, reference):
    # x, y, z of prediction and reference should be same
    u, v_group = npi.group_by(prediction, np.arange(len(prediction)))
    u0, v0_group = npi.group_by(reference, np.arange(len(reference)))

    # from reference
    n0 = len(u0)
    ious0 = np.zeros(n0, dtype=np.float32)
    for i in range(n0):
        pred_label = stats.mode(prediction[v0_group[i]]).mode
        pred_group_idx = np.where(u == pred_label)[0]
        pred_group = np.array([v_group[k] for k in pred_group_idx][0])
        ious0[i] = len(np.intersect1d(v0_group[i], pred_group, assume_unique=False, return_indices=False)) / len(
            np.union1d(v0_group[i], pred_group))
    # from prediction
    n = len(u)
    ious = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ref_label = stats.mode(reference[v_group[i]]).mode
        ref_group_idx = np.where(u0 == ref_label)[0]
        ref_group = np.array([v0_group[k] for k in ref_group_idx][0])
        ious[i] = len(np.intersect1d(v_group[i], ref_group, assume_unique=False, return_indices=False)) / len(
            np.union1d(v_group[i], ref_group))

    m_iou = (np.mean(ious0) + np.mean(ious)) / 2
    precision = np.sum(ious > 0.5) / n * 100.0
    recall = np.sum(ious0 > 0.5) / n0 * 100.0

    f1_score = 2 * precision * recall / (precision + recall)
    m_iou_detected = np.mean(ious0[ious0 > 0.5] + ious[ious > 0.5]) / 2

    return m_iou, f1_score, m_iou_detected


def watershedseg(points, min_res=(0.3, 0.3, 0.3)):
    xyz_min = np.min(points[:, :3], axis=0)
    xyz_max = np.max(points[:, :3], axis=0)

    block_shape = np.floor((xyz_max[:3] - xyz_min[:3]) / min_res).astype(np.int32) + 1
    block_shape = block_shape[[1, 0, 2]]
    block_x = xyz_max[1] - points[:, 1]
    block_y = points[:, 0] - xyz_min[0]
    block_z = points[:, 2] - xyz_min[2]
    block_ijk = np.floor(np.concatenate([block_x[:, np.newaxis], block_y[:, np.newaxis], block_z[:, np.newaxis]],axis=1) / min_res).astype(np.int32)
    block_idx = np.ravel_multi_index((np.transpose(block_ijk)).astype(np.int32), block_shape)
    blk = np.zeros(block_shape[0] * block_shape[1] * block_shape[2], dtype=np.float32)
    markers = np.zeros(block_shape[0] * block_shape[1] * block_shape[2], dtype=np.int32)

    block_idx_u, block_u_idx, block_inverse_idx = np.unique(block_idx, return_index=True, return_inverse=True)
    blk[block_idx] = npg.aggregate(block_inverse_idx,points[:, -2], func=np.mean)[block_inverse_idx]
    markers[block_idx] = npg.aggregate(block_inverse_idx,points[:, -1], func=np.max)[block_inverse_idx]
    # blk[block_idx] = points[:,-2]
    # markers[block_idx] = points[:,-1]
    blk = np.reshape(blk, block_shape)
    markers = np.reshape(markers, block_shape)

    if np.sum(points[:,-1])>0:
        labels = watershed(blk, markers,mask=blk>=0)
        pt_labels=np.reshape(labels, -1, order='C')[block_idx]
        pt_ds=np.reshape(blk, -1, order='C')[block_idx]
        return pt_labels,pt_ds
    else:
        return np.zeros(len(points)),np.zeros(len(points))

def apply():
    data_dir = r"..\data"
    out_dir = r"output"
    # nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    # min_res = np.array(configs["model"]["voxel_resolution_in_meter"])
    min_res = np.array([0.3,0.3,0.3])
    eps=0.001
    app_list = r'..\server_prev_step_for_refined_training_after_ref_train\config\app_list_stem.txt'
    with open(app_list, 'r') as prep_file:
        app_pcd_fnames = prep_file.read().split('\n')
    app_pcd_fnames = [pcd_fname for pcd_fname in app_pcd_fnames if pcd_fname]
    for app_pcd_fname in app_pcd_fnames:
        fname = glob.glob(os.path.join(data_dir, app_pcd_fname+'.la*'))[0]
        las = laspy.open(fname).read()
        columns = np.transpose(np.array([las.x, las.y, las.z, las.itc_edit, las.classification]))  # las.point_format.dimension_names
        stem_ind=columns[:,-1]==4
        stem_ids=np.zeros(len(columns))
        stem_ids[stem_ind]=las.itc_edit[stem_ind]
        tree = KDTree(columns[stem_ind, :3])
        d, idx = tree.query(columns[~stem_ind, :3])
        columns_dist=np.zeros(len(columns))
        columns_dist[~stem_ind] = d+eps
        columns_pred =watershedseg(np.concatenate([columns[:,:3], columns_dist[:, np.newaxis],stem_ids[:, np.newaxis]],axis=-1), min_res=min_res[:3] * 0.5)

        # np.savetxt('tmp.txt', np.concatenate([columns[:, :3], np.transpose(columns_pred)], axis=-1))

        # fname = glob.glob(os.path.join(data_dir, app_pcd_fname + '.la*'))[0]
        # las = laspy.open(fname).read()

        las.add_extra_dim(laspy.ExtraBytesParams(name="watershed_seg", type="int32", description="watershed_seg"))
        las.watershed_seg = columns_pred[0]
        las.write(os.path.join(out_dir, "{}_watershed3d_segs.laz".format(app_pcd_fname)))
        print(f"Finished processing {app_pcd_fname}")

        ii=1


def assess():
    out_dir = r"output"
    app_list = r'..\server_prev_step_for_refined_training_after_ref_train\config\valid_list_stem.txt'
    with open(app_list, 'r') as prep_file:
        app_pcd_fnames = prep_file.read().split('\n')
    app_pcd_fnames = [pcd_fname for pcd_fname in app_pcd_fnames if pcd_fname]
    for app_pcd_fname in app_pcd_fnames:
        fname = glob.glob(os.path.join(out_dir, app_pcd_fname + '_watershed3d_segs.la*'))[0]
        las = laspy.open(fname).read()
        columns = np.transpose(np.array([las.x, las.y, las.z, las.itc_edit, las.watershed_seg]))
        miou, f1_score, m_iou_detected = calc_seg_iou(columns[:,-1], columns[:,-2])
        print("%s ---> Finish assessing %s (mIoU, F1-score, mIoU(Detected)):\t%10.5f\t%10.5f\t%10.5f" % (datetime.datetime.now(), fname,miou, f1_score, m_iou_detected))


if __name__ == '__main__':
    if args.mode == "apply":
        apply()

    if args.mode == "assess":
        assess()
