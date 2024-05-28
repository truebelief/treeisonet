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
from scipy import stats

import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx

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
    precision = np.sum(ious0 > 0.5) / n0 * 100.0
    recall = np.sum(ious > 0.5) / n * 100.0

    f1_score = 2 * precision * recall / (precision + recall)
    m_iou_detected = np.mean(ious0[ious0 > 0.5] + ious[ious > 0.5]) / 2

    return m_iou, f1_score, m_iou_detected

def shortestpath(columns,min_res):
    xyz_min = np.min(columns[:, :3], axis=0)
    xyz_max = np.max(columns[:, :3], axis=0)
    block_shape = np.floor((xyz_max[:3] - xyz_min[:3]) / min_res).astype(np.int32) + 1
    block_shape = block_shape[[1, 0, 2]]
    block_x = xyz_max[1] - columns[:, 1]
    block_y = columns[:, 0] - xyz_min[0]
    block_z = columns[:, 2] - xyz_min[2]
    block_ijk = np.floor(np.concatenate([block_x[:, np.newaxis], block_y[:, np.newaxis], block_z[:, np.newaxis]],axis=1) / min_res).astype(np.int32)
    block_idx = np.ravel_multi_index((np.transpose(block_ijk)).astype(np.int32), block_shape)
    block_idx_u, block_idx_uidx, block_inverse_idx = np.unique(block_idx, return_index=True, return_inverse=True)
    columns_dec = columns[block_idx_uidx]
    stems_ind = columns_dec[:, -1] == 4
    stems_idx = np.where(stems_ind)[0]
    stems_pos = columns_dec[stems_idx]
    tree = KDTree(columns_dec[:, :3])
    # _, target_idx = tree.query(seg_peak_pos[:, :3], k=1)
    # stem_idx_u, stem_idx_uidx = np.unique(np.floor((columns_dec[stems_idx,:3]-xyz_min[:3])/self.min_res/4), return_index=True)

    # target_idx=stems_idx[stem_idx_uidx]
    target_idx = []
    itc_unq, itc_idx_groups = npi.group_by(stems_pos[:, -2], np.arange(len(stems_pos[:, -2])))
    for i, itc_idx in enumerate(itc_idx_groups):
        stems_pos_itc = stems_pos[itc_idx, :]
        stem_idx_u, stem_idx_uidx = np.unique(np.floor((stems_pos_itc[:, :3] - xyz_min[:3]) / min_res / 4), axis=0,return_index=True)
        target_idx.append(stems_idx[itc_idx[stem_idx_uidx]])

        # seg_peak_pos[i,:] = segs_pos[np.argmin(segs_pos[:, 2]),:3]
    target_idx = np.concatenate(target_idx)
    G = nx.Graph()
    K = 6
    max_d = 1.0
    ds, idxs = tree.query(columns_dec[:, :3], k=K)
    # idxs=idxs[:,1:]
    # ds=ds[:,1:]
    keep_ind = ds[:, 1] <= max_d
    idxs = idxs[keep_ind]
    ds = ds[keep_ind]
    for j in range(len(idxs)):
        node1 = idxs[j, 0]
        for k in range(1, K):
            node2 = idxs[j, k]
            weight = ds[j, k]
            if weight < max_d:
                G.add_edge(node1, node2, weight=weight)
    # target=np.where(stems_ind)[0]
    source = np.where(~stems_ind)[0]
    # for i in range(len(source)):
    #     for j in range(len(target_idx)):
    #         paths=nx.shortest_path(G, source=source[i], target=target_idx[j])
    cutoff = 100.0
    # target_idx[0] in G
    distances = {target: dict(nx.single_target_shortest_path_length(G, target, cutoff=cutoff)) for target in target_idx if target in G}
    node_assignments = {}

    for k in range(len(source)):
        node = source[k]
        # Find the target with the minimum distance to this node
        nearest_target = min(distances, key=lambda x: distances[x].get(node, float('inf')))
        # nearest_target = min(distances, key=lambda x: distances[x][node])
        node_assignments[node] = nearest_target
    # nonstem_pred=np.zeros(len(source))
    dec_pred = np.zeros(len(columns_dec))
    # nonstem_pred[np.array(list(node_assignments.keys()))]=np.array(list(node_assignments.values()))
    # dec_pred[~stems_ind]=nonstem_pred
    # all_pred=dec_pred[block_inverse_idx]
    # np.savetxt('tmp',np.concatenate([columns,all_pred[:,np.newaxis]],2))

    dec_pred[np.array(list(node_assignments.keys()))] = columns_dec[np.array(list(node_assignments.values())), -2]
    all_pred = dec_pred[block_inverse_idx]
    all_pred[columns[:, -1] == 4] = columns[columns[:, -1] == 4, 3]
    remain_ind = all_pred == 0
    # remain_pred=all_pred[remain_ind]
    done_pred = all_pred[~remain_ind]

    tree = KDTree(columns[~remain_ind, :3])

    ds, idxs = tree.query(columns[remain_ind, :3], k=1)
    all_pred[remain_ind] = done_pred[idxs]
    return all_pred


def assess():
    out_dir = r"output"
    app_list = r'..\server_prev_step_for_refined_training_after_ref_train\config\valid_list_stem.txt'
    with open(app_list, 'r') as prep_file:
        app_pcd_fnames = prep_file.read().split('\n')
    app_pcd_fnames = [pcd_fname for pcd_fname in app_pcd_fnames if pcd_fname]
    for app_pcd_fname in app_pcd_fnames:
        fname = glob.glob(os.path.join(out_dir, app_pcd_fname + '_shortestpath_segs.la*'))[0]
        las = laspy.open(fname).read()
        columns = np.transpose(np.array([las.x, las.y, las.z, las.itc_edit, las.shortestpah_seg]))
        miou, f1_score, m_iou_detected = calc_seg_iou(columns[:,-1], columns[:,-2])
        print("%s ---> Finish assessing %s (mIoU, F1-score, mIoU(Detected)):\t%10.5f\t%10.5f\t%10.5f" % (datetime.datetime.now(), fname,miou, f1_score, m_iou_detected))


def apply():
    data_dir = r"..\data"
    out_dir = r"output"
    min_res = np.array([0.3, 0.3, 0.3])
    app_list = r'..\server_prev_step_for_refined_training_after_ref_train\config\app_list_stem.txt'
    with open(app_list, 'r') as prep_file:
        app_pcd_fnames = prep_file.read().split('\n')
    app_pcd_fnames = [pcd_fname for pcd_fname in app_pcd_fnames if pcd_fname]
    for app_pcd_fname in app_pcd_fnames:
        fname = glob.glob(os.path.join(data_dir, app_pcd_fname + '.la*'))[0]
        las = laspy.open(fname).read()
        columns = np.transpose(np.array([las.x, las.y, las.z, las.itc_edit, las.classification]))  # las.point_format.dimension_names
        all_pred=shortestpath(columns,min_res)

        las.add_extra_dim(laspy.ExtraBytesParams(name="shortestpah_seg", type="int32", description="shortestpah_seg"))
        las.shortestpah_seg = all_pred
        las.write(os.path.join(out_dir, "{}_shortestpath_segs.laz".format(app_pcd_fname)))
        print(f"Finished processing {app_pcd_fname}")

if __name__ == '__main__':
    if args.mode == "apply":
        apply()

    if args.mode == "assess":
        assess()



