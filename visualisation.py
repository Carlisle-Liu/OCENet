import cv2
import numpy as np
import torchvision.transforms as transforms


def visualise_prediction_init(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = 'temp/'
        name = '{:02d}_init.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualise_prediction_ref(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = 'temp/'
        name = '{:02d}_ref.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualise_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = 'temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualise_pred_dis(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = 'temp/'
        name = '{:02d}_pred_dis.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualise_gt_dis(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = 'temp/'
        name = '{:02d}_gt_dis.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualise_original_img(rec_img):
    img_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.4850 / .229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk, :, :, :]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = 'temp/'
        name = '{:02d}_img.png'.format(kk)
        current_img = current_img.transpose((1, 2, 0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path + name, new_img)


def visualise_boundary_label(bds_gts):
    for kk in range(bds_gts.shape[0]):
        bds_gts_kk = bds_gts[kk, :, :, :]
        bds_gts_kk = bds_gts_kk.detach().cpu().numpy().squeeze()
        bds_gts_kk *= 255.0
        bds_gts_kk = bds_gts_kk.astype(np.uint8)
        save_path = 'temp/'
        name = '{:02d}_gt_bds.png'.format(kk)
        cv2.imwrite(save_path + name, bds_gts_kk)

def visualise_boundary_pred(bds_pred):
    for kk in range(bds_pred.shape[0]):
        bds_pred_kk = bds_pred[kk, :, :, :]
        bds_pred_kk = bds_pred_kk.detach().cpu().numpy().squeeze()
        bds_pred_kk *= 255.0
        bds_pred_kk = bds_pred_kk.astype(np.uint8)
        save_path = 'temp/'
        name = '{:02d}_sal_grad.png'.format(kk)
        cv2.imwrite(save_path + name, bds_pred_kk)
