import argparse
import os
import shutil

import h5py
import logging
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.BLGnet import BLGNet

# from networks.efficientunet import UNet
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Prostate', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Prostate', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='BLGnet', help='model_name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=16,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _ = net(input)
            else:
                out_main, _ = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    # second_metric = calculate_metric_percase(prediction == 2, label == 2)
    # third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}_{}_labeled/BLGNet".format(
        FLAGS.exp, FLAGS.model, FLAGS.labeled_num)
    test_save_path = "../model/{}_{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.model, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = BLGNet(in_chns=1, class_num=2).cuda()
    save_mode_path = os.path.join(
        snapshot_path, 'BLGNet_best_model.pth')
    checkpoint = torch.load(save_mode_path)
    net.load_state_dict(checkpoint)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    metrics_file = open(test_save_path + 'metrics.txt', 'w')
    for case in tqdm(image_list):
        first_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_metric = np.asarray(first_metric)

        single_avg_metric = first_metric
        metrics_file.write(f"Case: {case}\n")
        metrics_file.write(
            f"Average Metric: dice={single_avg_metric[0]}, jc={single_avg_metric[1]}, hd95={single_avg_metric[2]}, asd={single_avg_metric[3]}\n")
        first_total += np.asarray(first_metric)
    avg_metric = first_total / len(image_list)
    avg_dice = avg_metric[0]
    avg_jc = avg_metric[1]
    avg_hd95 = avg_metric[2]
    avg_asd = avg_metric[3]
    metrics_file.write(f"Overall Metrics:\n")
    metrics_file.write(f"Average Overall Metrics: dice={avg_dice}, jc={avg_jc}, hd95={avg_hd95}, asd={avg_asd}\n")

    metrics_file.close()
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
