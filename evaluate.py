import os.path as osp
import numpy as np
import cv2
from viseval.eval_det import eval_det


def open_label_file(path, dtype=np.float32):
    label = np.loadtxt(path, delimiter=',', dtype=dtype,
                       ndmin=2, usecols=range(8))
    if not len(label):
        label = label.reshape(0, 8)
    return label


def main():
    dataset_dir = '/mnt/sdb/visdrone/Dataset/detection/VisDrone2019-DET-val/'
    res_dir = '/mnt/sdb/visdrone/Dataset/detection/VisDrone2019-DET-val/results_debug/'
    # dataset_dir = 'E:/datasets/visdrone/VisDrone2019-DET-test-dev/'
    # res_dir = 'E:/datasets/visdrone/VisDrone2019-DET-test-dev/results/'

    gt_dir = osp.join(dataset_dir, 'annotations')
    img_dir = osp.join(dataset_dir, 'images')
    data_list_path = osp.join(dataset_dir, 'val_list.txt')

    all_gt = []
    all_det = []
    allheight = []
    allwidth = []

    for filename in open(data_list_path).readlines():
        filename = filename.strip()
        img_path = osp.join(img_dir, filename + '.jpg')
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        allheight.append(height)
        allwidth.append(width)

        label = open_label_file(
            osp.join(gt_dir, filename + '.txt'), dtype=np.int32)
        all_gt.append(label)

        det = open_label_file(
            osp.join(res_dir, filename + '.txt'))
        all_det.append(det)

    ap_all, ap_50, ap_75, ar_1, ar_10, ar_100, ar_500, ap_classwise = eval_det(
        all_gt, all_det, allheight, allwidth, per_class=True)

    print('Average Precision  (AP) @[ IoU=0.50:0.95 | maxDets=500 ] = {}%.'.format(ap_all))
    print('Average Precision  (AP) @[ IoU=0.50      | maxDets=500 ] = {}%.'.format(ap_50))
    print('Average Precision  (AP) @[ IoU=0.75      | maxDets=500 ] = {}%.'.format(ap_75))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets=  1 ] = {}%.'.format(ar_1))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets= 10 ] = {}%.'.format(ar_10))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets=100 ] = {}%.'.format(ar_100))
    print('Average Recall     (AR) @[ IoU=0.50:0.95 | maxDets=500 ] = {}%.'.format(ar_500))

    for i, ap in enumerate(ap_classwise):
        print('Class {} AP = {}%'.format(i, ap))


if __name__ == '__main__':
    main()
