from torchvision import transforms
from yacs.config import CfgNode as CN

from mmseg.compose import Compose
from mmseg.datasets import CityscapesDataset, ADE20KDataset
from utils.coco_dataset import COCODataset


def cityscapes_datasets(FLAGS):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    crop_size = (512, 1024)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 1024),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    train_pipeline = Compose(train_pipeline)
    val_pipeline = Compose(test_pipeline)

    train_set = CityscapesDataset(data_root=FLAGS.data_root,
                                  img_dir='leftImg8bit/train',
                                  ann_dir='gtFine/train',
                                  pipeline=train_pipeline)

    val_set = CityscapesDataset(data_root=FLAGS.data_root,
                                img_dir='leftImg8bit/val',
                                ann_dir='gtFine/val',
                                pipeline=val_pipeline,
                                test_mode=True)
    return train_set, val_set, None


def ade20k_datasets(FLAGS):
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    crop_size = (512, 512)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    train_pipeline = Compose(train_pipeline)
    val_pipeline = Compose(test_pipeline)

    train_set = ADE20KDataset(data_root=FLAGS.data_root,
                              img_dir='images/training',
                              ann_dir='annotations/training',
                              pipeline=train_pipeline)

    val_set = ADE20KDataset(data_root=FLAGS.data_root,
                            img_dir='images/validation',
                            ann_dir='annotations/validation',
                            pipeline=val_pipeline,
                            test_mode=True)
    return train_set, val_set, None


def coco_datasets(FLAGS):
    _C = CN()

    _C.OUTPUT_DIR = ''
    _C.RANK = 0

    _C.TEST = CN()

    # size of images for each device
    # Test Model Epoch

    _C.TEST.USE_GT_BBOX = True

    # nms
    _C.TEST.IMAGE_THRE = 0.0
    _C.TEST.NMS_THRE = 1.0
    _C.TEST.SOFT_NMS = False
    _C.TEST.OKS_THRE = 0.9
    _C.TEST.IN_VIS_THRE = 0.2
    _C.TEST.COCO_BBOX_FILE = FLAGS.data_root + 'person_detection_results/COCO_val2017_detections_AP_H_56_person.json'

    _C.MODEL = CN()
    _C.MODEL.TARGET_TYPE = 'gaussian'
    _C.MODEL.IMAGE_SIZE = [192, 256]  # width * height, 192 * 256 | 288, 384
    _C.MODEL.HEATMAP_SIZE = [48, 64]  # width * height, 48, 64 | 72, 96
    _C.MODEL.SIGMA = 2  # 2 | 3

    _C.DATASET = CN()
    _C.DATASET.DATA_FORMAT = 'jpg'
    _C.DATASET.SELECT_DATA = False

    # training data augmentation
    _C.DATASET.FLIP = True
    _C.DATASET.SCALE_FACTOR = 0.35
    _C.DATASET.ROT_FACTOR = 45
    _C.DATASET.PROB_HALF_BODY = 0.3
    _C.DATASET.NUM_JOINTS_HALF_BODY = 8
    _C.DATASET.COLOR_RGB = True

    _C.LOSS = CN()
    _C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_set = COCODataset(_C, FLAGS.data_root, 'train2017', True,
                            transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                            ]))
    val_set = COCODataset(_C, FLAGS.data_root, 'val2017', False,
                            transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                            ]))

    return train_set, val_set, None
