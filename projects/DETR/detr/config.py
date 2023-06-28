# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_detr_config(cfg):
    # cfg is a CfgNode object. by zwz
    """
    Add config for DETR.
    """
    cfg.MODEL.DETR = CN()
    cfg.MODEL.DETR.NUM_CLASSES = 80

    # For Segmentation
    cfg.MODEL.DETR.FROZEN_WEIGHTS = ''

    # LOSS
    cfg.MODEL.DETR.GIOU_WEIGHT = 2.0
    cfg.MODEL.DETR.L1_WEIGHT = 5.0
    cfg.MODEL.DETR.DEEP_SUPERVISION = True
    cfg.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1

    # TRANSFORMER
    cfg.MODEL.DETR.NHEADS = 8
    cfg.MODEL.DETR.DROPOUT = 0.1
    cfg.MODEL.DETR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DETR.ENC_LAYERS = 6
    cfg.MODEL.DETR.DEC_LAYERS = 6
    cfg.MODEL.DETR.PRE_NORM = False

    cfg.MODEL.DETR.HIDDEN_DIM = 256
    cfg.MODEL.DETR.NUM_OBJECT_QUERIES = 100

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1


# def add_ddetrs_config(cfg):
#     """
#     Add config for DDETRS.
#     """
#     cfg.MODEL.DDETRS = CN()
#     cfg.MODEL.DDETRS.NUM_CLASSES = 80
#     cfg.MODEL.OTA = False
    
#     # DataLoader
#     cfg.INPUT.DATASET_MAPPER_NAME = "detr" # use "coco_instance_lsj" for LSJ aug
#     # LSJ aug
#     cfg.INPUT.IMAGE_SIZE = 1024
#     cfg.INPUT.MIN_SCALE = 0.1
#     cfg.INPUT.MAX_SCALE = 2.0
#     # mixup
#     cfg.INPUT.USE_MIXUP = False
#     cfg.INPUT.MIXUP_PROB = 1.0
    
#     # DataLoader
#     # cfg.INPUT.SAMPLING_FRAME_NUM = 1
#     # cfg.INPUT.SAMPLING_FRAME_RANGE = 20
#     # cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
#     # cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"
#     cfg.INPUT.TEST_H_FLIP = False  # add test time h flip

#     # LOSS
#     cfg.MODEL.DDETRS.MASK_WEIGHT = 2.0
#     cfg.MODEL.DDETRS.DICE_WEIGHT = 5.0
#     cfg.MODEL.DDETRS.GIOU_WEIGHT = 2.0
#     cfg.MODEL.DDETRS.L1_WEIGHT = 5.0
#     cfg.MODEL.DDETRS.CLASS_WEIGHT = 2.0
#     cfg.MODEL.DDETRS.DEEP_SUPERVISION = True
#     # cfg.MODEL.DDETRS.NO_OBJECT_WEIGHT = 0.1
#     cfg.MODEL.DDETRS.MASK_STRIDE = 4
#     cfg.MODEL.DDETRS.MATCH_STRIDE = 4
#     cfg.MODEL.DDETRS.FOCAL_ALPHA = 0.25

#     cfg.MODEL.DDETRS.SET_COST_CLASS = 2
#     cfg.MODEL.DDETRS.SET_COST_BOX = 5
#     cfg.MODEL.DDETRS.SET_COST_GIOU = 2

#     # TRANSFORMER
#     cfg.MODEL.DDETRS.NHEADS = 8
#     cfg.MODEL.DDETRS.DROPOUT = 0.1
#     cfg.MODEL.DDETRS.DIM_FEEDFORWARD = 1024
#     cfg.MODEL.DDETRS.ENC_LAYERS = 6
#     cfg.MODEL.DDETRS.DEC_LAYERS = 6

#     cfg.MODEL.DDETRS.HIDDEN_DIM = 256
#     cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES = 300
#     cfg.MODEL.DDETRS.DEC_N_POINTS = 4
#     cfg.MODEL.DDETRS.ENC_N_POINTS = 4
#     cfg.MODEL.DDETRS.NUM_FEATURE_LEVELS = 4

#     cfg.SOLVER.OPTIMIZER = "ADAMW"
#     cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1


#     ## support Swin backbone
#     cfg.MODEL.SWIN = CN()
#     cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
#     cfg.MODEL.SWIN.PATCH_SIZE = 4
#     cfg.MODEL.SWIN.EMBED_DIM = 96
#     cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
#     cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
#     cfg.MODEL.SWIN.WINDOW_SIZE = 7
#     cfg.MODEL.SWIN.MLP_RATIO = 4.0
#     cfg.MODEL.SWIN.QKV_BIAS = True
#     cfg.MODEL.SWIN.QK_SCALE = None
#     cfg.MODEL.SWIN.DROP_RATE = 0.0
#     cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
#     cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
#     cfg.MODEL.SWIN.APE = False
#     cfg.MODEL.SWIN.PATCH_NORM = True
#     cfg.MODEL.SWIN.OUT_IINDICES = (0,1,2,3)
#     cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
#     cfg.MODEL.SWIN.USE_CHECKPOINT = False

#     # supprt ConvNeXt backbone
#     cfg.MODEL.CONVNEXT = CN()
#     cfg.MODEL.CONVNEXT.NAME = "tiny"
#     cfg.MODEL.CONVNEXT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
#     cfg.MODEL.CONVNEXT.USE_CHECKPOINT = False

#     # find_unused_parameters
#     cfg.FIND_UNUSED_PARAMETERS = True