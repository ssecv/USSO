# Copyright (c) Facebook, Inc. and its affiliates.
import os
from typing import Tuple
from dct_encoding.mask_encoding import DctMaskEncoding
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, normalize_boxes

import numpy as np
import matplotlib.pyplot as plt
import cv2

@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        association_weight = cfg.MODEL.MASK_FORMER.ASSOCIATION_WEIGHT


        contrastive_weight = cfg.MODEL.MASK_FORMER.CONTRASTIVE_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_association=association_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight, 
            "loss_association_ce": association_weight,
            
            "loss_mask_object_pred_masks": mask_weight,
            "loss_dice_object_pred_masks": dice_weight,
            "loss_mask_shadow_pred_masks": mask_weight,
            "loss_dice_shadow_pred_masks": dice_weight,

            "loss_mask_object_structure_mask": mask_weight,
            "loss_dice_object_structure_mask": dice_weight,
            "loss_mask_shadow_structure_mask": mask_weight,
            "loss_dice_shadow_structure_mask": dice_weight,
            "loss_contrastive": contrastive_weight,
            }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "association", "contrastive"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [
                    {"instances": x["instances"].to(self.device), "association_anno": x["association_anno"].to(self.device)}
                    for x in batched_inputs
                ]
                targets = self.prepare_sobav2_targets(gt_instances, images)
            else:
                targets = None


            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            object_mask_pred_results = outputs["object_pred_masks"]
            shadow_mask_pred_results = outputs["shadow_pred_masks"]
            association_logits = outputs["association_logits"]

            object_mask_pred_results = F.interpolate(
                object_mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            shadow_mask_pred_results = F.interpolate(
                shadow_mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            

            del outputs

            processed_results = []
            for mask_cls_result, object_mask_pred_result, shadow_mask_pred_result, asso_logits, input_per_image, image_size in zip(
                mask_cls_results, object_mask_pred_results, shadow_mask_pred_results, association_logits, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                if self.sem_seg_postprocess_before_inference:
                    object_mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        object_mask_pred_result, image_size, height, width
                    )
                    shadow_mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        shadow_mask_pred_result, image_size, height, width
                    )


                    mask_cls_result = mask_cls_result.to(object_mask_pred_result)

                # instance segmentation inference
                if self.instance_on:
                    results, associations = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, 
                                                                                        object_mask_pred_result, 
                                                                                        shadow_mask_pred_result, 
                                                                                        asso_logits)
                    processed_results.append(results)
                    processed_results.append(associations)

            return processed_results

                

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets
    
    def extract_features(self, masks):
        """
        Extract contours and Canny edges from given tensor masks.
        """
        num_masks = masks.shape[0]
        
        contours_list = torch.zeros_like(masks, dtype=torch.float32)

        for i in range(num_masks):
            mask_np = masks[i].cpu().numpy()

            # Rescale mask values from [0, 1] to [0, 255]
            mask_np = (mask_np * 255).astype(np.uint8)

            # For contours
            body = cv2.blur(mask_np, ksize=(5, 5))
            body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
            body = body ** 0.5
            tmp = body[np.where(body > 0)]
            if len(tmp) != 0:
                body[np.where(body > 0)] = np.floor(tmp / np.max(tmp) * 255)
            structure = np.clip(mask_np - body, 0, 255)
            contours_list[i] = torch.from_numpy(structure / 255.0).to(masks.device)
            
        return contours_list#, edges_list



    def prepare_sobav2_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        # height, width = images.image_sizes[0]
        new_targets = []
        for targets_per_image in targets:
            association_bbox = targets_per_image['association_anno'].gt_boxes ## xyxy

            ## half_labels_num is half of the number of instances of GT, the first half is object, and the second half is shadow
            instance_num = len(targets_per_image['instances'].gt_classes)
            half_instance_num = instance_num // 2

            # pad gt
            gt_masks = targets_per_image['instances'].gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            
            structures = self.extract_features(padded_masks)
            object_structures = structures[:half_instance_num] 
            shadow_structures = structures[half_instance_num:] 


            object_masks = padded_masks[:half_instance_num]
            shadow_masks = padded_masks[half_instance_num:]

            new_targets.append(
                {
                    "h_pad": h_pad,
                    "w_pad": w_pad,

                    "association_score": torch.ones(half_instance_num, 1).to(association_bbox.tensor.device),

                    "labels": targets_per_image['instances'].gt_classes,
                    "object_labels": targets_per_image['instances'].gt_classes[:half_instance_num],
                    "shadow_labels": targets_per_image['instances'].gt_classes[half_instance_num:],

                    "object_masks": object_masks,
                    "shadow_masks": shadow_masks,


                    "object_structures": object_structures,
                    "shadow_structures": shadow_structures,
                }
            )
        return new_targets


    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info


    def instance_inference(self, mask_cls, object_mask_pred, shadow_mask_pred, assoc_logits):
        # Perform softmax processing on classification
        obj_prob = F.softmax(mask_cls, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        # Perform sigmoid processing on the degree of matching
        asso_scores = assoc_logits.sigmoid()

        # Find the index with a correlation greater than the threshold based on assoc_logits
        assoc_logits_threshold = 0.4
        assoc_indices = (asso_scores.squeeze() >= assoc_logits_threshold).nonzero(as_tuple=True)[0]
        # print("assoc_indices: ", sum((asso_scores.squeeze() >= assoc_logits_threshold)))

        # Use these indexes to extract associated masks and logits
        obj_scores = obj_scores[assoc_indices]
        obj_labels = obj_labels[assoc_indices]
        asso_scores = asso_scores[assoc_indices]
        object_mask_pred = object_mask_pred[assoc_indices]
        shadow_mask_pred = shadow_mask_pred[assoc_indices]


        ####################################################################

        # Create Instances, which contains the image size
        image_size = object_mask_pred.shape[-2:]

        results = []
        associations = []
        result = Instances(image_size)

        object_pred_masks = (object_mask_pred > 0).float()
        shadow_pred_masks = (shadow_mask_pred > 0).float()
        asso_pred_masks = torch.logical_or(object_pred_masks, shadow_pred_masks).float()

        object_pred_boxes = BitMasks(object_pred_masks).get_bounding_boxes()
        shadow_pred_boxes = BitMasks(shadow_pred_masks).get_bounding_boxes()
        asso_pred_boxes = BitMasks(asso_pred_masks).get_bounding_boxes()


        pred_mask = torch.cat([object_pred_masks, shadow_pred_masks], dim=0)

        pred_boxes = Boxes(torch.cat([object_pred_boxes.tensor, shadow_pred_boxes.tensor], dim=0))

        class_num = pred_mask.shape[0]
        # Create pred_classes
        object_classes = torch.zeros(class_num//2, dtype=torch.long)
        shadow_classes = torch.ones(class_num//2, dtype=torch.long)

        pred_classes = torch.cat([object_classes, shadow_classes], dim=0)
        # Create pred_associations
        pred_associations = torch.cat([torch.arange(1, class_num//2 + 1), torch.arange(1, class_num//2 + 1)]).tolist()

        scores = torch.cat([asso_scores, asso_scores], dim=0).flatten()

        result.pred_masks = pred_mask
        result.pred_classes = pred_classes
        result.pred_boxes = pred_boxes
        result.scores = scores
        result.pred_associations = pred_associations

        results.append({'instances': result})

        ####################################################################

        # Create association Instances, which contains the image dimensions
        association = Instances(image_size)

        pred_associations = torch.arange(1, class_num//2 + 1).tolist()
        association_pred_masks = asso_pred_masks
        association_pred_boxes = asso_pred_boxes

        pred_boxes = Boxes(association_pred_boxes.tensor.type(torch.int).cpu())
        pred_classes = np.zeros(class_num//2, dtype=int)
        scores = asso_scores.flatten().cpu().numpy()

        association.pred_masks = association_pred_masks.cpu().numpy()
        association.pred_associations = pred_associations
        association.pred_boxes = pred_boxes
        association.pred_classes = pred_classes
        association.scores = scores
        association.pred_associations = pred_associations

        associations.append({'instances': association})
        return results, associations