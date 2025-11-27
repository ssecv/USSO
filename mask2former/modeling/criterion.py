# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging
from dct_encoding.mask_encoding import DctMaskEncoding
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from ..utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.dct_encoding = DctMaskEncoding(vec_dim=300, mask_size=128)

    def scale_instance_features(self, gt_center_object, gt_contour_points_object, original_hw, target_hw):
        # Calculate scale factors for height and width
        original_h, original_w = original_hw
        target_h, target_w = target_hw[-2:]

        scale_factor_h = target_h / original_h
        scale_factor_w = target_w / original_w
        
        # Scale the centers
        scaled_centers = gt_center_object * torch.tensor([scale_factor_w, scale_factor_h]).to(gt_center_object.device)
        
        # Scale the contour points
        scaled_contour_points = gt_contour_points_object * torch.tensor([scale_factor_w, scale_factor_h]).to(gt_contour_points_object.device)
        
        # Calculate scaled distances based on scaled centers and contour points
        scaled_distances = torch.norm(scaled_contour_points - scaled_centers.unsqueeze(1), dim=2)
        
        return scaled_centers, scaled_distances

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["object_labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes_s = torch.cat([t["shadow_labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    

    def dice_coefficient(self, x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss


    def loss_masks(self, outputs, targets, indices, num_masks):
        assert "object_pred_masks" in outputs
        assert "shadow_pred_masks" in outputs
        # assert "asso_pred_masks" in outputs

        losses = {}
        mask_type_lists = [
            "object_pred_masks", "shadow_pred_masks",  
            "object_structure_mask", "shadow_structure_mask",
        ]

        for mask_type in mask_type_lists:
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)
            src_masks = outputs[mask_type]
            src_masks = src_masks[src_idx]
            
            if mask_type == 'object_pred_masks':
                masks = [t["object_masks"] for t in targets]
            elif mask_type == 'shadow_pred_masks':
                masks = [t["shadow_masks"] for t in targets]
            elif mask_type == 'asso_pred_masks':
                masks = [t['association_masks'] for t in targets]

            elif mask_type == 'object_structure_mask':
                masks = [t["object_structures"] for t in targets]
            elif mask_type == 'shadow_structure_mask':
                masks = [t['shadow_structures'] for t in targets]
            elif mask_type == 'asso_structure_mask':
                masks = [t["asso_structures"] for t in targets]


            target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
            target_masks = target_masks.to(src_masks)
            target_masks = target_masks[tgt_idx]

            src_masks = src_masks[:, None]
            target_masks = target_masks[:, None]

            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

            losses[f'loss_mask_{mask_type}'] = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
            if mask_type in ["object_pred_masks", "shadow_pred_masks", ]:
                losses[f'loss_dice_{mask_type}'] = dice_loss_jit(point_logits, point_labels, num_masks)

            del src_masks
            del target_masks
        return losses


    # After bipartite graph matching, used as positive samples.
    def loss_contrastive(self, outputs, targets, indices, num_masks):
        temperature = 0.5
        epsilon = 1e-7
        losses = []
        masksize = 128

        # Resize masks
        object_structure_pred_results = outputs['object_structure_mask']
        shadow_structure_pred_results = outputs['shadow_structure_mask']
        resized_masks_object = F.interpolate(
            object_structure_pred_results,
            size=(masksize, masksize),
            mode="bilinear",
            align_corners=False,
        )
        resized_masks_shadow = F.interpolate(
            shadow_structure_pred_results,
            size=(masksize, masksize),
            mode="bilinear",
            align_corners=False,
        )
        resized_masks_object = (resized_masks_object > 0).float()
        resized_masks_shadow = (resized_masks_shadow > 0).float()

        for b, (pos_i, pos_j) in enumerate(indices):
            batch_resized_masks_object = resized_masks_object[b]
            batch_resized_masks_shadow = resized_masks_shadow[b]

            # Encode using DCT or any other encoding method
            # Example placeholder for encoding, replace with your encoding method
            dct_vectors_object = self.dct_encoding.encode(batch_resized_masks_object)
            dct_vectors_shadow = self.dct_encoding.encode(batch_resized_masks_shadow)

            # Normalize vectors for cosine similarity calculation
            dct_vectors_object_normalized = F.normalize(dct_vectors_object, dim=1)
            dct_vectors_shadow_normalized = F.normalize(dct_vectors_shadow, dim=1)

            # Compute cosine similarities for all pairs
            similarities_matrix = torch.mm(dct_vectors_object_normalized, dct_vectors_shadow_normalized.t())

            # Extract positive similarities based on indices
            positive_indices = (pos_i, pos_j)
            positive_similarities = similarities_matrix[positive_indices]

            # Create a mask for negatives
            negative_mask = torch.ones_like(similarities_matrix)
            negative_mask[positive_indices] = 0
            
            # Compute loss for positive samples
            positive_loss = -torch.log(torch.sigmoid(positive_similarities) / temperature + epsilon).mean()

            # Compute loss for negative samples
            negative_similarities = similarities_matrix * negative_mask
            negative_loss = -torch.log((1 - torch.sigmoid(negative_similarities) + epsilon) / temperature).mean()

            # Combine the losses
            combined_loss = positive_loss + negative_loss
            losses.append(combined_loss)

        loss_contrastive = torch.stack(losses).mean()

        return {'loss_contrastive': loss_contrastive}
    
    

    def loss_association(self, outputs, targets, indices, num_masks):
        assert 'association_logits' in outputs
        src_logits = outputs['association_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['association_score'][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.zeros_like(src_logits).float()
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_association_ce = self._neg_loss(src_logits, target_classes)

        losses = {'loss_association_ce': loss_association_ce}
        return losses

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'association': self.loss_association,
            "contrastive": self.loss_contrastive,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
