# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample
from ..utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def calculate_iou(pred, target):
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def iou_loss(pred, target):
    iou = calculate_iou(pred, target)
    return 1.0 - iou.mean()

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, cost_association: float = 1, 
                    num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
            cost_association: Matching score cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_association = cost_association
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["association_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            ''' ###### class match ###### '''
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_association_prob = outputs['association_logits'][b].sigmoid()
            tgt_association_labels_permute = targets[b]['association_score'].permute(1, 0)
            
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_object_class = -out_prob[:, targets[b]['object_labels']]

            cost_association = -(out_association_prob.matmul(tgt_association_labels_permute) / \
                    (tgt_association_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                    (1 - out_association_prob).matmul(1 - tgt_association_labels_permute) / \
                    ((1 - tgt_association_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
        

            ''' ###### mask match ###### '''
            out_object_mask = outputs["object_pred_masks"][b]  # [num_queries, H_pred, W_pred]
            out_shadow_mask = outputs["shadow_pred_masks"][b]

            # gt masks are already padded when preparing target
            tgt_object_mask = targets[b]['object_masks'].to(out_object_mask)
            tgt_shadow_mask = targets[b]['shadow_masks'].to(out_shadow_mask)

            out_object_mask = out_object_mask[:, None]
            out_shadow_mask = out_shadow_mask[:, None]


            tgt_object_mask = tgt_object_mask[:, None]
            tgt_shadow_mask = tgt_shadow_mask[:, None]


            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_object_mask.device)
            # get gt labels
            tgt_object_mask = point_sample(
                tgt_object_mask,
                point_coords.repeat(tgt_object_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            tgt_shadow_mask = point_sample(
                tgt_shadow_mask,
                point_coords.repeat(tgt_shadow_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_object_mask = point_sample(
                out_object_mask,
                point_coords.repeat(out_object_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            out_shadow_mask = point_sample(
                out_shadow_mask,
                point_coords.repeat(out_shadow_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_object_mask = out_object_mask.float()
                tgt_object_mask = tgt_object_mask.float()
                out_shadow_mask = out_shadow_mask.float()
                tgt_shadow_mask = tgt_shadow_mask.float()

                # Compute the focal loss between masks
                cost_mask_object = batch_sigmoid_ce_loss_jit(out_object_mask, tgt_object_mask)
                cost_mask_shadow = batch_sigmoid_ce_loss_jit(out_shadow_mask, tgt_shadow_mask)

                # Compute the dice loss betwen masks
                cost_dice_object = batch_dice_loss_jit(out_object_mask, tgt_object_mask)
                cost_dice_shadow = batch_dice_loss_jit(out_shadow_mask, tgt_shadow_mask)

            ''' ###### structure mask match ###### '''
            out_object_structure_mask = outputs['object_structure_mask'][b]  # [num_queries, H_pred, W_pred]
            out_shadow_structure_mask = outputs['shadow_structure_mask'][b]

            # gt masks are already padded when preparing target
            tgt_object_structure_mask = targets[b]['object_structures'].to(out_object_structure_mask)
            tgt_shadow_structure_mask = targets[b]['shadow_structures'].to(out_shadow_structure_mask)

            out_object_structure_mask = out_object_structure_mask[:, None]
            out_shadow_structure_mask = out_shadow_structure_mask[:, None]
     

            tgt_object_structure_mask = tgt_object_structure_mask[:, None]
            tgt_shadow_structure_mask = tgt_shadow_structure_mask[:, None]


            # all masks share the same set of points for efficient matching!
            # get gt labels
            tgt_object_structure_mask = point_sample(
                tgt_object_structure_mask,
                point_coords.repeat(tgt_object_structure_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            tgt_shadow_structure_mask = point_sample(
                tgt_shadow_structure_mask,
                point_coords.repeat(tgt_shadow_structure_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_object_structure_mask = point_sample(
                out_object_structure_mask,
                point_coords.repeat(out_object_structure_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            out_shadow_structure_mask = point_sample(
                out_shadow_structure_mask,
                point_coords.repeat(out_shadow_structure_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_object_structure_mask = out_object_structure_mask.float()
                tgt_object_structure_mask = tgt_object_structure_mask.float()
                out_shadow_structure_mask = out_shadow_structure_mask.float()
                tgt_shadow_structure_mask = tgt_shadow_structure_mask.float()

                # Compute the focal loss between masks
                cost_structure_mask_object = batch_sigmoid_ce_loss_jit(out_object_structure_mask, tgt_object_structure_mask)
                cost_structure_mask_shadow = batch_sigmoid_ce_loss_jit(out_shadow_structure_mask, tgt_shadow_structure_mask)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask_object
                + self.cost_mask * cost_mask_shadow

                + self.cost_mask * cost_structure_mask_object
                + self.cost_mask * cost_structure_mask_shadow

                + self.cost_class * cost_object_class
                + self.cost_association * cost_association

                + self.cost_dice * cost_dice_object
                + self.cost_dice * cost_dice_shadow
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
