import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_iou(boxes1_xyxy, boxes2_xyxy):
    """Compute pairwise IoU for two box sets in [x1, y1, x2, y2] format.

    Args:
        boxes1_xyxy: Tensor of shape [N, 4].
        boxes2_xyxy: Tensor of shape [M, 4].

    Returns:
        Tensor of shape [N, M] containing pairwise IoUs.
    """
    num_boxes1 = boxes1_xyxy.size(0)
    num_boxes2 = boxes2_xyxy.size(0)

    top_left = torch.max(
        boxes1_xyxy[:, :2].unsqueeze(1).expand(num_boxes1, num_boxes2, 2),
        boxes2_xyxy[:, :2].unsqueeze(0).expand(num_boxes1, num_boxes2, 2),
    )
    bottom_right = torch.min(
        boxes1_xyxy[:, 2:].unsqueeze(1).expand(num_boxes1, num_boxes2, 2),
        boxes2_xyxy[:, 2:].unsqueeze(0).expand(num_boxes1, num_boxes2, 2),
    )

    intersection_wh = (bottom_right - top_left).clamp(min=0)
    intersection_area = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (
        boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]
    )
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (
        boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]
    )
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection_area

    return intersection_area / union_area.clamp(min=1e-6)


class YOLOLoss(nn.Module):
    """YOLO-style detection loss for the MP4 object detector.

    Each grid cell predicts B boxes followed by C class scores:

    [x, y, w, h, confidence] * B + [class_1, ..., class_C]

    The loss has three terms (see Lecture 12):

    1. Regression:
       lambda_coord * 1_ij^obj * ((x - x_hat)^2 + (y - y_hat)^2)
       lambda_coord * 1_ij^obj * ((sqrt(w) - sqrt(w_hat))^2 + (sqrt(h) - sqrt(h_hat))^2)
    2. Object / no-object confidence:
       1_ij^obj * (C - C_hat)^2 + lambda_noobj * 1_ij^noobj * (C - C_hat)^2
    3. Class prediction:
       1_i^obj * sum_c (p_i(c) - p_hat_i(c))^2

    1_ij^obj = 1 for the single predictor j in cell i that has the highest IoU
    with the ground-truth box.
    """

    def __init__(self, grid_size, boxes_per_cell, lambda_coord, lambda_noobj):
        """
        Args:
            grid_size: Number of cells per spatial dimension (S in the slide).
            boxes_per_cell: Number of box predictors per cell (B in the slide).
            lambda_coord: Weight on the box regression terms.
            lambda_noobj: Weight on the no-object confidence term.
        """
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def split_prediction_tensor(self, prediction_tensor):
        """Split the raw model output into box predictions and class scores.

        Args:
            prediction_tensor: Tensor of shape [N, S, S, B * 5 + C].

        Returns:
            (predicted_boxes, predicted_class_scores) where predicted_boxes has
            shape [N, S, S, B, 5] (each row is [x, y, w, h, confidence]) and
            predicted_class_scores has shape [N, S, S, C].
        """
        pass

    def xywh_to_xyxy(self, boxes_xywh, col_row):
        """Convert boxes from center-size form to corner form.

        Args:
            boxes_xywh: Tensor of shape [N, 4] with boxes as
                [x_center, y_center, width, height], where x and y are
                within-cell offsets in [0, 1].
            col_row: Tensor of shape [N, 2] containing the (col, row) grid
                index of each box's cell, used to recover the absolute
                image-normalized center via (col + x) / S.

        Returns:
            Tensor of shape [N, 4] with boxes as [x1, y1, x2, y2] in
            image-normalized coordinates.

        Example:
            A box in cell (col=3, row=2) with offset (0.5, 0.5) and size
            (0.3, 0.4) (with S=14) becomes center ((3.5)/14, (2.5)/14) and
            corners (3.5/14 - 0.3/2, 2.5/14 - 0.4/2, 3.5/14 + 0.3/2, 2.5/14 + 0.4/2).
        """
        # TODO: Return boxes in corner form from boxes in center-size form.
        # Input x and y are offsets within the cell; col_row gives the cell's grid position.
        pass

    def choose_responsible_box(self, predicted_boxes, target_boxes, has_object_mask):
        """Find the responsible predictor for each cell that contains an object.

        Args:
            predicted_boxes: Tensor of shape [N, S, S, B, 5].
            target_boxes: Tensor of shape [N, S, S, 4].
            has_object_mask: Boolean tensor of shape [N, S, S].

        Returns:
            (responsible_box_predictions, responsible_box_ious,
            responsible_predictor_index) with shapes [num_object_cells, 5],
            [num_object_cells, 1], and [num_object_cells].

        The responsible predictor is whichever of the B predictors has the
        highest IoU with the ground-truth box in that cell.
        """
        # TODO: Use has_object_mask to filter predicted and target boxes to cells
        # that contain objects.

        # TODO: Edge case: no object cells in batch.

        # TODO: Get (col, row) grid indices for cells with objects, and use
        # self.xywh_to_xyxy to convert predicted and target boxes to corner form
        # in image-normalized coordinates for IoU computation. Loop over the B
        # predictors to compute their IoUs with the GT box and find the max.
        # Note that compute_iou returns an [N, N] matrix of pairwise IoUs, but you
        # only need the part comparing each predictor to its own cell's GT box.

        # TODO: Return the responsible predictor's box predictions, IoU with the GT
        # box, and predictor index within the cell.
        pass

    def build_responsible_mask(
        self,
        predicted_boxes,
        has_object_mask,
        responsible_predictor_index,
    ):
        """Build a [N, S, S, B] boolean mask representing 1_ij^obj in the formula.

        The mask is True for the single responsible predictor in each object
        cell (the one with the highest IoU with the ground-truth box) and
        False everywhere else, including all predictors in empty cells.
        """
        # TODO: Create a responsible boolean mask. Remember to account for the edge
        # case where there are no object cells.
        pass

    def regression_xy_loss(
        self,
        responsible_box_predictions,
        target_boxes_for_object_cells,
    ):
        """Sum of squared errors on (x, y) for the responsible predictor in each object cell.

        sum_i sum_j 1_ij^obj * ((x_i - x_hat_i)^2 + (y_i - y_hat_i)^2)

        Note: The 1_ij^obj factor is implicit, since both inputs have already been
        filtered down to the single responsible predictor in each object cell.
        """
        pass

    def regression_wh_loss(
        self,
        responsible_box_predictions,
        target_boxes_for_object_cells,
    ):
        """Sum of squared errors on sqrt(w), sqrt(h) for the responsible predictor.

        sum_i sum_j 1_ij^obj * ((sqrt(w_i) - sqrt(w_hat_i))^2 +
                                 (sqrt(h_i) - sqrt(h_hat_i))^2)

        Note: The 1_ij^obj factor is implicit, similarly to the xy loss.

        The square root makes the loss scale-invariant: a 2px error on a 10px
        box is penalized more than the same error on a 100px box.
        """
        # TODO: Regress sqrt(w) and sqrt(h). Clamp to 1e-6 before sqrt, since early
        # in training w/h can go slightly negative, and sqrt of a negative number gives NaN.
        pass

    def object_confidence_loss(self, responsible_box_predictions, responsible_box_ious):
        """Sum of squared errors on confidence for predictors assigned to real objects.

        sum_i sum_j 1_ij^obj * (C_i - C_hat_i)^2

        The confidence target is the IoU between the predicted box and the
        ground-truth box (detached, so it acts as a fixed regression target).
        """
        # TODO: The confidence target isn't 1.0 for matched boxes; it's the IoU with
        # the GT box. Use responsible_box_ious as the target and detach it; otherwise
        # gradients flow back through the IoU computation and training becomes unstable.
        pass

    def no_object_confidence_loss(self, predicted_boxes, responsible_mask):
        """Sum of squared errors on confidence for all non-responsible predictors.

        sum_i sum_j 1_ij^noobj * (C_i - C_hat_i)^2

        Covers both empty cells and the losing predictors in object cells.
        The target confidence is 0 for all of these.
        """
        pass

    def class_probability_loss(
        self,
        predicted_class_scores,
        target_class_scores,
        has_object_mask,
    ):
        """Sum of squared errors in class probabilities, only for cells that contain an object.

        sum_i 1_i^obj * sum_c (p_i(c) - p_hat_i(c))^2

        Class scores are per cell, not per predictor box.
        """
        pass

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """Compute the full YOLO loss and return each named component.

        Args:
            pred_tensor: Model output of shape [N, S, S, B * 5 + C].
            target_boxes: Ground-truth boxes of shape [N, S, S, 4].
            target_cls: One-hot class targets of shape [N, S, S, C].
            has_object_map: Boolean tensor of shape [N, S, S].

        Returns:
            Dict with total_loss and the individual components: reg_loss,
            reg_xy_loss, reg_wh_loss, obj_loss, no_obj_loss, cls_loss.
        """
        # TODO: Step 1: Split pred_tensor into predicted_boxes and predicted_class_scores.

        # TODO: Step 2: Find the responsible predictor for each object cell and
        # build the responsible mask (1_ij^obj).

        # TODO: Step 3: Compute regression loss terms.

        # TODO: Step 4: Compute confidence loss terms for object and no-object predictors.

        # TODO: Step 5: Compute class probability.

        # TODO: Step 6: Scale the terms by the appropriate weights, and divide
        # every term by batch_size to normalize.

        return {
            "total_loss": ...,
            "reg_loss": ...,
            "reg_xy_loss": ...,
            "reg_wh_loss": ...,
            "obj_loss": ...,
            "no_obj_loss": ...,
            "cls_loss": ...,
        }
