import clip
import uuid
import imageio
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import torch
import numpy as np
from easydict import EasyDict
from PIL import Image
import matplotlib.pyplot as plt
import collections
import cv2
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import os


FLAGS = {
    "prompt_engineering": True,
    "this_is": True,
    "temperature": 100.0,
    "use_softmax": False,
}
FLAGS = EasyDict(FLAGS)
STANDARD_COLORS = ["White"]

# Parameters for drawing figure.
display_input_size = (10, 10)
fig_size_w = 35
# fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
mask_color = "red"

MAX_BOXES_TO_DRAW = 8  # @param {type:"integer"}
# NMS_THRESHOLD = 0.4  # @param {type:"slider", min:0, max:0.9, step:0.05}
NMS_THRESHOLD = 0.1  # @param {type:"slider", min:0, max:0.9, step:0.05}
MIN_RPN_SCORE_THRESH = 0.8  # @param {type:"slider", min:0, max:1, step:0.01}
MIN_BOX_AREA = 10  # @param {type:"slider", min:0, max:10000, step:1.0}
MAX_BOX_AREA = 3000  # @param {type:"slider", min:0, max:10000, step:1.0}


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


single_template = ["a photo of {article} {}."]

multiple_templates = [
    "There is {article} {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {article} {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of {article} {}.",
    "itap of my {}.",  # itap: I took a picture of
    "itap of the {}.",
    "a photo of {article} {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {article} {}.",
    "a good photo of the {}.",
    "a bad photo of {article} {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {article} {}.",
    "a bright photo of the {}.",
    "a dark photo of {article} {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {article} {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {article} {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {article} {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {article} {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {article} {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {article} {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {article} {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]


def nms(dets, scores, thresh, max_dets=1000):
    """Non-maximum suppression.
    Args:
      dets: [N, 4]
      scores: [N,]
      thresh: iou threshold. Float
      max_dets: int.
    """
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < max_dets:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        overlap = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]
    return keep


def draw_bounding_box_on_image(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    color="red",
    thickness=4,
    display_str_list=(),
    use_normalized_coordinates=True,
):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input "color".
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            xmin * im_width,
            xmax * im_width,
            ymin * im_height,
            ymax * im_height,
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_left = min(5, left)
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin


def draw_bounding_box_on_image_array(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    color="red",
    thickness=4,
    display_str_list=(),
    use_normalized_coordinates=True,
):
    """Adds a bounding box to an image (numpy array).

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
      image: a numpy array with shape [height, width, 3].
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_bounding_box_on_image(
        image_pil,
        ymin,
        xmin,
        ymax,
        xmax,
        color,
        thickness,
        display_str_list,
        use_normalized_coordinates,
    )
    np.copyto(image, np.array(image_pil))


def draw_mask_on_image_array(image, mask, color="red", alpha=0.4):
    """Draws mask on an image.

    Args:
      image: uint8 numpy array with shape (img_height, img_height, 3)
      mask: a uint8 numpy array of shape (img_height, img_height) with
        values between either 0 or 1.
      color: color to draw the keypoints with. Default is red.
      alpha: transparency value between 0 and 1. (default: 0.4)

    Raises:
      ValueError: On incorrect data type for image or masks.
    """
    if image.dtype != np.uint8:
        raise ValueError("`image` not of type np.uint8")
    if mask.dtype != np.uint8:
        raise ValueError("`mask` not of type np.uint8")
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError("`mask` elements should be in [0, 1]")
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "The image has spatial dimensions %s but the mask has "
            "dimensions %s" % (image.shape[:2], mask.shape)
        )
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(
        list(rgb), [1, 1, 3]
    )
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert("RGBA")
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert("L")
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert("RGB")))


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=0.5,
    agnostic_mode=False,
    line_thickness=1,
    groundtruth_box_visualization_color="black",
    skip_scores=False,
    skip_labels=False,
    mask_alpha=0.4,
    plot_color=None,
):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      instance_masks: a numpy array of shape [N, image_height, image_width] with
        values ranging between 0 and 1, can be None.
      instance_boundaries: a numpy array of shape [N, image_height, image_width]
        with values ranging between 0 and 1, can be None.
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box to be visualized
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      groundtruth_box_visualization_color: box color for visualizing groundtruth
        boxes
      skip_scores: whether to skip score when drawing a single detection
      skip_labels: whether to skip label when drawing a single detection

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_score_map = {}
    box_to_instance_boundaries_map = {}

    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ""
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in list(category_index.keys()):
                            class_name = category_index[classes[i]]["name"]
                        else:
                            class_name = "N/A"
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = "{}%".format(int(100 * scores[i]))
                    else:
                        float_score = ("%.2f" % scores[i]).lstrip("0")
                        display_str = "{}: {}".format(display_str, float_score)
                    box_to_score_map[box] = int(100 * scores[i])

                box_to_display_str_map[box].append(display_str)
                if plot_color is not None:
                    box_to_color_map[box] = plot_color
                elif agnostic_mode:
                    box_to_color_map[box] = "DarkOrange"
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
                        classes[i] % len(STANDARD_COLORS)
                    ]

    # Handle the case when box_to_score_map is empty.
    if box_to_score_map:
        box_color_iter = sorted(
            box_to_color_map.items(), key=lambda kv: box_to_score_map[kv[0]]
        )
    else:
        box_color_iter = box_to_color_map.items()

    # Draw all boxes onto image.
    for box, color in box_color_iter:
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            draw_mask_on_image_array(
                image, box_to_instance_masks_map[box], color=color, alpha=mask_alpha
            )
        if instance_boundaries is not None:
            draw_mask_on_image_array(
                image, box_to_instance_boundaries_map[box], color="red", alpha=1.0
            )
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates,
        )

    return image


def paste_instance_masks(masks, detected_boxes, image_height, image_width):
    """Paste instance masks to generate the image segmentation results.

    Args:
      masks: a numpy array of shape [N, mask_height, mask_width] representing the
        instance masks w.r.t. the `detected_boxes`.
      detected_boxes: a numpy array of shape [N, 4] representing the reference
        bounding boxes.
      image_height: an integer representing the height of the image.
      image_width: an integer representing the width of the image.

    Returns:
      segms: a numpy array of shape [N, image_height, image_width] representing
        the instance masks *pasted* on the image canvas.
    """

    def expand_boxes(boxes, scale):
        """Expands an array of boxes by a given scale."""
        # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227  # pylint: disable=line-too-long
        # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
        # whereas `boxes` here is in [x1, y1, w, h] form
        w_half = boxes[:, 2] * 0.5
        h_half = boxes[:, 3] * 0.5
        x_c = boxes[:, 0] + w_half
        y_c = boxes[:, 1] + h_half

        w_half *= scale
        h_half *= scale

        boxes_exp = np.zeros(boxes.shape)
        boxes_exp[:, 0] = x_c - w_half
        boxes_exp[:, 2] = x_c + w_half
        boxes_exp[:, 1] = y_c - h_half
        boxes_exp[:, 3] = y_c + h_half

        return boxes_exp

    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812  # pylint: disable=line-too-long
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    _, mask_height, mask_width = masks.shape
    scale = max((mask_width + 2.0) / mask_width, (mask_height + 2.0) / mask_height)

    ref_boxes = expand_boxes(detected_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
    segms = []
    for mask_ind, mask in enumerate(masks):
        im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        # Process mask inside bounding boxes.
        padded_mask[1:-1, 1:-1] = mask[:, :]

        ref_box = ref_boxes[mask_ind, :]
        w = ref_box[2] - ref_box[0] + 1
        h = ref_box[3] - ref_box[1] + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)

        mask = cv2.resize(padded_mask, (w, h))
        mask = np.array(mask > 0.5, dtype=np.uint8)

        x_0 = min(max(ref_box[0], 0), image_width)
        x_1 = min(max(ref_box[2] + 1, 0), image_width)
        y_0 = min(max(ref_box[1], 0), image_height)
        y_1 = min(max(ref_box[3] + 1, 0), image_height)

        im_mask[y_0:y_1, x_0:x_1] = mask[
            (y_0 - ref_box[1]) : (y_1 - ref_box[1]),
            (x_0 - ref_box[0]) : (x_1 - ref_box[0]),
        ]
        segms.append(im_mask)

    segms = np.array(segms)
    assert masks.shape[0] == segms.shape[0]
    return segms


def plot_mask(color, alpha, original_image, mask):
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(original_image)

    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(
        list(rgb), [1, 1, 3]
    )
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert("RGBA")
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert("L")
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    img_w_mask = np.array(pil_image.convert("RGB"))
    return img_w_mask


def display_image(path_or_array, size=(10, 10)):
    if isinstance(path_or_array, str):
        image = np.asarray(Image.open(open(image_path, "rb")).convert("RGB"))
    else:
        image = path_or_array

    plt.figure(figsize=size)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# use conda install tensorflow instead of pip!!!
class ViLDDetector:
    def __init__(
        self, clip_version="ViT-B/32", saved_model_dir="./image_path_v2"
    ) -> None:
        # torch.cuda.set_per_process_memory_fraction(0.9, None)
        clip_model, clip_preprocess = clip.load(clip_version)
        clip_model.cuda().eval()
        print(
            "Model parameters:",
            f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}",
        )
        print("Input resolution:", clip_model.visual.input_resolution)
        print("Context length:", clip_model.context_length)
        print("Vocab size:", clip_model.vocab_size)
        self.clip_model = clip_model
        # Extra prompt engineering: swap A with B for every (A, B) in list.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        session = tf.Session(
            graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options)
        )
        _ = tf.saved_model.loader.load(session, ["serve"], saved_model_dir)
        self.session = session

    def build_text_embedding(self, categories):
        if FLAGS.prompt_engineering:
            templates = multiple_templates
        else:
            templates = single_template

        run_on_gpu = torch.cuda.is_available()

        with torch.no_grad():
            all_text_embeddings = []
            print("Building text embeddings...")
            for category in tqdm(categories):
                texts = [
                    template.format(
                        processed_name(category["name"], rm_dot=True),
                        article=article(category["name"]),
                    )
                    for template in templates
                ]
                if FLAGS.this_is:
                    texts = [
                        "This is " + text
                        if text.startswith("a") or text.startswith("the")
                        else text
                        for text in texts
                    ]
                texts = clip.tokenize(texts)  # tokenize
                if run_on_gpu:
                    texts = texts.cuda()
                text_embeddings = self.clip_model.encode_text(
                    texts
                )  # embed with text encoder
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embedding = text_embeddings.mean(dim=0)
                text_embedding /= text_embedding.norm()
                all_text_embeddings.append(text_embedding)
            all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
            if run_on_gpu:
                all_text_embeddings = all_text_embeddings.cuda()
        return all_text_embeddings.cpu().numpy().T

    @staticmethod
    def box_to_coordinates(box):
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        # return (box[1]+box[3])/2, (box[0]+box[2])/2

    @staticmethod
    def coordinates_to_action(coordinates, image_shape):
        return coordinates[0] / image_shape, coordinates[1] / image_shape

    def detect(
        self,
        image_path,
        category_names,
        detection_path='./detections',
        vild_params=None,
        plot_on=False,
        prompt_swaps=[("block", "cube")],
    ):
        category_name_string = ";".join(category_names)
        #################################################################
        # Preprocessing categories and get params
        for a, b in prompt_swaps:
            category_name_string = category_name_string.replace(a, b)
        category_names = [x.strip() for x in category_name_string.split(";")]
        category_names = ["background"] + category_names
        categories = [
            {
                "name": item,
                "id": idx + 1,
            }
            for idx, item in enumerate(category_names)
        ]
        category_indices = {cat["id"]: cat for cat in categories}

        if vild_params is not None:
            (
                max_boxes_to_draw,
                nms_threshold,
                min_rpn_score_thresh,
                min_box_area,
                max_box_area,
            ) = vild_params
        else:
            (
                max_boxes_to_draw,
                nms_threshold,
                min_rpn_score_thresh,
                min_box_area,
                max_box_area,
            ) = (
                MAX_BOXES_TO_DRAW,
                NMS_THRESHOLD,
                MIN_RPN_SCORE_THRESH,
                MIN_BOX_AREA,
                MAX_BOX_AREA,
            )

        fig_size_h = min(max(5, int(len(category_names) / 2.5)), 10)

        #################################################################
        # Obtain results and read image
        (
            roi_boxes,
            roi_scores,
            detection_boxes,
            scores_unused,
            box_outputs,
            detection_masks,
            visual_features,
            image_info,
        ) = self.session.run(
            [
                "RoiBoxes:0",
                "RoiScores:0",
                "2ndStageBoxes:0",
                "2ndStageScoresUnused:0",
                "BoxOutputs:0",
                "MaskOutputs:0",
                "VisualFeatOutputs:0",
                "ImageInfo:0",
            ],
            feed_dict={
                "Placeholder:0": [
                    image_path,
                ]
            },
        )

        roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
        # no need to clip the boxes, already done
        roi_scores = np.squeeze(roi_scores, axis=0)

        detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
        scores_unused = np.squeeze(scores_unused, axis=0)
        box_outputs = np.squeeze(box_outputs, axis=0)
        detection_masks = np.squeeze(detection_masks, axis=0)
        visual_features = np.squeeze(visual_features, axis=0)

        image_info = np.squeeze(image_info, axis=0)  # obtain image info
        image_scale = np.tile(image_info[2:3, :], (1, 2))
        image_height = int(image_info[0, 0])
        image_width = int(image_info[0, 1])

        rescaled_detection_boxes = detection_boxes / image_scale  # rescale

        # Read image
        image = np.asarray(Image.open(open(image_path, "rb")).convert("RGB"))
        assert image_height == image.shape[0]
        assert image_width == image.shape[1]

        #################################################################
        # Filter boxes

        # Apply non-maximum suppression to detected boxes with nms threshold.
        nmsed_indices = nms(detection_boxes, roi_scores, thresh=nms_threshold)

        # Compute RPN box size.
        box_sizes = (
            rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]
        ) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

        # Filter out invalid rois (nmsed rois)
        valid_indices = np.where(
            np.logical_and(
                np.isin(np.arange(len(roi_scores), dtype=np.int), nmsed_indices),
                np.logical_and(
                    np.logical_not(np.all(roi_boxes == 0.0, axis=-1)),
                    np.logical_and(
                        roi_scores >= min_rpn_score_thresh,
                        np.logical_and(
                            box_sizes > min_box_area, box_sizes < max_box_area
                        ),
                    ),
                ),
            )
        )[0]

        detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
        detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
        detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
        detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
        rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][
            :max_boxes_to_draw, ...
        ]

        #################################################################
        # Compute text embeddings and detection scores, and rank results
        text_features = self.build_text_embedding(categories)

        raw_scores = detection_visual_feat.dot(text_features.T)
        if FLAGS.use_softmax:
            scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
        else:
            scores_all = raw_scores

        indices = np.argsort(
            -np.max(scores_all, axis=1)
        )  # Results are ranked by scores
        indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])

        #################################################################
        # Print found_objects
        found_objects = {}
        for a, b in prompt_swaps:
            category_names = [
                name.replace(b, a) for name in category_names
            ]  # Extra prompt engineering.
        for anno_idx in indices[0 : int(rescaled_detection_boxes.shape[0])]:
            scores = scores_all[anno_idx]
            if np.argmax(scores) == 0:
                continue
            found_object = category_names[np.argmax(scores)]
            if found_object == "background":
                continue
            print("Found a", found_object, "with score:", np.max(scores))
            cat = category_names[np.argmax(scores)]
            box = rescaled_detection_boxes[anno_idx]
            coordinates = self.box_to_coordinates(box)
            normalized_coordinates = coordinates[0] / image_height, coordinates[1] / image_width
            box = box.tolist()
            if cat not in found_objects:
                found_objects[cat] = {
                    "box": box,
                    "coordinates": coordinates,
                    "normalized_coordinates": normalized_coordinates,
                }
            else:
                previous_value = found_objects.pop(cat)
                if (
                    previous_value["normalized_coordinates"][1] < normalized_coordinates[1]
                ):  # y-axis in real-world (x in the image view)
                    if "on the left" not in cat:
                        new_previous_cat = cat + " on the left"
                        new_cat = cat + " on the right"
                    else:
                        new_previous_cat = cat
                        new_cat = cat.replace("on the left", "in the middle")
                else:
                    if "on the right" not in cat:
                        new_previous_cat = cat + " on the right"
                        new_cat = cat + " on the left"
                    else:
                        new_previous_cat = cat
                        new_cat = cat.replace("on the right", "in the middle")
                found_objects[new_previous_cat] = previous_value
                found_objects[new_cat] = {
                    "box": box,
                    "coordinates": coordinates,
                    "normalized_coordinates": normalized_coordinates,
                }

        if not plot_on:
            return found_objects

        print(found_objects)
        #################################################################
        # Plot detected boxes on the input image.
        ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
        processed_boxes = np.concatenate(
            [xmin, ymin, xmax - xmin, ymax - ymin], axis=-1
        )
        segmentations = paste_instance_masks(
            detection_masks, processed_boxes, image_height, image_width
        )

        overall_fig_size = (18, 24)
        if len(indices_fg) == 0:
            display_image(np.array(image), size=overall_fig_size)
            print("ViLD does not detect anything belong to the given category")

        else:
            # numbered_categories = [
            #     {
            #         "name": str(idx),
            #         "id": idx,
            #     }
            #     for idx in range(50)
            # ]
            # numbered_category_indices = {cat["id"]: cat for cat in numbered_categories}

            image_with_detections = visualize_boxes_and_labels_on_image_array(
                np.array(image),
                rescaled_detection_boxes[indices_fg],
                valid_indices[:max_boxes_to_draw][indices_fg],
                detection_roi_scores[indices_fg],
                # category_index= numbered_category_indices,
                category_index=category_indices,
                instance_masks=segmentations[indices_fg],
                use_normalized_coordinates=False,
                max_boxes_to_draw=max_boxes_to_draw,
                min_score_thresh=min_rpn_score_thresh,
                skip_scores=False,
                skip_labels=True,
            )

            # plt.figure(figsize=overall_fig_size)
            # plt.imshow(image_with_detections)
            # # plt.axis("off")
            # plt.title("ViLD detected objects and RPN scores.")
            # plt.show()
            if not os.path.exists(detection_path):
                os.makedirs(detection_path)
            uid = str(uuid.uuid4())
            uid_path = os.path.abspath(f"{detection_path}/{uid}.jpg") 
            imageio.imwrite(uid_path, image_with_detections)
            found_objects.update(uid_path=uid_path)

        return found_objects


class FakeViLDDetector:
    def detect(self, *args, **kwargs):
        found_objects = {
            "blue block": {
                "box": np.array(
                    [117.53889, 13.807173, 132.59515, 31.58398], dtype=np.float32
                ).tolist(),
                "coordinates": (125.0670166015625, 22.695575714111328),
                "normalized_coordinates": (0.6100830078125, 0.1107101254346894),
            },
            "red block on the left": {
                "box": np.array(
                    [74.45434, 27.625505, 89.988945, 43.785732], dtype=np.float32
                ).tolist(),
                "coordinates": (82.22164154052734, 35.70561981201172),
                "normalized_coordinates": (0.4010811782464748, 0.17417375518054498),
            },
            "red block on the right": {
                "box": np.array(
                    [93.52788, 71.56357, 107.83365, 85.7643], dtype=np.float32
                ).tolist(),
                "coordinates": (100.6807632446289, 78.66393280029297),
                "normalized_coordinates": (0.4911256743640434, 0.38372650146484377),
            },
        }
        return found_objects
