# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser
import torch

import cv2
import mmcv

from mmtrack.apis import inference_sot, init_model


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('--input', help='input video file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--gt_bbox_file', help='The path of gt_bbox file')
    args = parser.parse_args()

    imgs = mmcv.VideoReader(args.input)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    print(model)

    if args.gt_bbox_file is not None:
        bboxes = mmcv.list_from_file(args.gt_bbox_file)
        init_bbox = list(map(float, bboxes[0].split(',')))
        # convert (x1, y1, w, h) to (x1, y1, x2, y2)
        init_bbox[2] += init_bbox[0]
        init_bbox[3] += init_bbox[1]

    for i, img in enumerate(imgs):
        if i == 0:
            result, data = inference_sot(model, img, init_bbox, frame_id=i)
            print(f"result={result}")
            break

    print(data)
    dummy_input = torch.randn(1, 3, 224, 224)  # Example shape: (batch_size, channels, height, width)
    dummy_img_metas = torch.randn(1, 3)  # Example shape: (batch_size, num_img_metas)
    torch.onnx.export(model, (dummy_input, dummy_img_metas), "object_tracking_model.onnx", verbose=True, dynamic_axes={'input': {2: 'height', 3: 'width'}})


if __name__ == '__main__':
    main()
