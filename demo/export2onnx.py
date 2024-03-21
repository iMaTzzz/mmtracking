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
    # print(model)

    if args.gt_bbox_file is not None:
        bboxes = mmcv.list_from_file(args.gt_bbox_file)
        init_bbox = list(map(float, bboxes[0].split(',')))
        # convert (x1, y1, w, h) to (x1, y1, x2, y2)
        init_bbox[2] += init_bbox[0]
        init_bbox[3] += init_bbox[1]

    for i, img in enumerate(imgs):
        if i == 0:
            result, data = inference_sot(model, img, init_bbox, frame_id=i)
            break

    scaling_factor = data['img_metas'][0][0].pop('scale_factor', None)
    print(f"scaling_factor={scaling_factor}")
    print(f"data={data}")
    dummy_img = torch.randn(1, 3, 224, 224)  # Example shape: (batch_size=1, channels=3, height=224, width=224)
    dummy_bbox = torch.tensor([0, 0, 100, 100])  # Example bbox, shape: (4, )
    dummy_z_feat = (torch.randn(1, 64, 32, 32), torch.randn(1, 128, 16, 16))  # Example shapes
    dummy_avg_channel = torch.tensor([0.5, 0.5, 0.5])  # Example avg_channel, shape: (3, )
    dynamic_axes = {'input0': {2: 'height', 3: 'width'},
                    'input1': {0: 'tl[x]', 1: 'tl[y]', 2: 'width', 3: 'height'}}
    #torch.onnx.export(model, (dummy_img, dummy_bbox, dummy_z_feat, dummy_avg_channel), "object_tracking_model.onnx", verbose=True, dynamic_axes=dynamic_axes)
    for key, value in data.items():
        # Wrap the variable in a tuple to make it a single input argument
        input_data = (value,)
        try:
            # Export the model
            if key == 'img_metas':
                for k, v in input_data.items():
                    torch.onnx.export(model, v, f"model_with_{key}.onnx", verbose=True)
            else:
                torch.onnx.export(model, input_data, f"model_with_{key}.onnx", verbose=True)
            print(f"Model exported successfully with {key}.")
        except Exception as e:
            print(f"Error exporting model with {key}: {e}")
    # torch.onnx.export(model, data, "object_tracking_model.onnx", verbose=True)


if __name__ == '__main__':
    main()
