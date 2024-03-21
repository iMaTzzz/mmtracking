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

    frame_id = data['img_metas'][0][0].pop('frame_id', None)
    print(f"frame_id={frame_id}")
    data.pop('img_metas')
    print(f"pre_data={data}")
    data['img'] = data['img'][0].detach().cpu()
    data['gt_bboxes'] = data['gt_bboxes'][0].detach().cpu()
    print(f"post_data={data}")
    dummy_img = torch.randn(1, 3, 224, 224)  # Example shape: (batch_size=1, channels=3, height=224, width=224)
    dummy_bbox = torch.tensor([0, 0, 100, 100])  # Example bbox, shape: (4, )
    dummy_z_feat = (torch.randn(1, 64, 32, 32), torch.randn(1, 128, 16, 16))  # Example shapes
    dummy_avg_channel = torch.tensor([0.5, 0.5, 0.5])  # Example avg_channel, shape: (3, )
    dynamic_axes = {'input0': {2: 'height', 3: 'width'},
                    'input1': {0: 'tl[x]', 1: 'tl[y]', 2: 'width', 3: 'height'}}
    #torch.onnx.export(model, (dummy_img, dummy_bbox, dummy_z_feat, dummy_avg_channel), "object_tracking_model.onnx", verbose=True, dynamic_axes=dynamic_axes)
    #for key, value in data.items():
        ## Wrap the variable in a tuple to make it a single input argument
        #print(f"key={key}, value={value}")
        ## Export the model
        #if key == 'img_metas':
            #try:
                #input_data = value[0][0]
                #print(f"Modified input_data={input_data}")
                #for k, v in input_data.items():
                    #print(f"key={k}, value={v}")
                    #torch.onnx.export(model, v, f"model_with_{k}.onnx", verbose=True)
            #except Exception as e:
                #print(f"Error exporting model with {k}: {e}")
        #else:
            #try:
                #torch.onnx.export(model, value, f"model_with_{key}.onnx", verbose=True)
            #except Exception as e:
                #print(f"Error exporting model with {key}: {e}")
        #print(f"Model exported successfully with {key}.")
    torch.onnx.export(model, data, "object_tracking_model.onnx", verbose=True)


if __name__ == '__main__':
    main()
