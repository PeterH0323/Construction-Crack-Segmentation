import os
import time
from pathlib import Path

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test, build_dataset_predict
from utils.utils import save_predict
from utils.convert_state import convert_state_dict
import cv2


def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', default="ENet", help="model name: (default ENet)")
    parser.add_argument('--dataset', default="custom_dataset", help="dataset: cityscapes, camvid or custom_dataset")
    parser.add_argument('--image_input_path', default="./inference_images/input_images", help="load predict_image")
    parser.add_argument('--num_workers', type=int, default=2, help="the number of parallel threads")
    parser.add_argument('--use_txt_list', type=bool, default=False, help="Using txt list in dataset files")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,
                        default=r"",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./inference_images/predict_output/",
                        help="saving path of prediction result")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    return args


def predict(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset, for those that do not provide label on the test set
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
    vid_writer = None
    vid_path = None

    for i, (input, size, name, mode, frame_count, img_original, vid_cap) in enumerate(test_loader):
        with torch.no_grad():
            input = input[None, ...]  # 增加多一个维度
            input = torch.tensor(input)  # [1, 3, 224, 224]
            input_var = input.cuda()
        start_time = time.time()
        output = model(input_var)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print(f'[{i + 1}/{total_batches}]  time: {time_taken * 1000:.4f} ms = {1 / time_taken:.1f} FPS')
        output = output.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        save_name = Path(name).stem + f'_predict'
        if mode == 'images':
            # 保存图片推理结果
            save_predict(output, None, save_name, args.dataset, args.save_seg_dir,
                         output_grey=True, output_color=True, gt_color=False)

        # 将结果和原图画到一起
        img = img_original
        mask = output
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
        img = img[:, :, ::-1]
        img[..., 2] = np.where(mask == 1, 255, img[..., 2])

        if mode == 'images':
            # 保存 推理+原图 结果
            cv2.imwrite(f"{os.path.join(args.save_seg_dir, save_name + '_img.png')}", img)
        else:
            # 保存视频
            save_path = os.path.join(args.save_seg_dir, save_name + '_predict.mp4')
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img)


def predict_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes)

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)

    # load the test set
    if args.use_txt_list:
        _, testLoader = build_dataset_test(args.dataset, args.num_workers, none_gt=True)
    else:
        _, testLoader = build_dataset_predict(args.image_input_path, args.dataset, args.num_workers, none_gt=True)

    print("=====> beginning testing")
    print("test set length: ", len(testLoader))
    predict(args, testLoader, model)


if __name__ == '__main__':

    args = parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, 'predict', args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    elif args.dataset == 'custom_dataset':
        args.classes = 2
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    predict_model(args)
