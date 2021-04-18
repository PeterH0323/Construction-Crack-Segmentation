import os
import time
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
    for i, (input, size, name) in enumerate(test_loader):
        with torch.no_grad():
            input_var = input.cuda()
        start_time = time.time()
        output = model(input_var)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print(f'[{i + 1}/{total_batches}]  time: {time_taken * 1000:.4f} ms = {1 / time_taken:.1f} FPS')
        output = output.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        # Save the predict greyscale output for Cityscapes official evaluation
        # Modify image name to meet official requirement
        name[0] = name[0].rsplit('_', 1)[0] + '_predict'
        save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                     output_grey=True, output_color=True, gt_color=False)

        # 将推理出来的 mask 写到原图中并保存成新的图片
        original_file = os.path.join(args.image_input_path, f"{name[0].split('_predict')[0]}.jpg")
        if not os.path.exists(original_file):
            original_file = original_file.replace(".jpg", ".png")
            if not os.path.exists(original_file):
                FileNotFoundError(
                    f"{name[0].split('_predict')[0]}.jpg or {name[0].split('_predict')[0]}.png is not found !")

        img = cv2.imread(original_file)  # 原图路径
        mask = output

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

        img = img[:, :, ::-1]
        img[..., 2] = np.where(mask == 1, 255, img[..., 2])

        cv2.imwrite(f"{os.path.join(args.save_seg_dir, name[0] + '_img.png')}", img)


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
