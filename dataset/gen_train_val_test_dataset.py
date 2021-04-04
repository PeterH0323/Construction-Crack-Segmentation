# -*- coding: utf-8 -*-
# @Time    : 2021/4/4 10:16
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : gen_train_val_test_dataset.py
# @Software: PyCharm
# @Brief   : 模仿 cityscape 数据结构生成数据集
import os
import shutil
from pathlib import Path
import random


def split_dataset(image_path: Path, label_path: Path, save_path: Path, factor=(0.8, 0.1, 0.1)):
    """
    在总数据集中，生成 train，val，test 数据集
    :param image_path: 总数据集图片位置
    :param label_path: 总数据集标签位置
    :param save_path: 最终生成新数据集的位置
    :param factor: 比率：(训练集占比，验证集占比，测试集占比)
    :return:
    """
    print("Ready to split dataset...")
    image_list = [item for item in image_path.iterdir()]  # 得出图片绝对路径list
    random.shuffle(image_list)  # 打乱数组

    # 根据比率得出数组中的结束索引号
    train_factor, val_factor, test_factor = factor
    train_max_index = int(len(image_list) * train_factor)
    val_max_index = train_max_index + int(len(image_list) * val_factor)

    # 得出新的 list
    train_list = image_list[:train_max_index]
    val_list = image_list[train_max_index:val_max_index]
    test_list = image_list[val_max_index:]

    # 进行新数据集的生成
    os.makedirs(str(save_path), exist_ok=True)

    new_image_root = save_path.joinpath("Images")
    if new_image_root.exists():
        # 清空文件
        print(f"Cleaning {new_image_root.name}, pls wait...")
        shutil.rmtree(new_image_root)

    os.makedirs(str(new_image_root))
    new_label_root = save_path.joinpath("Labels")
    if new_label_root.exists():
        # 清空文件
        print(f"Cleaning {new_label_root.name}, pls wait...")
        shutil.rmtree(new_label_root)
    os.makedirs(str(new_label_root))

    # 移动文件
    dataset_type = ['train', 'val', 'test']
    index = 0
    for new_image_list in [train_list, val_list, test_list]:
        dataset_file_type = dataset_type[index]
        print(f"Generating {dataset_file_type} dataset, pls wait...")
        index += 1
        txt_path = save_path.joinpath(f"custom_dataset_{dataset_file_type}_list.txt")
        if txt_path.exists():
            txt_path.unlink()
        label_save_path = new_label_root.joinpath(dataset_file_type)
        label_save_path.mkdir(exist_ok=True)
        image_save_path = new_image_root.joinpath(dataset_file_type)
        image_save_path.mkdir(exist_ok=True)

        with open(str(txt_path), "a+", encoding='UTF-8') as f:
            for image in new_image_list:
                shutil.copy(str(image), str(image_save_path.joinpath(image.name)))

                label_suffix = '.jpg'
                if not label_path.joinpath(image.name).with_suffix(label_suffix).exists():
                    label_suffix = '.png'

                shutil.copy(str(label_path.joinpath(image.stem).with_suffix(label_suffix)),
                            str(label_save_path.joinpath(image.name).with_suffix(label_suffix)))

                f.write(f"Images/{dataset_file_type}/{image.name} "
                        f"Labels/{dataset_file_type}/{image.with_suffix(label_suffix).name}\n")

    # 生成 trainvl txt
    with save_path.joinpath(f"custom_dataset_train_list.txt").open(encoding='UTF-8') as f_train:
        final_txt = f_train.read()
    with save_path.joinpath(f"custom_dataset_val_list.txt").open(encoding='UTF-8') as f_val:
        final_txt += f_val.read()
    with save_path.joinpath(f"custom_dataset_trainval_list.txt").open("w", encoding='UTF-8') as f_trainaval:
        f_trainaval.write(final_txt)

    print(f"All done, saved in {save_path}")


if __name__ == '__main__':
    split_dataset(image_path=Path(r"C:\Users\PeterH\Desktop\1-200_0402\Images"),
                  label_path=Path(r"C:\Users\PeterH\Desktop\1-200_0402\Labels"),
                  save_path=Path(r"C:\Users\PeterH\Desktop\1-200_0402\new_dataset")
                  )
