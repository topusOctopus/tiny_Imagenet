# reformat validation set in order to read it by pytorch ImageFolder
import os
import shutil
import argparse
import pandas as pd

from logger import logger


def create_folders(args, folders_names):
    """
    Create classes folders from given path and folder names.
    :param args: Element from argparse. Here used args.d (str). Path to directory with data.
    :param folders_names: (list). List of strings, consists of classes names
    :return:
    """

    for i in range(len(folders_names)):
        add_path = folders_names[i]
        new_path = os.path.join(args.d, "new_val", add_path)
        os.mkdir(new_path)


def change_files_dir(args, val_annotations):
    """
    Move images from one directory to another
    :param args: Element from argparse. Here used args.d (str). Path to directory with data.
    :param val_annotations: (pandas df). Info about img name and it's class name
    :return:
    """

    for i in range(len(val_annotations['pic_name'])):
        file_path = os.path.join(args.d, "val/images", val_annotations['pic_name'][i])
        destination_path = os.path.join(args.d, "new_val", val_annotations['folder_name'][i])
        shutil.move(file_path, destination_path)


def main():
    parser = argparse.ArgumentParser(description='Args to preprocess validation data')
    parser.add_argument('-d', type=str, default="../data/tiny-imagenet-200",
                        help="Path to directory with data")
    args = parser.parse_args()

    val_annotations = pd.read_csv(os.path.join(args.d, "val/val_annotations.txt"),
                                  sep='\t', header=None, names=('pic_name', 'folder_name', '1', '2', '3', '4'))
    val_annotations = val_annotations.drop(columns=['1', '2', '3', '4'])
    folders_names = val_annotations.folder_name.unique()

    if not os.path.isdir(os.path.join(args.d, "new_val")):
        os.mkdir(os.path.join(args.d, "new_val"))
    if not os.listdir(os.path.join(args.d, "new_val")):
        create_folders(args, folders_names)
    else:
        logger.info('not empty')
    if not os.listdir(os.path.join(args.d, "val/images")):
        pass
    else:
        change_files_dir(args, val_annotations)


if __name__ == '__main__':
    main()
