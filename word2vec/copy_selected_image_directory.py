#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import util
import os
import shutil


def dir_name_generator(path):
    for line in open(path):
        items = line.strip().split()
        yield items[0]


if __name__ == "__main__":
    try:
        # set command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--selected_list_path", help="input: set a path to a selected list file")
        parser.add_argument("--src_dir_path", help="input: set a path to a source directory")
        parser.add_argument("--dst_dir_path", help="output: set a path to a destination directory")

        # parse arguments
        args = parser.parse_args()
        selected_list_path = args.selected_list_path
        src_dir_path = args.src_dir_path
        dst_dir_path = args.dst_dir_path

        # check paths
        util.check_input_path(selected_list_path)
        util.check_input_dir_path(src_dir_path)
        if not os.path.exists(dst_dir_path):
            os.mkdir(dst_dir_path)

        for dir_name in dir_name_generator(selected_list_path):
            src_image_dir_path = os.path.join(src_dir_path, dir_name)
            dst_image_dir_path = os.path.join(dst_dir_path, dir_name)
            shutil.copytree(src_image_dir_path, dst_image_dir_path)

    except IOError, e:
        print(e)
