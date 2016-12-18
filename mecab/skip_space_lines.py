#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
FILE_PATH = "/home/ubuntu/data/word2vec/jawiki-wakati.txt"
OUT_PATH = "/home/ubuntu/data/word2vec/jawiki-wakati_without_spaces.txt"

if __name__ == "__main__":
    outf = open(OUT_PATH, "w")
    for line in open(FILE_PATH):
        line = line.strip()
        if line is not "":
            outf.write("{i}\n".format(i=line))
            
