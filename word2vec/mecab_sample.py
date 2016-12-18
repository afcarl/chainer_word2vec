#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import sys
import MeCab


if __name__ == "__main__":
    m = MeCab.Tagger(" -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    message = "なのはちゃんがケチなのは仕方ない。"
    print(message)
    result = m.parse(message)
    print(result)



