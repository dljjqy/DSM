#!/usr/bin/env python
import argparse

class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--src',
                            type=str,
                            # default='',
                            help='source file.')
        parser.add_argument('--des',
                            type=str,
                            # default='',
                            help='destination file.')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


if __name__ == '__main__':
    args = Options().parse()
    print(args)
