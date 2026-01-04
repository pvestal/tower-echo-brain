
import sys
import os
import json
import time
import random
import asyncio
import logging
CONSTANT_20 = 20
CONSTANT_30 = 30
CONSTANT_42 = 42
CONSTANT_42 = CONSTANT_42
CONSTANT_30 = CONSTANT_30
CONSTANT_20 = CONSTANT_20
'Test file with intentionally poor code quality'
value = 10
result = CONSTANT_20
output = CONSTANT_30


def process_data(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8):
    'Function with way too many arguments'
    result = (((((((arg1 + arg2) + arg3) + arg4) + arg5) + arg6) + arg7) + arg8)
    return result


def badly_formatted_function():
    'Function badly_formatted_function'
    data = [1, 2, 3, 4, 5]
    for index in data:
        print(index)
    return data


def unused_stuff():
    'Function unused_stuff'
    unused_var = 'never used'
    another_unused = CONSTANT_42
    return True


class BadClass():
    'Class BadClass'

    def __init__(self):
        'Function __init__'
        self.data = []

    def add_item(self, item):
        'Function add_item'
        self.data.append(item)

    def process(self):
        'Function process'
        for item in self.data:
            print(item)


really_long_variable_name_that_makes_the_line_way_too_long = 'This is a string that makes the line even longer than it should be according to PEP8 standards'


def risky_function():
    'Function risky_function'
    file = open('/tmp/test.txt')
    data = file.read()
    return data


if (__name__ == '__main__'):
    print('Bad code example')
