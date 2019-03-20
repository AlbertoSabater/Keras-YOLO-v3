#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:40:10 2019

@author: asabater
"""

classes = ['basket',
             'bed',
             'blanket',
             'book',
             'bottle',
             'cell',
             'cell_phone',
             'cloth',
             'comb',
             'container',
             'dent_floss',
             'detergent',
             'dish',
             'door',
             'electric_keys',
             'food/snack',
             'fridge',
             'kettle',
             'keyboard',
             'knife/spoon/fork',
             'laptop',
             'large_container',
             'microwave',
             'milk/juice',
             'monitor',
             'mop',
             'mug/cup',
             'oven/stove',
             'pan',
             'perfume',
             'person',
             'pills',
             'pitcher',
             'shoe',
             'shoes',
             'soap_liquid',
             'tap',
             'tea_bag',
             'thermostat',
             'tooth_brush',
             'tooth_paste',
             'towel',
             'trash_can',
             'tv',
             'tv_remote',
             'vacuum',
             'washer/dryer']


def main():
    # write classes to file
    with open('adl_classes.txt', 'w+') as f:
        for c in classes:
            f.write(c + '\n')
            
            
if __name__ == '__main__':
    main()
