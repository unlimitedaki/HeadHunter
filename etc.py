import os
import argparse

def cs_result_filename(data_type,args):
    return "{}_{}_omcs_of_dataset.json".format(data_type,args.cs_mode)

def surfaceText_filename(data_type):
    return "{}_csqa_surfaceTexts.json".format(data_type)