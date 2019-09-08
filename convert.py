import matplotlib
matplotlib.use('Agg')

import utils
import argparse
from openimages import OpenImages

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert Open Images annotations into MS Coco format')
    parser.add_argument('-p', dest='path', default="/media/robert/Daten/DATA/open_images",
                        help='path to openimages data', 
                        type=str)
    parser.add_argument("--prefix", dest='prefix', default="sub-",
                        help='file prefix to use. E.g. if subset file are to be converted',
                        type=str)
    args = parser.parse_args()
    return args


args = parse_args()
data_dir = args.path
annotation_dir = '{}{}'.format(data_dir, '/annotations')
prefix = args.prefix

for subset in ['val', 'test', 'train']:
    print('converting {} data'.format(subset))
    # Select corresponding image directory
    image_dir = '{}{}'.format(data_dir, "/" + subset)
    # Convert annotations
    utils.convert_openimages_subset(annotation_dir, image_dir, subset, prefix=prefix)

