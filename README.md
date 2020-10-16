download google Open Image dataset each images and each annotation. And convert to CoCo format.

# open images downloader
This repo combines the power of openimages2coco with an open images downloader.
Download a subset of [Open Images](https://storage.googleapis.com/openimages/web/index.html "Open Images Homepage") files based on labels and convert them to [MS Coco](http://cocodataset.org "MS Coco Homepage") annotation format.
I have to thank the authors of openimages2coco and pytorch-ssd for the base code of this repo.

### Functionality

This conversion routine will load the original .csv annotation files form Open Images, convert the annotations into the list/dict based format of [MS Coco annotations](http://cocodataset.org/#format-data) and store them as a .json file in the same folder.


### Download subset of data with bounding boxes

```bash
python open_images_downloader.py --root /media/robert/Daten/DATA/OI/face_plate --class_names "Vehicle registration plate, Traffic sign, Human face, Street light" --num_workers 20
```

It will download data into the folder ~/data/open_images.

The content of the data directory looks as follows.

```
class-descriptions-boxable.csv       test                        validation
sub-test-annotations-bbox.csv        test-annotations-bbox.csv   validation-annotations-bbox.csv
sub-train-annotations-bbox.csv       train
sub-validation-annotations-bbox.csv  train-annotations-bbox.csv
```

The folders train, test, validation contain the images. The files like sub-train-annotations-bbox.csv 
is the annotation file.

### conversion

Download the CocoAPI from https://github.com/cocodataset/cocoapi \
Install Coco API:
```
cd PATH_TO_COCOAPI/PythonAPI
make install
```

Download Open Images from https://storage.googleapis.com/openimages/web/download.html \
-> Store the images in three folders called: ```train, val and test``` \
-> Store the annotations for all three splits in a separate folder called: ```annotations```

Run conversion:
```
ptyhon convert.py -p PATH_TO_OPENIMAGES
python convert.py -p /media/robert/Daten/DATA/OI/face_plate
```

### Output

The generated annotations can be loaded and used with the standard MS Coco tools:
```
from pycocotools.coco import COCO

# Example for the validation set
openimages = COCO('PATH_TO_OPENIMAGES/annotations/val-annotations-bbox.json')
```

### Issues
- The evaluation tools from the Coco API are not yet working with the converted Open Images annotations
- A few images are in a weird format that returns a list [image, some_annotations] when loaded with skimage.io.imread(image). These seem to be corrupted .jpg files and we will contact the data set developers on this issue. For the moment something like the following function can be used to catch all possible formatting issues including RGBA and monochrome images:
```
def load_image(self, image_id):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(self.image_info[image_id]['path'])
    # If image has additional annotations
    if image.ndim == 1:
        image = image[0]
    # If grayscale. Convert to RGB for consistency.
    if image.ndim == 2:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image
```
