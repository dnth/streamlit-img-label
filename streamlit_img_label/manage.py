import os
import re
import numpy as np
from PIL import Image
from .annotation import output_xml, read_xml

class ImageManager:
    def __init__(self, img_filename, xml_filename=None):
        self._img_filename = img_filename
        self._xml_filename = xml_filename or img_filename.replace(os.path.splitext(img_filename)[1], ".xml")
        self._img = Image.open(img_filename)
        self._rects = []
        self._load_rects()
        self._resized_ratio_w = 1
        self._resized_ratio_h = 1

        self._current_rects = None

    def _load_rects(self):
        if os.path.exists(self._xml_filename):
            rects_xml = read_xml(self._xml_filename)
            if rects_xml:
                self._rects = rects_xml

    def get_img(self):
        return self._img

    def get_rects(self):
        return self._rects

    def resizing_img(self, max_height=700, max_width=700):
        resized_img = self._img.copy()
        if resized_img.height > max_height:
            ratio = max_height / resized_img.height
            resized_img = resized_img.resize(
                (int(resized_img.width * ratio), int(resized_img.height * ratio))
            )
        if resized_img.width > max_width:
            ratio = max_width / resized_img.width
            resized_img = resized_img.resize(
                (int(resized_img.width * ratio), int(resized_img.height * ratio))
            )

        self._resized_ratio_w = self._img.width / resized_img.width
        self._resized_ratio_h = self._img.height / resized_img.height
        return resized_img

    def _resize_rect(self, rect):
        resized_rect = {}
        resized_rect["left"] = rect["left"] / self._resized_ratio_w
        resized_rect["width"] = rect["width"] / self._resized_ratio_w
        resized_rect["top"] = rect["top"] / self._resized_ratio_h
        resized_rect["height"] = rect["height"] / self._resized_ratio_h
        if "label" in rect:
            resized_rect["label"] = rect["label"]
        return resized_rect

    def get_resized_rects(self):
        return [self._resize_rect(rect) for rect in self._rects]

    def _chop_box_img(self, rect):
        rect["left"] = int(rect["left"] * self._resized_ratio_w)
        rect["width"] = int(rect["width"] * self._resized_ratio_w)
        rect["top"] = int(rect["top"] * self._resized_ratio_h)
        rect["height"] = int(rect["height"] * self._resized_ratio_h)
        left, top, width, height = (
            rect["left"],
            rect["top"],
            rect["width"],
            rect["height"],
        )

        raw_image = np.asarray(self._img).astype("uint8")
        prev_img = np.zeros(raw_image.shape, dtype="uint8")
        prev_img[top : top + height, left : left + width] = raw_image[
            top : top + height, left : left + width
        ]
        prev_img = prev_img[top : top + height, left : left + width]
        label = ""
        if "label" in rect:
            label = rect["label"]
        return (Image.fromarray(prev_img), label)

    def init_annotation(self, rects):
        self._current_rects = rects
        return [self._chop_box_img(rect) for rect in self._current_rects]

    def set_annotation(self, index, label):
        self._current_rects[index]["label"] = label

    def save_annotation(self):
        output_xml(self._xml_filename, self._img, self._current_rects)


class ImageDirManager:
    def __init__(self, img_dir_name, xml_dir_name=None):
        self._img_dir_name = img_dir_name
        self._xml_dir_name = xml_dir_name or img_dir_name
        self._img_files = []
        self._xml_files = []

    def get_all_files(self, allow_types=["png", "jpg", "jpeg"]):
        allow_types += [i.upper() for i in allow_types]
        mask = ".*\.[" + "|".join(allow_types) + "]"
        self._img_files = [file for file in os.listdir(self._img_dir_name) if re.match(mask, file)]
        self._xml_files = [file.split(".")[0] + ".xml" for file in self._img_files]
        return self._img_files

    def get_to_relabel_files(self, txt_file, allow_types=["png", "jpg", "jpeg"]):
        with open(txt_file, 'r') as file:
            txt_files = [line.strip() for line in file]

        allow_types += [i.upper() for i in allow_types]
        mask = ".*\.[" + "|".join(allow_types) + "]"

        self._img_files = [file for file in os.listdir(self._img_dir_name) if re.match(mask, file)]
        self._xml_files = [file.split(".")[0] + ".xml" for file in self._img_files]

        # Extract the stem from txt_files
        txt_stems = [os.path.splitext(filename)[0] for filename in txt_files]

        # Filter list
        self._img_files = [filename for filename in self._img_files if os.path.splitext(filename)[0] in txt_stems]
        self._xml_files = [filename for filename in self._xml_files if os.path.splitext(filename)[0] in txt_stems]

        return self._img_files

    def get_exist_annotation_files(self, txt_file=None):
        if txt_file:
            with open(txt_file, 'r') as file:
                self._annotations_files = [line.strip() for line in file]
        else:
            self._annotations_files = [file for file in os.listdir(self._xml_dir_name) if re.match(r".*\.xml", file)]
        
        return self._annotations_files


    def set_all_files(self, files):
        self._img_files = files
        self._xml_files = [file.split(".")[0] + ".xml" for file in self._img_files]

    def set_annotation_files(self, files):
        self._annotations_files = files

    def get_image(self, index):
        img_file = os.path.join(self._img_dir_name, self._img_files[index])
        xml_file = os.path.join(self._xml_dir_name, self._xml_files[index])
        return ImageManager(img_file, xml_file)

    def _get_next_image_helper(self, index):
        while index < len(self._img_files) - 1:
            index += 1
            if self._xml_files[index] not in self._annotations_files:
                return index
        return None

    def get_next_annotation_image(self, index):
        image_index = self._get_next_image_helper(index)
        if image_index:
            return image_index
        if not image_index and len(self._img_files) != len(self._annotations_files):
            return self._get_next_image_helper(0)
