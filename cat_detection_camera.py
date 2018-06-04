#!/usr/bin/env python3
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Object detection library demo.

 - Takes an input image and tries to detect person, dog, or cat.
 - Draws bounding boxes around detected objects.
 - Saves an image with bounding boxes around detected objects.
"""
import argparse
import io
import sys
from PIL import Image, ImageDraw, ImageFont

from picamera import PiCamera
from time import time, strftime, sleep


from aiy.vision.leds import Leds
from aiy.vision.leds import PrivacyLed
from aiy.vision.inference import CameraInference
# from aiy.vision.annotator import Annotator
from aiy.toneplayer import TonePlayer

import aiy_cat_detection

# Sound setup
MODEL_LOAD_SOUND = ('C6w', 'c6w', 'C6w')
BEEP_SOUND = ('E6q', 'C6q')
player = TonePlayer(gpio=22, bpm=30)


def _crop_center(image):
    width, height = image.size
    size = min(width, height)
    x, y = (width - size) / 2, (height - size) / 2
    return image.crop((x, y, x + size, y + size)), (x, y)

def main():
    """Object detection camera inference example."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_frames',
        '-n',
        type=int,
        dest='num_frames',
        default=-1,
        help='Sets the number of frames to run for, otherwise runs forever.')

    parser.add_argument(
        '--num_pics',
        '-p',
        type=int,
        dest='num_pics',
        default=-1,
        help='Sets the max number of pictures to take, otherwise runs forever.')

    '''
    parser.add_argument(
        '--num_objects',
        '-c',
        type=int,
        dest='num_objects',
        default=3,
        help='Sets the number of object inferences to print.')
    '''
    args = parser.parse_args()

    def print_classes(classes, object_count):
        s = ''
        for index, (obj, prob) in enumerate(classes):
            if index > object_count - 1:
                break
            s += '%s=%1.2f\t|\t' % (obj, prob)
        print('%s\r' % s)

    leds = Leds()

    with PiCamera() as camera, PrivacyLed(leds):
        camera.sensor_mode = 5
        camera.resolution = (1640, 922)
        camera.framerate = 30
        camera.start_preview(fullscreen=True)

        pics = 0

        # ToDo: see if the annotator work with object detection
        # conclusion - it doesn't: conflicts with saving the image

        # Annotator renders in software so use a smaller size and scale results
        # for increased performance.
#        annotator = Annotator(camera, dimensions=(320, 180))
        scale_x = 320 / 1640
        scale_y = 180 / 922

        # Incoming boxes are of the form (x, y, width, height). Scale and
        # transform to the form (x1, y1, x2, y2).
        def transform(bounding_box):
            x, y, width, height = bounding_box
            return (scale_x * x, scale_y * y, scale_x * (x + width),
                    scale_y * (y + height))

        with CameraInference(aiy_cat_detection.model()) as inference:
            print("Camera inference started")
            player.play(*MODEL_LOAD_SOUND)

            last_time = time()
            save_pic = False

            for result in inference.run():
#                annotator.clear()
                for i, obj in enumerate(aiy_cat_detection.get_objects(result, 0.3)):
                    print('Object #%d: %s' % (i, str(obj)))
                    x, y, width, height = obj.bounding_box
                    if obj.kind == 1:
                        save_pic = True
                        player.play(*BEEP_SOUND)

#                    annotator.bounding_box(transform(obj.bounding_box), fill=0)

#                annotator.update()rm
                now = time()
                duration = (now - last_time)
                if duration > 0.50:
                    print("Total process time: %s seconds. Bonnet inference time: %s ms " % (duration, result.duration_ms))

                last_time = now

                if save_pic:
                    # save the clean image
                    filename = "images/image_%s.jpg" % strftime("%Y%m%d-%H%M%S")
                    camera.capture(filename)
                    pics +=1
                    save_pic = False

                    # save the annotated image
                    # image = Image.open(filename)
                    # draw = ImageDraw.Draw(image)

                if pics == args.num_pics:
                    break

        camera.stop_preview()

if __name__ == '__main__':
    main()
