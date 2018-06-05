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
"""Modified Object detection library demo by chadwallacehart
Original reference:

 - Takes an input image and tries to detect person or cat.
 - Beeps when a cat is seen and saves the image.
"""
import argparse

from picamera import PiCamera
from time import time, strftime


from aiy.vision.leds import Leds
from aiy.vision.leds import PrivacyLed
from aiy.toneplayer import TonePlayer

from aiy.vision.inference import CameraInference
import aiy_cat_detection

# Sound setup
MODEL_LOAD_SOUND = ('C6w', 'c6w', 'C6w')
BEEP_SOUND = ('E6q', 'C6q')
player = TonePlayer(gpio=22, bpm=30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_frames',
        '-f',
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

    args = parser.parse_args()

    with PiCamera() as camera, PrivacyLed(Leds()):
        # See the Raspicam documentation for mode and framerate limits:
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        # Set to the highest resolution possible at 16:9 aspect ratio
        camera.sensor_mode = 5
        camera.resolution = (1640, 922)
        camera.start_preview(fullscreen=True)

        with CameraInference(aiy_cat_detection.model()) as inference:
            print("Camera inference started")
            player.play(*MODEL_LOAD_SOUND)

            last_time = time()
            pics = 0
            save_pic = False

            for f, result in enumerate(inference.run()):

                for i, obj in enumerate(aiy_cat_detection.get_objects(result, 0.3)):

                    print('%s Object #%d: %s' % (strftime("%Y-%m-%d-%H:%M:%S"), i, str(obj)))
                    x, y, width, height = obj.bounding_box
                    if obj.label == 'CAT':
                        save_pic = True
                        player.play(*BEEP_SOUND)

                # save the image if there was 1 or more cats detected
                if save_pic:
                    # save the clean image
                    camera.capture("images/image_%s.jpg" % strftime("%Y%m%d-%H%M%S"))
                    pics +=1
                    save_pic = False

                if f == args.num_frames or pics == args.num_pics:
                    break

                now = time()
                duration = (now - last_time)

                # The Movidius chip runs at 35 ms per image.
                # Then there is some additional overhead for the object detector to
                # interpret the result and to save the image. If total process time is
                # running slower than 50 ms it could be a sign the CPU is geting overrun
                if duration > 0.50:
                    print("Total process time: %s seconds. Bonnet inference time: %s ms " %
                          (duration, result.duration_ms))

                last_time = now

        camera.stop_preview()

if __name__ == '__main__':
    main()
