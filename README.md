
#RNNoise training for 16K audio


Notification
============
This project is refered to Dr.Jean-Marc Valin efforts from [RNNoise: Learning Noise Suppression](https://people.xiph.org/~jm/demo/rnnoise/)

References:
Paper: [A Hybrid DSP/Deep Learning Approach toReal-Time Full-Band Speech Enhancement](https://jmvalin.ca/papers/rnnoise_mmsp2018.pdf)
Original Github Repo: [RNNoise Original Project](https://github.com/xiph/rnnoise)


How to use
==========
This project is done one year ago when I started doing NS things, so codes are not well organized. If you have any questions, feel free to ask.

This can code is able to accepct a directory of __wav__ file for training rather than __raw__ file.

Following the CMakeLists.txt for compiling the projcet
The __src/denoice.c__ is the main thing on modification from __48k -> 16k__, and __training/run.sh__ is how to train in 16k audio. 

you also need to check __src/compile.sh__ for compiling src directory

