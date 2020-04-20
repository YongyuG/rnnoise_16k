====
RNNoise training for 16K audio
====

Notification
============
This project is refered to Dr.Jean-Marc Valin efforts from [RNNoise: Learning Noise Suppression](https://people.xiph.org/~jm/demo/rnnoise/)

References:
Jean-Marc Valinm, [A Hybrid DSP/Deep Learning Approach toReal-Time Full-Band Speech Enhancement](https://jmvalin.ca/papers/rnnoise_mmsp2018.pdf)
[RNNoise Original Project](https://github.com/xiph/rnnoise)


How to use
==========
This project is done one year ago when I started doing NS things, so codes are not well organized. If you have any questions, feel free to ask.

Following the CMakeLists.txt for compiling the projcet
The src/denoice.c is the main thing on modification from 48k -> 16k, and training/run.sh is how to train in 16k audio

you also need to check src/compile.sh for compiling

