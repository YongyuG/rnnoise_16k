
RNNoise training for 16K audio
==============================


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

you also need to check __src/compile.sh__ for compiling src directory,

Pay attention, I use __src/denoise.c__ for feature extractions. __src/denoise16.c__ is something that I did for experiments.

if you wanna use less frames or more frames for training, modify the __main__ function variable __count__ inside the __src/denoise.c__

The whole process is:
* cd src
* bash compile.sh *which will generate binary for creating mix features and labels, use  **denoise.c** inside compile.sh*
* ./src/denoise_training /data/speech_dir /data/noise_dir mixed.wav > training_16k_v3.f32 *the mixed.wav is the raw file which you can check whether wavs have been mixed*
* python bin2hdf5.py training_16k_v3.f32 80000000 75 training_16k_v3.h5
* python rnn_train_16k.py
* python dump_rnn.py weights.hdf5 rnn_data.c rnn_data.rnnn name

## Replace with new trained model
if you follow the instructions and __training/run.sh__, new __rnn_data.c__ and __rnn_data.h__ which are come from your new trained model will be generated.
Replace the old __rnn_data.c__ and  __rnn_data.h__ in __src__ directory with the new one, using __CMakeList.txt__ in the working directory,
* cmake .
* make

the binary file will be generated in __bin__ directory, you can also change the name of your binary inside __CMakeList.txt__

### The way to use binary file
`
Binary File <Input Noisy File> <Output Path>
`

__e.g:__ 

`
./bin/rnn_gao_new noisy.wav out.wav
`
