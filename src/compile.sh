#!/bin/sh

gcc -DTRAINING=1 -Wall -W -O3 -O0 -g -Wunused-variable -Wreturn-type -I../include denoise.c kiss_fft.c pitch.c celt_lpc.c rnn.c rnn_data.c -o /media/yongyug/DATA/new_train/work/denoise_training -lm
