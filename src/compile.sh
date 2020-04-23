#!/bin/sh

gcc -DTRAINING=1 -Wall -W -O3 -O0 -g -Wunused-variable -I../include denoise16.c kiss_fft.c pitch.c celt_lpc.c rnn.c rnn_data.c -o /media/yongyug/DATA/new_train/work/denoise_training -lm
