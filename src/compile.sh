#!/bin/sh

gcc -DTRAINING=1 -Wall -W -O3 -O0 -g -Wunused-variable -I../include denoise.c kiss_fft.c pitch.c celt_lpc.c rnn.c rnn_data.c -o denoise_training -lm
