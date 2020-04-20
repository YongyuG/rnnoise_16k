#!/bin/sh

gcc -DTRAINING=1 -Wall -W -O3 -O0 -g -Wunused-variable -I../include denoise_gao.c kiss_fft.c pitch.c celt_lpc.c rnn.c rnn_data.c -o denoise_training_gao -lm
