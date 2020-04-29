/* Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "rnnoise.h"
#include "pitch.h"
#include "arch.h"
#include "rnn.h"
#include "rnn_data.h"
#include <dirent.h>
#include <time.h>
#if TRAINING
#define DR_MP3_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION
#include "dr_mp3.h"
#include "dr_wav.h"
#endif
#define BLOCK_SIZE 8000
#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (40<<FRAME_SIZE_SHIFT)
//#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)
#define MAX_FILE 999999
#define MAX_FILE_NAME 1000
#define SAMPLE_RATE 16000

//#define PITCH_MIN_PERIOD 60
//#define PITCH_MAX_PERIOD 768
//#define PITCH_FRAME_SIZE 960

/*for 16K speech files*/
#define PITCH_MIN_PERIOD 20
#define PITCH_MAX_PERIOD 256
#define PITCH_FRAME_SIZE 320  

#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define SQUARE(x) ((x)*(x))

#define SMOOTH_BANDS 1

#if SMOOTH_BANDS
//#define NB_BANDS 22
#define NB_BANDS 18
#else
//#define NB_BANDS 21
#define NB_BANDS 17
#endif

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)


#ifndef TRAINING
#define TRAINING 0
#endif

static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};


typedef struct {
  int init;
  kiss_fft_state *kfft;
  float half_window[FRAME_SIZE];
  float dct_table[NB_BANDS*NB_BANDS];
} CommonState;

struct DenoiseState {
  float analysis_mem[FRAME_SIZE];
  float cepstral_mem[CEPS_MEM][NB_BANDS];
  int memid;
  float synthesis_mem[FRAME_SIZE];
  float pitch_buf[PITCH_BUF_SIZE];
  float pitch_enh_buf[PITCH_BUF_SIZE];
  float last_gain;
  int last_period;
  float mem_hp_x[2];
  float lastg[NB_BANDS];
  RNNState rnn;
};

#if SMOOTH_BANDS
void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    //printf(eband5ms[i]);
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
      tmp += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r;
      tmp += X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i;
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
    }
  }
}
#else
void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    opus_val32 sum = 0;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++) {
      sum += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
      sum += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
    }
    bandE[i] = sum;
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++)
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = bandE[i];
  }
}
#endif


CommonState common;

static void check_init() {
  int i;
  if (common.init) return;
  common.kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE, NULL, NULL, NULL, 0);
  for (i=0;i<FRAME_SIZE;i++)
    common.half_window[i] = sin(.5*M_PI*sin(.5*M_PI*(i+.5)/FRAME_SIZE) * sin(.5*M_PI*(i+.5)/FRAME_SIZE));
  for (i=0;i<NB_BANDS;i++) {
    int j;
    for (j=0;j<NB_BANDS;j++) {
      common.dct_table[i*NB_BANDS + j] = cos((i+.5)*j*M_PI/NB_BANDS);
      if (j==0) common.dct_table[i*NB_BANDS + j] *= sqrt(.5);
    }
  }
  common.init = 1;
}

static void dct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[j*NB_BANDS + i];
    }
    out[i] = sum*sqrt(2./22);
  }
}

#if 0
static void idct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[i*NB_BANDS + j];
    }
    out[i] = sum*sqrt(2./22);
  }
}
#endif

static void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  opus_fft(common.kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

static void inverse_transform(float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  opus_fft(common.kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}

static void apply_window(float *x) {
  int i;
  check_init();
  for (i=0;i<FRAME_SIZE;i++) {
    x[i] *= common.half_window[i];
    x[WINDOW_SIZE - 1 - i] *= common.half_window[i];
  }
}

int rnnoise_get_size() {
  return sizeof(DenoiseState);
}

int rnnoise_init(DenoiseState *st) {
  memset(st, 0, sizeof(*st));
  return 0;
}

DenoiseState *rnnoise_create() {
  DenoiseState *st;
  st = malloc(rnnoise_get_size());
  rnnoise_init(st);
  return st;
}

void rnnoise_destroy(DenoiseState *st) {
  free(st);
}

#if TRAINING
int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;
#endif

static void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
  int i;
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(x);
  forward_transform(X, x);
#if TRAINING
  for (i=lowpass;i<FREQ_SIZE;i++)
    X[i].r = X[i].i = 0;
#endif
  compute_band_energy(Ex, X);
}

static int compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in) {
  int i;
  float E = 0;
  float *ceps_0, *ceps_1, *ceps_2;
  float spec_variability = 0;
  float Ly[NB_BANDS];
  float p[WINDOW_SIZE];
  float pitch_buf[PITCH_BUF_SIZE>>1];
  int pitch_index;
  float gain;
  float *(pre[1]);
  float tmp[NB_BANDS];
  float follow, logMax;
  frame_analysis(st, X, Ex, in);
  RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE], in, FRAME_SIZE);
  pre[0] = &st->pitch_buf[0];
  pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
  pitch_search(pitch_buf+(PITCH_MAX_PERIOD>>1), pitch_buf, PITCH_FRAME_SIZE,
               PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD, &pitch_index);
  pitch_index = PITCH_MAX_PERIOD-pitch_index;

  gain = remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
          PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
  st->last_period = pitch_index;
  st->last_gain = gain;
  for (i=0;i<WINDOW_SIZE;i++)
    p[i] = st->pitch_buf[PITCH_BUF_SIZE-WINDOW_SIZE-pitch_index+i];
  apply_window(p);
  forward_transform(P, p);
  compute_band_energy(Ep, P);
  compute_band_corr(Exp, X, P);
  for (i=0;i<NB_BANDS;i++) Exp[i] = Exp[i] * (1.0f / sqrtf(.001f + Ex[i] * Ep[i]));
  dct(tmp, Exp);
  for (i=0;i<NB_DELTA_CEPS;i++) features[NB_BANDS+2*NB_DELTA_CEPS+i] = tmp[i];
  features[NB_BANDS+2*NB_DELTA_CEPS] -= 1.3f;
  features[NB_BANDS+2*NB_DELTA_CEPS+1] -= 0.9f;
  features[NB_BANDS+3*NB_DELTA_CEPS] = .01f*(pitch_index-300);
  logMax = -2;
  follow = -2;
  for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2f+Ex[i]);
    Ly[i] = MAX16(logMax-7, MAX16(follow-1.5f, Ly[i]));
    logMax = MAX16(logMax, Ly[i]);
    follow = MAX16(follow-1.5f, Ly[i]);
    E += Ex[i];
  }
  if (!TRAINING && E < 0.04f) {
    /* If there's no audio, avoid messing up the state. */
    RNN_CLEAR(features, NB_FEATURES);
    return 1;
  }
  dct(features, Ly);
  features[0] -= 12;
  features[1] -= 4;
  ceps_0 = st->cepstral_mem[st->memid];
  ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM+st->memid-1] : st->cepstral_mem[st->memid-1];
  ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM+st->memid-2] : st->cepstral_mem[st->memid-2];
  for (i=0;i<NB_BANDS;i++) ceps_0[i] = features[i];
  st->memid++;
  for (i=0;i<NB_DELTA_CEPS;i++) {
    features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
    features[NB_BANDS+i] = ceps_0[i] - ceps_2[i];
    features[NB_BANDS+NB_DELTA_CEPS+i] =  ceps_0[i] - 2*ceps_1[i] + ceps_2[i];
  }
  /* Spectral variability features. */
  if (st->memid == CEPS_MEM) st->memid = 0;
  for (i=0;i<CEPS_MEM;i++)
  {
    int j;
    float mindist = 1e15f;
    for (j=0;j<CEPS_MEM;j++)
    {
      int k;
      float dist=0;
      for (k=0;k<NB_BANDS;k++)
      {
        float tmp;
        tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
        dist += tmp*tmp;
      }
      if (j!=i)
        mindist = MIN32(mindist, dist);
    }
    spec_variability += mindist;
  }
  features[NB_BANDS+3*NB_DELTA_CEPS+1] = spec_variability/CEPS_MEM-2.1f;
  return TRAINING && E < 0.1f;
}

static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y) {
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(x, y);
  apply_window(x);
  for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}

static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g) {
  int i;
  float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  for (i=0;i<NB_BANDS;i++) {
#if 0
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
    r[i] = MIN16(1, MAX16(0, r[i]));
#else
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
    r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
#endif
    r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
  }
  interp_band_gain(rf, r);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r += rf[i]*P[i].r;
    X[i].i += rf[i]*P[i].i;
  }
  float newE[NB_BANDS];
  compute_band_energy(newE, X);
  float norm[NB_BANDS];
  float normf[FREQ_SIZE]={0};
  for (i=0;i<NB_BANDS;i++) {
    norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
  }
  interp_band_gain(normf, norm);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= normf[i];
    X[i].i *= normf[i];
  }
}

float rnnoise_process_frame(DenoiseState *st, float *out, const float *in) {
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[WINDOW_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS];
  float gf[FREQ_SIZE]={1};
  float vad_prob = 0;
  int silence;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
  silence = compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);

  if (!silence) {
    compute_rnn(&st->rnn, g, &vad_prob, features);
    pitch_filter(X, P, Ex, Ep, Exp, g); //由于会造成截顶，所以不用pitch_filter
    for (i=0;i<NB_BANDS;i++) {
      float alpha = .6f;
      g[i] = MAX16(g[i], alpha*st->lastg[i]);
      st->lastg[i] = g[i];
      if(g[i]>1.0) {
          fprintf(stderr, "check gain ================ %f \n", g[i]);
      }


    }
    interp_band_gain(gf, g);
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
        if (gf[i] > 1.0){
            fprintf(stderr, "check gain gf================ %f \n", gf[i]);
        }

        X[i].r *= gf[i];
      X[i].i *= gf[i];
    }
#endif
  }

  frame_synthesis(st, out, X);
  return vad_prob;
}

/*Gao new implmentation edition in SNR*/
#if TRAINING

float calculate_wav_energy(float *wav_buffer, uint64_t sampleCount){
    float *input = wav_buffer;
    size_t blocks = sampleCount/BLOCK_SIZE;
    size_t frames = sampleCount/FRAME_SIZE;
    int flag=0;
    float E_max = 0.00001;
    float E_speech = 0.0;
    if (sampleCount > BLOCK_SIZE) {
        while (flag < blocks) {
            printf("blocks ==== %d, samplecount = %d, flag == %d\n", blocks, sampleCount, flag);
            for (int j = 0; j < BLOCK_SIZE; j++) {
                E_speech += input[j] * input[j];

            }

            if (E_speech > E_max) E_max = E_speech;
            input += BLOCK_SIZE;
            E_speech = 0.0;
            flag++;
        }
    }else if  (sampleCount < BLOCK_SIZE && sampleCount > FRAME_SIZE){
        while (flag < frames) {
            for (int j = 0; j < FRAME_SIZE; j++) {
                E_speech += input[j] * input[j];
            }

            if (E_speech > E_max) E_max = E_speech;
            input += FRAME_SIZE;
            E_speech = 0.0;
            flag++;
        }
    }else{
        for (int i = 0; i < sampleCount; i++) {
            E_speech = input[i] * input[i];
            if (E_speech > E_max) E_max = E_speech;
        }
    }
    return E_max;

}



void get_file_list(DIR *dir, char* path, char **file_list, int *count){
    struct dirent *entry;
    char single_file[120];
    //char list_tmp[1000][1000];
    while ((entry = readdir(dir))!=NULL){

        if(strcmp(entry->d_name,".")==0 || strcmp(entry->d_name,"..")==0)
        {continue;
        }else{


            sprintf(single_file,"%s/%s",path,entry->d_name); //构成文件全路径v
            //printf("speech single file full path name === %s \n", single_file);
            strcpy(file_list[*count], single_file);
            (*count) ++;
        }
    }

}

void get_file_list2(DIR *dir, char* path, char **file_list, int *count){
    struct dirent *entry;
    char single_file[120];
    char *pFile;
    while ((entry = readdir(dir))!=NULL){
        //printf("this file  ===== %s\n", entry->d_name);
        pFile = strchr(entry->d_name, '.');
        if (pFile!=NULL) {
            //printf("file things ==== %s\n", pFile);
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0 || strcmp(pFile, ".wav") != 0) {
                continue;
            }
            //  printf("check ===== %s\n", entry->d_name);

            sprintf(single_file, "%s/%s", path, entry->d_name); //构成文件全路径v
            // printf("count %d, wav single file full path name === %s \n", *count, single_file);
            strcpy(file_list[*count], single_file);
            (*count) ++;

        }else{
            continue;
        }
    }
    printf("num of file in directory ====== %d\n",*count);

}


void get_file_list3(DIR *dir, char* path, char **file_list, int *count){
    struct dirent *entry;
    char single_file[120];
    char *pFile;
    while ((entry = readdir(dir))!=NULL){
        pFile = strchr(entry->d_name, '.');
        if (pFile!=NULL) {
            //printf("file things ==== %s\n", pFile);
            if (strcmp(pFile, ".wav")==0) {
                printf("check ===== %s\n", entry->d_name);

                sprintf(single_file, "%s/%s", path, entry->d_name); //构成文件全路径v
                printf("count %d, wav single file full path name === %s \n", *count, single_file);
                strcpy(file_list[*count], single_file);
                (*count)++;
            }else{
                continue;
            }
        }else{
            continue;
        }
    }
    printf("num of file in directory ====== %d\n",*count);
}

float *wavRead_f32(const char *filename, uint32_t *sampleRate, uint64_t *sampleCount, uint32_t *channels) {
    drwav_uint64 totalSampleCount = 0;
    fprintf(stderr, "read file name ==== %s\n", filename);

    float *input = drwav_open_file_and_read_pcm_frames_f32(filename, channels, sampleRate, &totalSampleCount);
    //fprintf(stderr, "input ==== %d\n", input);
    if (input == NULL) {
        drmp3_config pConfig;
        input = drmp3_open_file_and_read_f32(filename, &pConfig, &totalSampleCount);
        if (input != NULL) {
            *channels = pConfig.outputChannels;
            *sampleRate = pConfig.outputSampleRate;
        }
    }
    if (input == NULL) {
        fprintf(stderr, "read file [%s] error.\n", filename);
        exit(1);
    }
    *sampleCount = totalSampleCount * (*channels);
    for (int32_t i = 0; i < *sampleCount; ++i) {
        input[i] = input[i] * 32768.0f;
    }
    return input;


}

float  *open_file(char **dir_list, int file_list_len, uint32_t *sampleRate, uint64_t *sampleCount, uint32_t *channels, int flag)

{

    //srand((unsigned)time(NULL));
    int i = rand()%(file_list_len);
//    if (flag ==1) {
//        //printf(" This should be speech file, %d th file === File_name ===  %s \n",i, dir_list[i]);
//    }else{
//        //printf(" This should be noise file , %d th file === File_name ===  %s \n",i, dir_list[i]);
//    }
    fprintf(stderr, "open file name =================== %s\n", dir_list[i]);

    //char *filename = "/media/yongyug/DATA/new_train/noise_dir/noise-free-sound-0711.wav";
    float *buffer = wavRead_f32(dir_list[i], sampleRate, sampleCount, channels);

    //printf("Energy ============= %f\n", *Energy);
    return buffer;
}

#endif



//define TRAINING 1
#if TRAINING

static float uni_rand() {

  return rand()/(double)RAND_MAX-.5;
}

static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}



int main(int argc, char **argv) {
    int i;
    int count=0;
    static const float a_hp[2] = {-1.99599, 0.99600};
    static const float b_hp[2] = {-2, 1};
    float a_noise[2] = {0};
    float b_noise[2] = {0};
    float a_sig[2] = {0};
    float b_sig[2] = {0};
    float mem_hp_x[2]={0};
    float mem_hp_n[2]={0};
    float mem_resp_x[2]={0};
    float mem_resp_n[2]={0};
    float x[FRAME_SIZE];
    float n[FRAME_SIZE];
    float xn[FRAME_SIZE];
    int16_t noisy_data[FRAME_SIZE];

    int vad_cnt=0;
    int gain_change_count=0;
    //float speech_gain = 1, noise_gain = 1;
    int speech_flag = 1, noise_flag = 1;
    char *out_file = argv[3];
    DenoiseState *st;
    DenoiseState *noise_state;
    DenoiseState *noisy;
    st = rnnoise_create();
    noise_state = rnnoise_create();
    noisy = rnnoise_create();
    if (argc!=4) {
        fprintf(stderr, "usage: %s <speech> <noise>  <output mixed>\n", argv[0]);
        return 1;
    }

    char **speech_array = (char**)malloc(sizeof(char*) * MAX_FILE);
    char **noise_array = (char**)malloc(sizeof(char*) * MAX_FILE);

    int k;
    for (k = 0; k < MAX_FILE; k++) {
        speech_array[k] = (char*)malloc(sizeof(char) * MAX_FILE_NAME);
        noise_array[k] = (char*)malloc(sizeof(char) * MAX_FILE_NAME);
    }

    int speechcount = 0;
    int noisecount = 0;
    DIR *dirSpeech, *dirNoise = NULL;

    float speech_energy = 0;
    float noise_energy = 0;
    float expcected_SNR = 0.0;
    if((dirSpeech = opendir(argv[1])) == NULL || (dirNoise = opendir(argv[2])) == NULL)
    {
        //printf("open dir failed !");
        return -1;
    }
    else{
        get_file_list3(dirSpeech, argv[1], speech_array, &speechcount);
        //printf("check if get count %d \n",speechcount);

        get_file_list3(dirNoise, argv[2], noise_array, &noisecount);
        //printf("check if get count %d \n",noisecount);
    }
    closedir(dirSpeech);
    closedir(dirNoise);
    srand((unsigned)time(NULL));



    float *speech_buffer=NULL;
    float *noise_buffer=NULL;
    uint32_t speech_sampleRate = 0;
    uint64_t speech_sampleCount = 0;
    uint32_t speech_channels = 0;
    size_t speech_frames = 0;

    uint32_t noise_sampleRate = 0;
    uint64_t noise_sampleCount = 0;
    uint32_t noise_channels = 0;
    size_t noise_frames = 0;

    float *speech_input=NULL;
    float *noise_input=NULL;

    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = 1;
    format.sampleRate = (drwav_uint32) SAMPLE_RATE;
    format.bitsPerSample = 16;
    drwav *pWav = drwav_open_file_write(out_file, &format);
    while (1)
    {
        //int frames_count = 0;
        kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], N[FREQ_SIZE], P[WINDOW_SIZE];
        float Ex[NB_BANDS], Ey[NB_BANDS], En[NB_BANDS], Ep[NB_BANDS];
        float Exp[NB_BANDS];
        float Ln[NB_BANDS];
        float features[NB_FEATURES];
        float g[NB_BANDS];
        float gf[FREQ_SIZE]={1};
        short tmp[FRAME_SIZE];

        float vad=0;
        float vad_prob;
        float E=0;
        if (count==1000000) break;
        fprintf(stderr, "count ==== %d\n", count);


        //if (count==50000000) break;
        if (++gain_change_count > 2821) {

        if (rand()%10==0) noise_flag = 0; else noise_flag = 1;
        if (rand()%10==0) speech_flag = 0; else speech_flag = 1;
        expcected_SNR = rand()%20;

        gain_change_count = 0;

        rand_resp(a_noise, b_noise);
        rand_resp(a_sig, b_sig);
        lowpass = FREQ_SIZE * 3000./8000. * pow(50., rand()/(double)RAND_MAX);

        //fprintf(stderr,"low pass ==== %f  \n",  FREQ_SIZE * 3000./8000. * pow(50., rand()/(double)RAND_MAX));
        for (i=0;i<NB_BANDS;i++) {
            if (eband5ms[i]<<FRAME_SIZE_SHIFT > lowpass) {
                band_lp = i;
                break;
                }
            }
        }


        if (speech_frames==0) {
            DRWAV_FREE(speech_buffer);
            speech_buffer = open_file(speech_array, speechcount,
                                      &speech_sampleRate, &speech_sampleCount, &speech_channels, 1);
            speech_energy = calculate_wav_energy(speech_buffer, speech_sampleCount);

            speech_frames = speech_sampleCount / FRAME_SIZE;
            speech_input = speech_buffer;
        }


        if (speech_flag != 0) {

            for (i = 0; i < FRAME_SIZE; i++) x[i] = speech_input[i];
            for (i = 0; i < FRAME_SIZE; i++) E += speech_input[i] *  speech_input[i];
        } else {
            for (i = 0; i < FRAME_SIZE; i++) x[i] = 0;
            E = 0;
        }


        if (noise_frames==0) {
            DRWAV_FREE(noise_buffer);
            noise_buffer = open_file(noise_array, noisecount,
                                     &noise_sampleRate, &noise_sampleCount, &noise_channels, 1);
            noise_energy = calculate_wav_energy(noise_buffer, noise_sampleCount);

            noise_frames = noise_sampleCount / FRAME_SIZE;
            noise_input = noise_buffer;
        }

            if (noise_flag != 0) {
                double noise_factor = pow(10., (10*log10(speech_energy/noise_energy) - expcected_SNR) / 20);

                //fprintf(stderr, "speech_energy %f ===, noise_energy ======= %f, noise_factor ==== %f\n", speech_energy, noise_energy, noise_factor);
                for (i = 0; i < FRAME_SIZE; i++) {
                    n[i] = noise_factor * noise_input[i];
                   // fprintf(stderr, "speech_energy %f ===, noise_energy ======= %f, noise_factor ==== %f, noise_input === %f, n ====%f\n", speech_energy, noise_energy, noise_factor, noise_input[i], n[i]);


                }

            } else {
                for (i = 0; i < FRAME_SIZE; i++) n[i] = 0;
            }

        for (i=0;i<FRAME_SIZE;i++) {
            xn[i] = x[i] + n[i];
            //fprintf(stderr, "xn ==== %f noise_factor = %f\n", n[i],noise_factor);


        }
        biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
        biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
        biquad(n, mem_hp_n, n, b_hp, a_hp, FRAME_SIZE);
        biquad(n, mem_resp_n, n, b_noise, a_noise, FRAME_SIZE);
//
        for (i=0;i<FRAME_SIZE;i++) {
            xn[i] = x[i] + n[i];

            noisy_data[i] = (int16_t)drwav_clamp(xn[i], -32768, 32767);

        }

        drwav_write(pWav, FRAME_SIZE, noisy_data);


        if (E > 1e9f) {
            vad_cnt=0;
        } else if (E > 1e8f) {
            vad_cnt -= 5;
        } else if (E > 1e7f) {
            vad_cnt++;
        } else {
            vad_cnt+=2;
        }


        if (vad_cnt < 0) vad_cnt = 0;
        if (vad_cnt > 15) vad_cnt = 15;

        if (vad_cnt >= 10) vad = 0;
        else if (vad_cnt > 0) vad = 0.5f;
        else vad = 1.f;

        frame_analysis(st, Y, Ey, x);
        frame_analysis(noise_state, N, En, n);
        for (int j = 0; j < FRAME_SIZE; j++) {
          // fprintf(stderr, "XXXXXX ================ %f \n", noise_input[j]);


        }
        for (i=0;i<NB_BANDS;i++) Ln[i] = log10(1e-2+En[i]);

        int silence = compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, xn);
        pitch_filter(X, P, Ex, Ep, Exp, g);
        //printf("%f %d\n", noisy->last_gain, noisy->last_period);
        for (i=0;i<NB_BANDS;i++) {


        g[i] = sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));



            if (g[i] > 1) g[i] = 1;
            if (silence || i > band_lp) g[i] = -1;
            if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;
            if (vad==0 && noise_flag==0) g[i] = -1;



        // fprintf(stderr, "check gain ================ %f \n", g[i]);


        }


        speech_frames--;
        noise_frames--;
        speech_input+=FRAME_SIZE;
        noise_input+=FRAME_SIZE;
        count++;


        #if 0
            for (i=0;i<NB_FEATURES;i++) printf("%f ", features[i]);
        for (i=0;i<NB_BANDS;i++) printf("%f ", g[i]);
        for (i=0;i<NB_BANDS;i++) printf("%f ", Ln[i]);
        printf("%f\n", vad);
        #endif
        #if 1
        for (int l = 0; l < NB_FEATURES; l++) {
            fprintf(stderr, "matrix features ===== : %f \n ", features[l]);
        }



        fwrite(features, sizeof(float), NB_FEATURES, stdout);
            fwrite(g, sizeof(float), NB_BANDS, stdout);
            fwrite(Ln, sizeof(float), NB_BANDS, stdout);
            fwrite(&vad, sizeof(float), 1, stdout);
        #endif
        #if 0
                compute_rnn(&noisy->rnn, g, &vad_prob, features);
            interp_band_gain(gf, g);
        #if 1
            for (i=0;i<FREQ_SIZE;i++) {
              X[i].r *= gf[i];
              X[i].i *= gf[i];
            }
        #endif
            frame_synthesis(noisy, xn, X);

            for (i=0;i<FRAME_SIZE;i++) tmp[i] = n[i];
            fwrite(tmp, sizeof(short), FRFAME_SIZE, fout);
        #endif

    }
    drwav_uninit(pWav);
    fprintf(stderr, "matrix size: %d x %d\n", count, NB_FEATURES + 2*NB_BANDS + 1);


    return 0;
}

#endif
