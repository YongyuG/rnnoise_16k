#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define DR_MP3_IMPLEMENTATION
#define DR_WAV_IMPLEMENTATION
#define FRAME_SIZE 160
#include <stdio.h>
#include "rnnoise.h"
#include "dr_mp3.h"
#include "dr_wav.h"
#define NN 160

#ifndef nullptr
#define nullptr 0
#endif
#ifndef MIN
#define MIN(A, B)        ((A) < (B) ? (A) : (B))
#endif

#define LOW_COEF 1

float *wavRead_f32(const char *filename, uint32_t *sampleRate, uint64_t *sampleCount, uint32_t *channels) {
    drwav_uint64 totalSampleCount = 0;
    float *input = drwav_open_file_and_read_pcm_frames_f32(filename, channels, sampleRate, &totalSampleCount);
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

void wavWrite_f32_to_int16(char *filename, const float *buffer, int sampleRate, uint32_t totalSampleCount, uint32_t channels) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = channels;
    format.sampleRate = (drwav_uint32) sampleRate;
    format.bitsPerSample = 16;
    int16_t *newbuffer = malloc(sizeof(int16_t)*(totalSampleCount - FRAME_SIZE));

    for (int32_t i = FRAME_SIZE; i < totalSampleCount; ++i) {
        //buffer[i] = drwav_clamp(buffer[i], -32768, 32767) * (1.0f / 32768.0f);
        newbuffer[i-FRAME_SIZE] = (int16_t)drwav_clamp(buffer[i], -32768, 32767);
    }
    drwav *pWav = drwav_open_file_write(filename, &format);
    if (pWav) {
        drwav_uint64 samplesWritten = drwav_write(pWav, totalSampleCount-FRAME_SIZE, newbuffer);
        drwav_uninit(pWav);
        if (samplesWritten != totalSampleCount-FRAME_SIZE) {
            fprintf(stderr, "write file [%s] error.\n", filename);
            exit(1);
        }
    }
}


void rnnDeNoise(char *in_file, char *out_file) {

    uint32_t sampleRate = 0;
    uint64_t sampleCount = 0;
    uint32_t channels = 0;
    float *buffer = wavRead_f32(in_file, &sampleRate, &sampleCount, &channels);
    float *input = buffer;
    DenoiseState *st;
    st = rnnoise_create();

    size_t frames = sampleCount / NN;

    /*For rnn_gao, 5月15号把这里移植到rnn_gao */
    for (int i = 0; i < sampleCount; i++) {
        //printf("buffer ==== %f\n", fabs(buffer[i]));
        if (buffer[i]> 32768*LOW_COEF){
            for (int j = 0; j < sampleCount; j++) {
                buffer[j] = (buffer[j] * LOW_COEF);

            }
            break;
        }
    }

    if (buffer != NULL) {
        for (int i = 0; i < frames; i++) {
            rnnoise_process_frame(st,input,input);
            input += NN;
        }
        //denoise_proc(buffer, sampleCount, sampleRate, channels);
        wavWrite_f32_to_int16(out_file, buffer, sampleRate, (uint32_t)sampleCount, channels);
        free(buffer);
    }
}


int main(int argc, char **argv) {
    if (argc < 2){
        printf("Usage:./rnn_gao_new [inputWav] [RNNnoise_output]\n");
        return -1;
    }
    char *in_file = argv[1];
    if (argc == 3) {
        printf("Start doing noise supreesion\n");
        char *out_file = argv[2];

        rnnDeNoise(in_file, out_file);
        printf("Finished RNNnoise Noise Supression \n");

        return 0;
    } else {
        printf("Usage:./rnn_gao [inputWav] [outputWav]\n");
    }
}
