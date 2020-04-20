
../src/denoise_training_gao /ssd/gaoyongyu/data/speech_dir /ssd/gaoyongyu/data/noise_dir mixed.wav > training_16k_v3.f32

python bin2hdf5.py training_16k_v3.f32 80000000 75 training_16k_v3.h5

python rnn_train_16k.py

python dump_rnn.py weights.hdf5 rnn_data.c rnn_data.rnnn name
