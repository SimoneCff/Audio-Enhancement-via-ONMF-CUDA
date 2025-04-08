#ifndef AUDIO_CUDA_HPP
#define AUDIO_CUDA_HPP

//Librerie C++
#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <iostream>

//Librerie C++
#include <sndfile.h>

//Librerie CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

using namespace std;
using namespace Eigen;

//Read audio file
tuple<int, VectorXd> read_audio_file(const string& file_path, int& num_frames, int& num_channels);

//Write audio file
void write_audio_file(const string& file_path, const VectorXd audio_data, int num_frames, int num_channels, int sample_rate);

//CUDA SFFT
tuple<MatrixXcd,MatrixXd> cu_sfft(VectorXd audio_data, int num_frames, int num_channels, int npersg);

//CUDA ISFFT
VectorXd cu_isfft(MatrixXcd& stft, int num_frames, int num_channels, int npersg);

#endif // !AUDIO_CUDA_HPP