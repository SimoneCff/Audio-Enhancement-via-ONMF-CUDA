#ifndef AUDIO_ENHANCEMENT_HPP
#define AUDIO_ENHANCEMENT_HPP

//File .Hpp locali
#include "onmf.hpp"
#include "audio_cuda.hpp"

//Librerie C++
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>

//Librerie Eigen per la manipolazione di matrici
#include <Eigen/Core>
//Librerie Audio
#include <sndfile.h>

//Funzione per la concatenazione dei dizionari
MatrixXd concatenate(const vector<MatrixXd> &matrices, int axis = 1);

// Classe per separazione audio
class Audio_Separation {
public:
    //Dizionari per l'estrazione delle features
    vector<MatrixXd> input_dictionaries;
    //Vettore Audio
    VectorXd audio;
    //Matrici STFT e Spettrogramma
    MatrixXcd stft;
    MatrixXd spectrogram;
    //Matrice W (Input diciotnaries concatenati)
    MatrixXd W;
    //File audio
    string audio_file;
    //Informazioni audio
    int rate;
    int channels;
    int num_frames;
    int num_channels;
    //Parametri per la separazione
    int patch_length;

    // Costruttore
    Audio_Separation(vector<MatrixXd> &input_dictionaries, const string &audio_file, int patch_length, int trunc_rate = 1);
    //Funzione per la separazione audio
    vector<MatrixXd> separate_audio(int recons_resolution, double alpha = 0);
    
};

//Funzione per applicare le maschere sulla stft
MatrixXcd topic_to_stft(const MatrixXcd &stft, const MatrixXd &NMF_Sxx, const MatrixXd &topic);

#endif // !AUDIO_ENHANCEMENT_HPP