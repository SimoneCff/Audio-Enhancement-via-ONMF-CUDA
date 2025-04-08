#ifndef DICTIONARYLEARNER_HPP
#define DICTIONARYLEARNER_HPP

//File .Hpp locali
#include "onmf.hpp"
#include "audio_cuda.hpp" 

//Librerie Eigen per la manipolazione di matrici
#include <Eigen/Dense>

//Librerie C++
#include <random>
#include <iomanip>
#include <iostream>
#include <chrono>

using namespace Eigen;
using namespace std;

class ONMF_Dictionary_Learner {
public:
    //Path del file audio
    string path;
    //Parametri per l'ONMF
    int n_components;
    int iterations;
    int sub_iterations;
    int num_patches;
    int batch_size;
    int patch_length;
    //Informazioni audio
    int num_frames;
    int num_channels;
    int rate;
    //Parametri per l'estrazione delle features
    bool is_matrix;
    bool is_color;
    // Dizionario e codice ONMF
    MatrixXd W; 
    MatrixXd code;
    // Vettore audio
    VectorXd audio;
    // Matrici STFT e Spettrogramma
    MatrixXcd stft;
    MatrixXd spectrogram;

    // Costruttore
    ONMF_Dictionary_Learner(const string &path,
                            int n_components = 100,
                            int iterations = 200,
                            int sub_iterations = 20,
                            int num_patches = 1000,
                            int batch_size = 20,
                            int patch_length = 7,
                            bool is_matrix = false,
                            bool is_color = true);

    // Estrae patch casuali dallo spettrogramma
    MatrixXd extract_random_patches();

    // Addestra il dizionario
    void train_dict();
};

#endif // DICTIONARYLEARNER_HPP