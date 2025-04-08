#ifndef DICTIONARYLEARNER_HPP
#define DICTIONARYLEARNER_HPP

#include "onmf.hpp"
#include "fft.hpp"
#include "audio.hpp"

#include <Eigen/Dense>
#include <random>
#include <iomanip>
#include <iostream>
#include <chrono>

using namespace Eigen;
using namespace std;

class ONMF_Dictionary_Learner {
public:
    string path;
    int n_components;
    int iterations;
    int sub_iterations;
    int num_patches;
    int batch_size;
    int patch_length;
    bool is_matrix;
    bool is_color;
    MatrixXd W; // Dizionario
    MatrixXd code;
    int rate;
    VectorXd audio; // Segnale audio
    VectorXd freq;
    VectorXd time;
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
                            bool is_color = true)
        : path(path), n_components(n_components), iterations(iterations),
          sub_iterations(sub_iterations), num_patches(num_patches),
          batch_size(batch_size), patch_length(patch_length),
          is_matrix(is_matrix), is_color(is_color)
    {
        // Inizializza il dizionario W e la matrice dei codici
        this->W = MatrixXd::Zero(patch_length, n_components);
        this->code = MatrixXd::Zero(n_components, iterations * batch_size);

        // Leggi il file audio e calcola la STFT
        tie(this->rate, this->audio) = readwav(this->path);
        cstft(this->audio, this->rate, this->freq, this->time, this->spectrogram, this->stft, 0);
    }

    // Estrae patch casuali dallo spettrogramma
    MatrixXd extract_random_patches() {
        int spectrogram_cols = this->spectrogram.cols();
        int freq_size = this->spectrogram.rows();

        MatrixXd X(freq_size * patch_length, num_patches);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, spectrogram_cols - patch_length);

        for (int i = 0; i < num_patches; ++i) {
            int a = dis(gen);  // start time of the patch
            MatrixXd Y = spectrogram.block(0, a, freq_size, patch_length);
            // Reshape to a column vector
            Y.resize(freq_size * patch_length, 1);
            X.col(i) = Y;
        }
        return X;
    }

    // Addestra il dizionario
    void train_dict() {
        cout << "Training dictionaries from patches..." << endl;
        // Evita copie non necessarie
        MatrixXd W_local = std::move(this->W);
        MatrixXd At, Bt;
        int history_val = 0;
    
        auto start_time = chrono::steady_clock::now(); // Misura il tempo di esecuzione

        for (int t = 0; t < this->iterations; t++) {
            MatrixXd X = this->extract_random_patches();

            // Crea oggetto Online_NMF sullo stack
            Online_NMF nmf(
                X,
                this->n_components,
                this->sub_iterations,
                this->batch_size,
                (t == 0 ? MatrixXd() : W_local),
                (t == 0 ? MatrixXd() : At),
                (t == 0 ? MatrixXd() : Bt),
                MatrixXd(),
                history_val,
                -1 // alpha di default
            );

            // Addestra il dizionario
            MatrixXd ignored;
            tie(W_local, At, Bt, ignored) = nmf.train_dict();
            history_val = nmf.history; // Oppure nmf.history se lo rendi public

            // Stampa progresso (ogni 10 iterazioni o ultima)
            if ((t + 1) % 10 == 0 || t == this->iterations - 1) {
                int progress = (t + 1) * 100 / this->iterations;
                cout << "\rProgress: [" << progress << "%] " << string(progress / 2, '=') << flush;
            }
        }
        auto end_time = chrono::steady_clock::now();
        double elapsed_time = chrono::duration_cast<chrono::duration<double>>(end_time - start_time).count();
        cout << "\nTraining completed in " << elapsed_time << " seconds.\n";

        // Aggiorna W evitando copie
        this->W = std::move(W_local);
        cout << "Final dictionary shape: (" << this->W.rows() << ", " << this->W.cols() << ")" << endl;
    }
};

#endif // DICTIONARYLEARNER_HPP