#ifndef AUDIO_ENHANCEMENT_HPP
#define AUDIO_ENHANCEMENT_HPP

#include "onmf.hpp"
#include "fft.hpp"
#include "audio.hpp"
#include "performance.hpp"

#include <iostream>
#include <string>

#include <Eigen/Core>

#include <sndfile.h> // Libsoundfile per la gestione audio

#include <vector>
#include <algorithm>
#include <cmath>

#include <ctime>
#include <chrono>

inline MatrixXd concatenate(const vector<MatrixXd> &matrices, int axis = 1) {
    if (matrices.empty()) {
        throw runtime_error("No matrices provided.");
    }
    const size_t rows = matrices[0].rows();
    const size_t cols = matrices[0].cols();
    for (const auto &mat : matrices) {
        if (axis == 1 && mat.rows() != rows) {
            throw runtime_error("All matrices must have the same number of rows for horizontal concatenation.");
        }
        if (axis == 0 && mat.cols() != cols) {
            throw runtime_error("All matrices must have the same number of columns for vertical concatenation.");
        }
    }
    // Calcolo dimensione risultato
    size_t result_rows = (axis == 0) ? 0 : rows;
    size_t result_cols = (axis == 1) ? 0 : cols;
    for (const auto &mat : matrices) {
        if (axis == 0) {
            result_rows += mat.rows();
        } else {
            result_cols += mat.cols();
        }
    }

    // Esegui concatenazione
    MatrixXd result(result_rows, result_cols);
    size_t current_row = 0, current_col = 0;
    for (const auto &mat : matrices) {
        if (axis == 0) {
            result.block(current_row, 0, mat.rows(), mat.cols()) = mat;
            current_row += mat.rows();
        } else {
            result.block(0, current_col, mat.rows(), mat.cols()) = mat;
            current_col += mat.cols();
        }
    }
    return result;
}


// Classe per separazione audio
class Audio_Separation {
public:
    vector<MatrixXd> input_dictionaries;
    VectorXd audio;
    VectorXd freq;
    VectorXd time;
    MatrixXcd stft;
    MatrixXd spectrogram;
    MatrixXd W;
    string audio_file;
    int rate;
    int patch_length;
    int channels;

    // Costruttore
    Audio_Separation(vector<MatrixXd> &input_dictionaries, const string &audio_file, int patch_length, int trunc_rate = 1) {
        this->input_dictionaries = input_dictionaries;
        this->audio_file = audio_file;
        tie(this->rate, this->audio) = readwav(audio_file);
        // STFT con eventuale troncamento (trunc_rate)
        cstft(this->audio, this->rate, this->freq, this->time, this->spectrogram, this->stft, trunc_rate);
        // Concatenazione lungo le colonne
        this->W = concatenate(input_dictionaries, 1);
        this->patch_length = patch_length;
    }

    vector<MatrixXd> separate_audio(int recons_resolution, double alpha = 0) {
        cout << "reconstructing given network..." << endl;

        // Map the spectrogram
        MatrixXd A = this->spectrogram;
        Eigen::Map<Eigen::MatrixXd> A_matrix(A.data(), A.size() / A.cols(), A.cols());
        int m = A_matrix.rows();
        int n = A_matrix.cols();

        // Initialize lists
        vector<MatrixXd> separated_specs, zeroed_dicts, separated_dicts;
        separated_specs.reserve(this->input_dictionaries.size());
        zeroed_dicts.reserve(this->input_dictionaries.size());
        separated_dicts.reserve(this->input_dictionaries.size());
        for (const auto &dict : this->input_dictionaries) {
            separated_specs.emplace_back(MatrixXd::Zero(m, n));
            zeroed_dicts.emplace_back(MatrixXd::Zero(dict.rows(), dict.cols()));
        }
        // Create "separated" dictionaries
        for (size_t i = 0; i < this->input_dictionaries.size(); ++i) {
            MatrixXd temp = zeroed_dicts[i];
            zeroed_dicts[i] = this->input_dictionaries[i];
            separated_dicts.push_back(concatenate(zeroed_dicts, 1));
            zeroed_dicts[i] = temp;
        }

        MatrixXd A_overlap_count = MatrixXd::Zero(m, n);
        int k = this->patch_length;
        auto t0 = chrono::steady_clock::now();

        // Calculate the number of steps
        int num_cols = (n - k) / recons_resolution;
        for (int i = 0; i <= n-k; i += recons_resolution) {  
            // Extract patch
            MatrixXd patch = A.block(0, i, m, k);
            // Flatten col
            MatrixXd patchVec = patch.reshaped(m * k, 1);

            // Calculate encoding
            MatrixXd code = update_code_within_radius(patchVec, this->W, MatrixXd(), 0.0, alpha, 100, 0.01);

            // Separate reconstructions
            vector<MatrixXd> patch_recons_list;
            patch_recons_list.reserve(separated_dicts.size());
            for (auto &D : separated_dicts) {
                MatrixXd recon = D * code;
                // Return M x k shape
                MatrixXd reconMat = recon.reshaped(m, k);
                patch_recons_list.push_back(std::move(reconMat));
            }
            // Overlap-add
            for (int r_idx = 0; r_idx < m; r_idx++) {
                for (int col_offset = 0; col_offset < k; col_offset++) {
                    double c_val = A_overlap_count(r_idx, i + col_offset);
                    for (size_t spec_idx = 0; spec_idx < separated_specs.size(); ++spec_idx) {
                        double oldVal = separated_specs[spec_idx](r_idx, i + col_offset);
                        double newVal = patch_recons_list[spec_idx](r_idx, col_offset);
                        separated_specs[spec_idx](r_idx, i + col_offset) = (c_val * oldVal + newVal) / (c_val + 1.0);
                    }
                    A_overlap_count(r_idx, i + col_offset) += 1.0;
                }
            }
            // Print progress every 10 patches
            int currentPatch = i / recons_resolution;
            if (currentPatch % 10 == 0) {
                cout << "reconstructing " << currentPatch << "th patch out of " << num_cols << endl;
            }
        }
        auto t1 = chrono::steady_clock::now();
        double elapsed = chrono::duration_cast<chrono::duration<double>>(t1 - t0).count();
        cout << "Reconstructed in " << elapsed << " seconds" << endl;
        return separated_specs;
    }
};

// topic_to_stft function with soft masking
inline MatrixXcd topic_to_stft(const MatrixXcd &stft, const MatrixXd &NMF_Sxx, const MatrixXd &topic) {
    MatrixXcd output = stft; 
    for (int i = 0; i < output.rows(); i++) {
        for (int j = 0; j < output.cols(); j++) {
            if (NMF_Sxx(i, j) == 0) {
                output(i, j) = 0;
            } else {
                output(i, j) *= topic(i, j) / NMF_Sxx(i, j);
            }
        }
    }
    return output;
}

#endif // !AUDIO_ENHANCEMENT_HPP