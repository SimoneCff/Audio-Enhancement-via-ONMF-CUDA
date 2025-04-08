#include "headers/audio_enhancement.hpp"
#include "headers/dictionary_learner.hpp"
#include "headers/audio.hpp"
#include "headers/fft.hpp"
#include "headers/performance.hpp"

#include <Eigen/Dense>
#include <sndfile.h>
#include <fstream>
#include <iostream>
#include <filesystem>


using namespace Eigen;
using namespace std;

namespace fs = filesystem;

tuple<MatrixXcd, MatrixXcd, int> Denoising(const vector<string> &training_data, const string &test_data, const string &method, const vector<int> &n_components, int patch_length, Performance_Writer &performance_writer) {
    if (training_data.size() != n_components.size()) {
        throw invalid_argument("The number of training audio files is not equal to the number of dictionaries!");
    }
    vector<MatrixXd> dicts;
    if (method == "ONMF") {
        // For ONMF, we override n_components as [50, 10] per the Python code
        auto t = chrono::steady_clock::now();
        vector<int> ncomp = n_components;
        for (size_t i = 0; i < training_data.size(); i++) {
            ONMF_Dictionary_Learner reconstructor(training_data[i],
                                                  ncomp[i],
                                                  100, // iterations
                                                  5,   // sub_iterations
                                                  100, // num_patches
                                                  50,  // batch_size
                                                  patch_length,
                                                  false,
                                                  true);
            cout << "A.spectrogram.shape: (" << reconstructor.spectrogram.rows() << ", " << reconstructor.spectrogram.cols() << ")" << endl;
            reconstructor.train_dict();
            
            MatrixXd dictionary = reconstructor.W;
            cout << "dictionary.shape = (" << dictionary.rows() << ", " << dictionary.cols() << ")" << endl;
            dicts.push_back(dictionary);
            auto t1 = chrono::steady_clock::now();
            double elapsed = chrono::duration_cast<chrono::duration<double>>(t1 - t).count();
            performance_writer.set_tempo_creazione_dict(elapsed);
        }
    }
    else {
        throw invalid_argument("Invalid Method!");
    }
    auto t = chrono::steady_clock::now();
    Audio_Separation separator(dicts, test_data, patch_length, 1);
    vector<MatrixXd> specs = separator.separate_audio(patch_length);
    auto t1 = chrono::steady_clock::now();
   

    MatrixXd noise_spec = specs[1];
    MatrixXd voice_spec = specs[0];
    
    MatrixXcd voice_stft = topic_to_stft(separator.stft,noise_spec + voice_spec, voice_spec);
    MatrixXcd noise_stft = topic_to_stft(separator.stft, noise_spec + voice_spec, noise_spec);

    double elapsed = chrono::duration_cast<chrono::duration<double>>(t1 - t).count();
    performance_writer.set_tempo_estrazione(elapsed);

    return make_tuple(voice_stft, noise_stft, separator.rate);
}

bool createDirectory(const string& path) {
    if (fs::create_directory(path)) {
        cout << "Directory created successfully: " << path << endl;
        return true;
    } else {
        if (fs::exists(path)) {
            cout << "Directory already exists: " << path << endl;
            return true;
        } else {
            cerr << "Error creating directory: " << path << endl;
            return false;
        }
    }
}

inline void reconstructAndSave(const MatrixXcd& stft_in, int rate, const string& file,const string& outPath) {
    MatrixXd dummySpec; // Non serve a nulla in questo contesto
    VectorXd audioOut;
    istft(stft_in, rate, dummySpec, audioOut);
    createDirectory("Output");

    fs::path test_path(file);
    string test_no_ext = test_path.stem().string();

    createDirectory("Output/"+test_no_ext);
    string full_path = "Output/" + test_no_ext +"/" +outPath;
    writewav(full_path, audioOut, rate);
}

int main(int argc, char *argv[]) {
    auto t = chrono::steady_clock::now();
    int patch_length = 1;

    Performance_Writer performance_writer("performance.csv");
    // Training data for ONMF denoising
    vector<string> training = {"Data/audio/whitman.wav", "Data/audio/WhiteNoise.wav"};
    vector<int> n_components = {60, 15};

    // Check if the test file is provided as a command-line argument
    string test;
    if (argc > 1) {
        test = argv[1]; // Use the first command-line argument
    } else {
        cerr << "Error: No test filename provided as a command-line argument." << endl;
        cerr << "Usage: " << argv[0] << " <test_filename.wav>" << endl;
        return 1; // Indicate an error
    }
    
    // ONMF-based Denoising
    cout << "Denoising using ONMF..." << endl;

    MatrixXcd voice_stft, noise_stft;
    int rate;
    tie(voice_stft, noise_stft, rate) = Denoising(training, test, "ONMF", n_components, patch_length, performance_writer);

    // Ricostruzioni rapide con la nuova utility
    reconstructAndSave(voice_stft, rate, test,"voice_onmf_(50,10)_a=100.wav");
    reconstructAndSave(noise_stft, rate, test,"noise_onmf_(50,10)_a=100.wav");

    auto t1 = chrono::steady_clock::now();
    double elapsed = chrono::duration_cast<chrono::duration<double>>(t1 - t).count();
    performance_writer.set_tempo_totale(elapsed);
    cout << "ONMF-based denoising is done in " << elapsed << " seconds" << endl;

    performance_writer.append_record(test);

    return 0;
}
