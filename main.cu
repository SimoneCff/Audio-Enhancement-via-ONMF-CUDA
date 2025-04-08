#include "headers/audio_enhancement.hpp"
#include "headers/dictionary_learner.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>


using namespace Eigen;
using namespace std;

namespace fs = filesystem;

tuple<MatrixXcd, MatrixXcd, int, int ,int> Denoising(const vector<string> &training_data, const string &test_data, const string &method, const vector<int> &n_components, int patch_length) {
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

    MatrixXcd voice_stft = topic_to_stft(separator.stft, noise_spec + voice_spec, voice_spec);
    MatrixXcd noise_stft = topic_to_stft(separator.stft, noise_spec + voice_spec, noise_spec);

    double elapsed = chrono::duration_cast<chrono::duration<double>>(t1 - t).count();

    return make_tuple(voice_stft, noise_stft, separator.rate, separator.num_channels, separator.num_frames);
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

int main(int argc, char *argv[]) {
    auto t = chrono::steady_clock::now();
    int patch_length = 1;


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
    int rate, num_channels, num_frames;
    tie(voice_stft, noise_stft, rate, num_channels, num_frames) = Denoising(training, test, "ONMF", n_components, patch_length);
    VectorXd voice = cu_isfft(voice_stft, num_frames, num_channels, 2048);
    VectorXd noise = cu_isfft(noise_stft, num_frames, num_channels, 2048);

    write_audio_file("out_voice.wav",voice,num_frames,num_channels,rate);
    write_audio_file("out_noise.wav",noise,num_frames,num_channels,rate);

    auto t1 = chrono::steady_clock::now();
    double elapsed = chrono::duration_cast<chrono::duration<double>>(t1 - t).count();
    cout << "ONMF-based denoising is done in " << elapsed << " seconds" << endl;

    return 0;
}
