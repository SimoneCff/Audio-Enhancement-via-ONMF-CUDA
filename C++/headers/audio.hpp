#ifndef audio_hpp 
#define audio_hpp

#include <iostream>
#include <sndfile.h>
#include <Eigen/Core>
#include <vector>
#include <tuple>
#include <string>
#include <stdexcept>

using namespace Eigen;

void nA(VectorXd& audio) {
    double max_val = audio.cwiseAbs().maxCoeff();
    if (max_val > 1e-9) {
        audio /= max_val;
    }
}

tuple<int, VectorXd> readwav(const string& audio_file) {
    SF_INFO sfinfo;
    SNDFILE* infile = sf_open(audio_file.c_str(), SFM_READ, &sfinfo);

    if (!infile) {
        throw runtime_error("Cannot open audio file: " + audio_file);
    }

    cout << "[DEBUG] Apertura file: " << audio_file 
         << "  | samplerate: " << sfinfo.samplerate 
         << "  | channels: " << sfinfo.channels 
         << "  | frames: " << sfinfo.frames << endl;

    VectorXd audio(sfinfo.frames * sfinfo.channels);
    sf_count_t read_count = sf_read_double(infile, audio.data(), sfinfo.frames * sfinfo.channels);

    if (read_count != sfinfo.frames * sfinfo.channels) {
        sf_close(infile);
        throw runtime_error("Error reading audio data from file: " + audio_file);
    }

    if (sfinfo.channels == 2) {
        // Converti da stereo a mono (prendi solo il canale sinistro)
        cout << "[DEBUG] Convert stereo to mono (taking left channel)" << endl;
        VectorXd mono(sfinfo.frames);
        for (int i = 0; i < sfinfo.frames; ++i) {
            mono(i) = audio(i * 2);
        }
        audio = mono;
    }

    // Normalizza l'audio
    nA(audio);
    cout << "[DEBUG] Audio length after normalization: " << audio.size() << endl;

    sf_close(infile);
    cout << "[DEBUG] Lettura completata: " << audio_file << endl;

    return make_tuple(sfinfo.samplerate, audio);
}

void writewav(const string& audio_file, const VectorXd& audio, int rate) {
    cout << "[DEBUG] Salvataggio file WAV: " << audio_file 
         << "  | Samplerate: " << rate 
         << "  | Lunghezza audio: " << audio.size() << endl;

    SF_INFO sfinfo;
    sfinfo.channels = 1;
    sfinfo.samplerate = rate;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* outfile = sf_open(audio_file.c_str(), SFM_WRITE, &sfinfo);
    if (!outfile) {
        throw runtime_error("[DEBUG] Error opening output file: " + audio_file);
    }

    sf_count_t write_count = sf_write_double(outfile, audio.data(), audio.size());
    if (write_count != audio.size()) {
        sf_close(outfile);
        throw runtime_error("[DEBUG] Error writing audio data: " + audio_file);
    }

    sf_close(outfile);
    cout << "[DEBUG] Scrittura completata: " << audio_file << endl;
}

#endif