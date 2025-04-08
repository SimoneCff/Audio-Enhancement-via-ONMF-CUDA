#include "../headers/audio_cuda.hpp"
#include <Eigen/Dense>
#include <sndfile.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <tuple>

using namespace std;
using namespace Eigen;

// Legge il file audio, converte in mono se necessario e restituisce (sample_rate, audio_data)
tuple<int, VectorXd> read_audio_file(const string& file_path, int& num_frames, int& num_channels) {
    // Apre il file audio con libsndfile
    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open(file_path.c_str(), SFM_READ, &sfinfo);

    // Controlla se il file è stato aperto correttamente
    if (!sndfile) {
        cerr << "Errore nell'apertura del file audio: " << file_path << " : " << sf_strerror(sndfile) << endl;
        throw runtime_error("Impossibile aprire il file audio.");
    }

    //Estrae le informazioni dal file audio
    num_channels = sfinfo.channels;
    int sample_rate = sfinfo.samplerate;
    num_frames = sfinfo.frames;

    // Legge tutti i campioni in un vettore
    VectorXd audio_data(num_frames * num_channels);
    // Legge i campioni dal file audio
    sf_read_double(sndfile, audio_data.data(), num_frames * num_channels);

    // Se audio è stereo, converte in mono (prendendo il canale sinistro)
    if (sfinfo.channels == 2) {
        cout << "[DEBUG] Convert stereo to mono (taking left channel)" << endl;
        // Crea un vettore mono
        VectorXd mono(sfinfo.frames);
        // Prende il canale sinistro 
        for (int i = 0; i < sfinfo.frames; ++i) {
            mono(i) = audio_data(i * 2);
        }
        // Sovrascrive il vettore audio_data con il vettore mono
        audio_data = mono;
        // Aggiorna il numero di canali
        num_channels = 1;
    }
    // Chiude il file audio
    sf_close(sndfile);
    // Stampa le informazioni del file audio
    cout << "File audio caricato con successo." << endl;
    cout << "Frequenza di campionamento: " << sample_rate << " Hz" << endl;
    cout << "Numero di canali: " << num_channels << endl;
    cout << "Numero di frame: " << num_frames << endl;

    return make_tuple(sample_rate, audio_data);
}

// Scrive il file audio usando libsndfile
void write_audio_file(const string& file_path, const VectorXd audio_data, int num_frames, int num_channels, int sample_rate) {
    SF_INFO sfinfo;
    // Imposta le informazioni del file audio
    sfinfo.samplerate = sample_rate;
    sfinfo.channels = num_channels;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    // Crea il file audio
    SNDFILE* sndfile = sf_open(file_path.c_str(), SFM_WRITE, &sfinfo);
    // Controlla se il file è stato creato correttamente
    if (!sndfile) {
        cerr << "Errore nell'apertura del file audio: " << sf_strerror(sndfile) << endl;
        throw runtime_error("Impossibile aprire il file audio.");
    }
    // Scrive i campioni nel file audio
    sf_write_double(sndfile, audio_data.data(), num_frames * num_channels);
    // Chiude il file audio
    sf_close(sndfile);
}


// Kernel per calcolare lo spettrogramma.
__global__ void computeSpectrogramKernel(const double* d_stft ,  // STFT (complesso)
                                           double* d_spectrogram, // spettrogramma (reale)
                                           int freqBins,   // numero di frequenze = npersg/2 + 1
                                           int timeBins)   // numero di frame STFT
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // indice globale
    int total = freqBins * timeBins; // numero totale di elementi
    // Calcola lo spettrogramma per ogni elemento
    if(idx < total) {
        // Calcola posizione: per il frame c e la frequenza r
        int r = idx / timeBins;
        int c = idx % timeBins;
        // Ogni complesso occupa 2 double (re, im)
        int baseIdx = (r * timeBins + c) * 2;
        // Calcola la magnetudine
        double realPart = d_stft[baseIdx];
        double imagPart = d_stft[baseIdx + 1];
        d_spectrogram[idx] = realPart * realPart + imagPart * imagPart;
    }
}


//Funzione per calcolare la STFT di un segnale audio tramite CUFFT.
tuple<MatrixXcd, MatrixXd> cu_sfft(VectorXd audio_data, int num_frames, int num_channels, int npersg) {
    // Calcola il numero di frame e di frequenze
    int hop = npersg / 2; 
    int n_frames = (num_frames - npersg) / hop + 1;
    int n_freq = npersg / 2 + 1;

    // Segmenta il segnale (ogni colonna è un frame di lunghezza npersg)
    MatrixXd frames(npersg, n_frames);
    for (int i = 0; i < n_frames; i++) {
        frames.col(i) = audio_data.segment(i * hop, npersg);
    }

    // Alloca la memoria sul device per i frame (reali)
    double* d_audio_frames = nullptr;
    cudaMalloc(&d_audio_frames, n_frames * npersg * sizeof(double));
    cudaMemcpy(d_audio_frames, frames.data(), n_frames * npersg * sizeof(double), cudaMemcpyHostToDevice);

    // Alloca memoria per l'output FFT: CUFFT converte D2Z producendo n_freq elementi per trasformata.
    cufftDoubleComplex* d_stft = nullptr;
    cudaMalloc(&d_stft, n_frames * n_freq * sizeof(cufftDoubleComplex));

    //Dichiarazione del piano CUFFT
    cufftHandle plan;

    // Crea un piano per batch (n_frames) di lunghezza npersg, trasformazione D2Z.
    cufftPlan1d(&plan, npersg, CUFFT_D2Z, n_frames);
    // Esegue la trasformata
    cufftExecD2Z(plan, d_audio_frames, d_stft);
    cudaDeviceSynchronize();

    // Copia il risultato STFT su host in una MatrixXcd
    MatrixXcd stft(n_freq, n_frames);
    cudaMemcpy(stft.data(), d_stft, n_frames * n_freq * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

    // Calcola lo spettrogramma: mag^2 della STFT
    MatrixXd spectrogram(n_freq, n_frames);
    double* d_spectrogram = nullptr;
    cudaMalloc(&d_spectrogram, n_frames * n_freq * sizeof(double));

    // Calcola il numero totale di elementi
    int total = n_freq * n_frames;
    // Imposta la dimensione del blocco e della griglia per il kernel 
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    // Richiama il kernel per calcolare lo spettrogramma, usando la STFT sotto forma di array di double
    computeSpectrogramKernel<<<gridSize, blockSize>>>(reinterpret_cast<double*>(d_stft),d_spectrogram, n_freq, n_frames);
    cudaDeviceSynchronize();

    // Copia lo spettrogramma su host
    cudaMemcpy(spectrogram.data(), d_spectrogram, n_frames * n_freq * sizeof(double), cudaMemcpyDeviceToHost);

    // Libera le risorse
    cudaFree(d_audio_frames);
    cudaFree(d_stft);
    cudaFree(d_spectrogram);
    cufftDestroy(plan);

    return make_tuple(stft, spectrogram);
}

//Funzione per ricostruire il segnale audio da una STFT tramite CUFFT.
VectorXd cu_isfft(MatrixXcd& stft, int original_num_frames, int num_channels, int npersg) {
    // Calcola il numero di frame e di frequenze
    int n_frames = stft.cols();
    int n_freq = npersg / 2 + 1;
    int hop = npersg / 2;

    // Alloca memoria device per la STFT
    cufftDoubleComplex* d_stft = nullptr;
    cudaMalloc(&d_stft, n_frames * n_freq * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_stft, stft.data(), n_frames * n_freq * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    // Alloca memoria sul device per il segnale tempo ricostruito (per ogni frame, reale)
    double* d_audio_frames = nullptr;
    cudaMalloc(&d_audio_frames, n_frames * npersg * sizeof(double));

    // Dichiarazione del piano CUFFT
    cufftHandle plan;
    // Crea un piano per batch (n_frames) per trasformata inversa Z2D.
    cufftPlan1d(&plan, npersg, CUFFT_Z2D, n_frames);
    // Esegue la trasformata inversa
    cufftExecZ2D(plan, d_stft, d_audio_frames);
    cudaDeviceSynchronize();

    // Copia le frame ricostruite su host
    MatrixXd frames(npersg, n_frames);
    cudaMemcpy(frames.data(), d_audio_frames, n_frames * npersg * sizeof(double), cudaMemcpyDeviceToHost);

    // Ricostruisce il segnale audio tramite overlap-add
    VectorXd audio_out = VectorXd::Zero(original_num_frames);
    VectorXd weight = VectorXd::Zero(original_num_frames); // per normalizzare la sovrapposizione
    for (int i = 0; i < n_frames; i++) {
        for (int j = 0; j < npersg; j++) {
            int pos = i * hop + j;
            if (pos < original_num_frames) {
                // Nota: CUFFT_Z2D non normalizza l'output, quindi si scala per 1/npersg
                audio_out(pos) += frames(j, i) / npersg;
                weight(pos) += 1.0;
            }
        }
    }
    // Normalizza l'output
    for (int i = 0; i < original_num_frames; i++) {
        if(weight(i) > 0) audio_out(i) /= weight(i);
    }

    // Libera le risorse
    cudaFree(d_stft);
    cudaFree(d_audio_frames);
    cufftDestroy(plan);

    return audio_out;
}