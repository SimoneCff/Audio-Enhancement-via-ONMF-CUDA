#ifndef FFT_HPP
#define FFT_HPP

#include <Eigen/Dense>
#include <fftw3.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Dimensione frame e hop (NPERSG/2)
#define NPERSG 2048
#define HOP (NPERSG/2)

using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::VectorXd;
using std::complex;
using std::vector;

// Genera una finestra di Hann di dimensione N
inline VectorXd hann_window(int N) {
    VectorXd w(N);
    for (int n = 0; n < N; n++) {
        w(n) = 0.5 * (1.0 - std::cos((2.0 * M_PI * n) / (N - 1)));
    }
    return w;
}

// Esegue un’eventuale troncamento e zero-padding in coda
inline VectorXd pad_audio(const VectorXd &audio, int trunc_rate) {
    // Se trunc_rate > 0, tronca l'audio
    VectorXd truncated = (trunc_rate > 0) 
                          ? audio.head(audio.size() / trunc_rate)
                          : audio;

    // Calcolo del padding, in modo che il segnale possa ospitare l’ultimo frame
    // evitandone il taglio
    int total_len = truncated.size();
    int remainder = (total_len - NPERSG) % HOP;
    int padding = 0;
    if (remainder != 0) {
        padding = HOP - remainder;
    }

    VectorXd padded(total_len + padding);
    padded.head(total_len) = truncated;
    padded.tail(padding).setZero();
    return padded;
}

/*
 * cstft: Calcola la STFT di un segnale in input.
 *  Input:
 *   - audio: segnale monofonico
 *   - rate: frequenza di campionamento
 *   - freq, time: assi delle frequenze e tempo (output)
 *   - spectrogram: matrice di magnitudo
 *   - stft_complex: matrice complessa STFT
 *   - trunc_rate: fattore di troncamento (0 se disabilitato)
 */
inline void cstft(const VectorXd &audio,
                  int rate,
                  VectorXd &freq,
                  VectorXd &time,
                  MatrixXd &spectrogram,
                  MatrixXcd &stft_complex,
                  int trunc_rate)
{
    // Prepara segnale tronco + zero-padded
    VectorXd padded_audio = pad_audio(audio, trunc_rate);
    int n_fft = NPERSG;
    int n_freq = n_fft / 2 + 1;

    // Calcolo numero di frame
    int n_frames = (padded_audio.size() - NPERSG) / HOP + 1;
    freq = VectorXd::LinSpaced(n_freq, 0, rate / 2);
    time = VectorXd::Zero(n_frames);
    spectrogram = MatrixXd::Zero(n_freq, n_frames);
    stft_complex = MatrixXcd::Zero(n_freq, n_frames);

    VectorXd window = hann_window(NPERSG);
    vector<double> in(NPERSG);
    vector<fftw_complex> out(n_freq);

    // Creazione del piano FFTW
    fftw_plan plan = fftw_plan_dft_r2c_1d(NPERSG, in.data(), out.data(), FFTW_ESTIMATE);

    // STFT a finestre sovrapposte
    for (int i = 0; i < n_frames; i++) {
        int start = i * HOP;
        // Prendi il frame corrente
        VectorXd frame_in = padded_audio.segment(start, NPERSG);
        // Applica finestra
        frame_in = frame_in.cwiseProduct(window);

        // Copia il frame in 'in'
        for (int j = 0; j < NPERSG; j++) {
            in[j] = frame_in[j];
        }

        // Esegui la FFT
        fftw_execute(plan);

        // Calcola stft_complex + magnitudo
        for (int k = 0; k < n_freq; k++) {
            double re = out[k][0];
            double im = out[k][1];
            stft_complex(k, i) = complex<double>(re, im);
            spectrogram(k, i) = std::sqrt(re * re + im * im);
        }
        time(i) = double(start) / rate;
    }

    fftw_destroy_plan(plan);
}

/*
 * istft: Calcola la trasformata inversa di uno spettrogramma complesso (STFT).
 *  Input:
 *   - stft: matrice complessa STFT di dimensioni (n_freq, n_frames)
 *   - rate: frequenza di campionamento (solo per reference)
 *   - reconstructed: matrice di magnitudo ricostruita (opzionale)
 *   - output_audio: segnale di uscita nel dominio del tempo
 */
inline void istft(const MatrixXcd &stft,
                  int rate,
                  MatrixXd &reconstructed,
                  VectorXd &output_audio)
{
    // Calcola dimensioni
    int n_freq = stft.rows();
    int n_frames = stft.cols();
    int n_fft = (n_freq - 1) * 2; // inverso di n_freq = n_fft/2 + 1
    int signal_length = (n_frames - 1) * (n_fft / 2) + n_fft;

    // Inizializza risultati
    reconstructed = MatrixXd::Zero(n_freq, n_frames);
    VectorXd full_output_audio = VectorXd::Zero(signal_length);
    VectorXd window_sum = VectorXd::Zero(signal_length);
    VectorXd window = hann_window(n_fft);

    // Prepara i buffer per FFTW
    vector<fftw_complex> in(n_freq);
    vector<double> out(n_fft);
    fftw_plan plan = fftw_plan_dft_c2r_1d(n_fft, in.data(), out.data(), FFTW_ESTIMATE);

    // Inverti ogni frame e fai overlap-add
    for (int i = 0; i < n_frames; i++) {
        // Copia i valori reali / immaginari in "in"
        for (int k = 0; k < n_freq; k++) {
            in[k][0] = stft(k, i).real();
            in[k][1] = stft(k, i).imag();
        }

        // Esegui la iFFT
        fftw_execute(plan);

        // Overlap-add
        int start = i * (n_fft / 2);
        for (int j = 0; j < n_fft; j++) {
            double val = (out[j] / double(n_fft)) * window[j];
            full_output_audio[start + j] += val;
            window_sum[start + j] += window[j];
        }

        // Ricostruisci la magnitudo
        for (int k = 0; k < n_freq; k++) {
            double re = in[k][0];
            double im = in[k][1];
            reconstructed(k, i) = std::sqrt(re * re + im * im);
        }
    }
    fftw_destroy_plan(plan);

    // Dividi per la somma delle finestre
    for (int i = 0; i < signal_length; i++) {
        if (window_sum[i] > 1e-12) {
            full_output_audio[i] /= window_sum[i];
        }
    }

    output_audio = full_output_audio;
}

#endif