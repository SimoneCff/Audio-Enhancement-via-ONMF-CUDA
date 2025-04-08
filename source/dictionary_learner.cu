#include "../headers/dictionary_learner.hpp"

// Costruttore della classe ONMF_Dictionary_Learner
ONMF_Dictionary_Learner::ONMF_Dictionary_Learner(const string &path,
    int n_components,
    int iterations,
    int sub_iterations,
    int num_patches,
    int batch_size,
    int patch_length,
    bool is_matrix,
    bool is_color)
: path(path), n_components(n_components), iterations(iterations),
sub_iterations(sub_iterations), num_patches(num_patches),
batch_size(batch_size), patch_length(patch_length),
is_matrix(is_matrix), is_color(is_color)
{
// Inizializza il dizionario W e la matrice dei codici
this->W = MatrixXd::Zero(patch_length, n_components);
this->code = MatrixXd::Zero(n_components, iterations * batch_size);

// Leggi il file audio e calcola la STFT
tie(this->rate, this->audio) = read_audio_file(path, this->num_frames, this->num_channels);
tie(this->stft, this->spectrogram) = cu_sfft(this->audio, this->num_frames, this->num_channels, 2048);
}

// Estrae patch casuali dallo spettrogramma
MatrixXd ONMF_Dictionary_Learner::extract_random_patches() {
    // Calcola la dimensione dello spettrogramma
    int spectrogram_cols = this->spectrogram.cols();
    int freq_size = this->spectrogram.rows();

    // Crea una matrice X per contenere i patch
    MatrixXd X(freq_size * patch_length, num_patches);

    //Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    //Distribuzione uniforme per estrarre patch casuali
    std::uniform_int_distribution<> dis(0, spectrogram_cols - patch_length);
    // Estrae num_patches patch casuali
    for (int i = 0; i < num_patches; ++i) {
        // Estrae un indice casuale
        int a = dis(gen); 
        // Estrae il patch
        MatrixXd Y = spectrogram.block(0, a, freq_size, patch_length);
        //Reshape il patch in un vettore
        Y.resize(freq_size * patch_length, 1);
        //Salva il patch nella matrice X
        X.col(i) = Y;
    }

    return X;
}

void ONMF_Dictionary_Learner::train_dict() {
    cout << "Training dictionaries from patches..." << endl;
    
    // Copia W in una variabile locale per evitare copie inutili
    MatrixXd W_local = std::move(this->W);
    // Matrici temporanee per At e Bt
    MatrixXd At, Bt;
    // Variabile per la history
    int history_val = 0;

    auto start_time = chrono::steady_clock::now(); // Misura il tempo di esecuzione

    // Itera per il numero di iterazioni
    for (int t = 0; t < this->iterations; t++) {
        // Estrae patch casuali dallo spettrogramma
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
        history_val = nmf.history;

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