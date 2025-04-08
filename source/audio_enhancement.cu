#include "../headers/audio_enhancement.hpp"

//Kernel per applicare la maschera di un topic ad un STFT
__global__ void topicToSTFTKernel(const double* d_stft, const double* d_Sxx,
    const double* d_topic, double* d_out,
    int rows, int cols)
{
    // Calcola l'indice globale della matrice
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Se l'indice è valido, calcola l'indice lineare e applica la maschera
    if (row < rows && col < cols) {
        // Calcola l'indice lineare
        int idx = row * cols + col;
        // Applica la maschera 
        if (d_Sxx[idx] == 0.0) { // Evita divisione per zero
            d_out[idx] = 0.0;
        } else { // Calcola il valore
            d_out[idx] = d_stft[idx] * (d_topic[idx] / d_Sxx[idx]);
        }
    }
}

MatrixXd concatenate(const vector<MatrixXd> &matrices, int axis) {
    // Verifica che ci siano matrici da concatenare
    if (matrices.empty()) {
        throw runtime_error("Nessuna matrice da concatenare.");
    }
    // Verifica che le matrici abbiano le stesse dimensioni
    const size_t rows = matrices[0].rows();
    const size_t cols = matrices[0].cols();

    for (const auto &mat : matrices) {
        if (axis == 1 && mat.rows() != rows) {
            throw runtime_error("Tutte le matrici devono avere lo stesso numero di righe per la concatenazione orizzontale.");
        }
        if (axis == 0 && mat.cols() != cols) {
            throw runtime_error("Tutte le matrici devono avere lo stesso numero di colonne per la concatenazione verticale.");
        }
    }
    // Calcolo dimensione risultato
    size_t result_rows = (axis == 0) ? 0 : rows;
    size_t result_cols = (axis == 1) ? 0 : cols;
    // Calcola la dimensione risultante
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
        // Verifica se la concatenazione è orizzontale o verticale
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

// Funzione Costruttore per la classe Audio_Separation
Audio_Separation::Audio_Separation(vector<MatrixXd> &input_dictionaries, const string &audio_file, int patch_length, int trunc_rate) {
    this->input_dictionaries = input_dictionaries;
    this->audio_file = audio_file;
    // Leggi il file audio e calcola la STFT
    tie(this->rate, this->audio) = read_audio_file(audio_file, this->num_frames, this->num_channels);
    tie(this->stft, this->spectrogram) = cu_sfft(this->audio, this->num_frames, this->num_channels, 2048);
    // Concatenazione lungo le colonne
    this->W = concatenate(input_dictionaries, 1);
    this->patch_length = patch_length;
}

// Funzione per la separazione audio
vector<MatrixXd> Audio_Separation::separate_audio(int recons_resolution, double alpha) {
    cout << "reconstructing given network..." << endl;

    // Estrai la matrice di spettrogramma e cre la matrice A
    MatrixXd A = this->spectrogram;
    Eigen::Map<Eigen::MatrixXd> A_matrix(A.data(), A.size() / A.cols(), A.cols());
    // Ottieni le dimensioni
    int m = A_matrix.rows();
    int n = A_matrix.cols();

    //Inizializzazione delle matrici di spettrogramma separate e dizionari
    vector<MatrixXd> separated_specs, zeroed_dicts, separated_dicts;
    separated_specs.reserve(this->input_dictionaries.size());
    zeroed_dicts.reserve(this->input_dictionaries.size());
    separated_dicts.reserve(this->input_dictionaries.size());
    // Inizializza le matrici di spettrogramma separate e dizionari
    for (const auto &dict : this->input_dictionaries) {
        separated_specs.emplace_back(MatrixXd::Zero(m, n));
        zeroed_dicts.emplace_back(MatrixXd::Zero(dict.rows(), dict.cols()));
    }

    // Concatena il dizionario con un dizionario nullo
    for (size_t i = 0; i < this->input_dictionaries.size(); ++i) {
        MatrixXd temp = zeroed_dicts[i];
        zeroed_dicts[i] = this->input_dictionaries[i];
        separated_dicts.push_back(concatenate(zeroed_dicts, 1));
        zeroed_dicts[i] = temp;
    }

    // Inizializza la matrice di conteggio degli overlap
    MatrixXd A_overlap_count = MatrixXd::Zero(m, n);
    int k = this->patch_length;
    auto t0 = chrono::steady_clock::now();

    // Calcola il numero di passi e inizia il ciclo
    int total_steps = (n - k) / recons_resolution + 1;
    int current_step = 0;

    // Ciclo per ogni patch
    for (int i = 0; i <= n - k; i += recons_resolution) {  
        // Estrae il patch
        MatrixXd patch = A.block(0, i, m, k);
        // Flatten del patch in un vettore colonna
        MatrixXd patchVec = patch.reshaped(m * k, 1);

        // Calcola l'encoding (funzione update_code_within_radius())
        MatrixXd code = update_code_within_radius(patchVec, this->W, MatrixXd(), 0.0, alpha, 100, 0.01);

        // Separazione delle ricostruzioni per ogni dizionario
        vector<MatrixXd> patch_recons_list;
        patch_recons_list.reserve(separated_dicts.size());
        // Calcola la ricostruzione per ogni dizionario
        for (auto &D : separated_dicts) {
            MatrixXd recon = D * code;
            // Riorganizza la ricostruzione in una matrice di dimensione M x k
            MatrixXd reconMat = recon.reshaped(m, k);
            patch_recons_list.push_back(std::move(reconMat));
        }
        
        // Overlap-add
        for (int r_idx = 0; r_idx < m; r_idx++) {
            for (int col_offset = 0; col_offset < k; col_offset++) {
                // Calcola la media pesata degli overlap
                double currCount = A_overlap_count(r_idx, i + col_offset);
                for (size_t spec_idx = 0; spec_idx < separated_specs.size(); ++spec_idx) {
                    // Calcola la media pesata
                    double oldVal = separated_specs[spec_idx](r_idx, i + col_offset);
                    double newVal = patch_recons_list[spec_idx](r_idx, col_offset);
                    // Aggiorna la media pesata
                    separated_specs[spec_idx](r_idx, i + col_offset) = (currCount * oldVal + newVal) / (currCount + 1.0);
                }
                // Aggiorna il conteggio degli overlap
                A_overlap_count(r_idx, i + col_offset) += 1.0;
            }
        }
        
        // Stampa il progresso ogni 10 patch o all'ultimo step
        if ((current_step + 1) % 10 == 0 || (current_step + 1) == total_steps) {
            int progress = (current_step + 1) * 100 / total_steps;
            cout << "\rProgress: [" << progress << "%] " << string(progress / 2, '=') << flush;
        }
    
        current_step++;
    }
    auto t1 = chrono::steady_clock::now();
    double elapsed = chrono::duration_cast<chrono::duration<double>>(t1 - t0).count();
    cout << "Reconstructed in " << elapsed << " seconds" << endl;
    // Ritorna le matrici di spettrogramma separate
    return separated_specs;
}

MatrixXcd topic_to_stft(const MatrixXcd &stft, const MatrixXd &NMF_Sxx, const MatrixXd &topic) {
    // Ottieni le dimensioni
    int rows = stft.rows();
    int cols = stft.cols();
    
    // Alloca memoria device per gli array (assumendo dati in double)
    double *d_stft = nullptr, *d_Sxx = nullptr, *d_topic = nullptr, *d_out = nullptr;
    cudaMalloc(&d_stft, rows * cols * sizeof(double));
    cudaMalloc(&d_Sxx, rows * cols * sizeof(double));
    cudaMalloc(&d_topic, rows * cols * sizeof(double));
    cudaMalloc(&d_out, rows * cols * sizeof(double));

    // Copia i dati degli host alle memorie device
    cudaMemcpy(d_stft, stft.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sxx, NMF_Sxx.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_topic, topic.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);

    // Configura il kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Lancia il kernel che esegue: out = (stft * (topic / NMF_Sxx)) (la divisione è elemento per elemento)
    topicToSTFTKernel<<<numBlocks, threadsPerBlock>>>(d_stft, d_Sxx, d_topic, d_out, rows, cols);
    cudaDeviceSynchronize();

    // Copia il risultato dal device all'host
    MatrixXcd output(rows, cols);
    cudaMemcpy(output.data(), d_out, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Libera la memoria allocata sul device
    cudaFree(d_stft);
    cudaFree(d_Sxx);
    cudaFree(d_topic);
    cudaFree(d_out);

    return output;
}