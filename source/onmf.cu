#include "../headers/onmf.hpp"
#include <cuda_runtime.h>

//Versione Global per la moltiplicazione tra matrici A e B 
__global__ void matMulTransposeKernel(const double* A, const double* B, double* C, int A_rows, int A_cols, int B_cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // indice per A^T (colonna di A)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // indice per colonna di B
    // C(i, j) = Aᵀ(i, :) * B(:, j)
    if (i < A_cols && j < B_cols) {
        // Calcola il prodotto scalare tra la riga i-esima di A^T e la colonna j-esima di B
        double sum = 0.0;
        for (int k = 0; k < A_rows; k++) {
            // A^T(i, k) = A(k, i)
            sum += A[k * A_cols + i] * B[k * B_cols + j];
        }
        // Salva il risultato
        C[i * B_cols + j] = sum;
    }
}

//Versione Global per  l'aggiornamento di H e calcolo della distanza
__global__ void updateAndDistanceKernel(const double* A, const double* B,
                                        double* H, const double* H0,
                                        const double* H_old,
                                        double* diffOut, double* oldOut,
                                        int rows, int cols,
                                        int A_cols, double alpha, double r, int i)
{
    // Allocazione della shared memory
    extern __shared__ double sdata[]; 
    double* sdiff = sdata;              
    double* sold = &sdata[blockDim.x];  

    // Indici globali
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    // Indice locale
    int local_tid = threadIdx.x;

    // Somme parziali
    double threadDiffSum = 0.0;
    double threadOldSum = 0.0;

    if (k < rows) {
        // Aggiornamento di H
        for (int j = 0; j < cols; ++j) {
            // Computa gradiente
            double sumVal = 0.0;
            // Calcola la somma per la riga k-esima di Aᵀ e la colonna j-esima di H
            for (int m = 0; m < A_cols; ++m) {
                sumVal += A[k * A_cols + m] * H[m * cols + j];
            }
            // Calcola il gradiente
            double grad = sumVal - B[k * cols + j] + alpha;

            // Passo di discesa del gradiente
            double step = 1.0 / (((i + 10) * A[k * A_cols + k]) + 1.0);
            // Aggiorna H
            double newVal = H[k * cols + j] - step * grad;
            H[k * cols + j] = fmax(0.0, newVal); // clamp a 0

        }
        
        // Proiezione r
        if (r >= 0) {
            double diffNorm = 0.0;
            // Calcola la norma della differenza tra H e H0
            for (int j = 0; j < cols; ++j) {
                double d = H[k * cols + j] - H0[k * cols + j];
                diffNorm += d*d;
            }
            // Calcola la norma della differenza
            diffNorm = sqrt(diffNorm);
            // Se la norma è maggiore di r, proietta H su H0
            if (diffNorm > r) {
                // Scala H0 verso H
                double scale = r / diffNorm;
                // Aggiorna H
                for (int j = 0; j < cols; ++j) {
                    double val = H0[k * cols + j] 
                               + scale * (H[k * cols + j] - H0[k * cols + j]);
                    H[k * cols + j] = val;
                }
            }
        }

        // Calcolo parziale della distanza
        // (norm(H - H_old)^2 e norm(H_old)^2)
        for (int j = 0; j < cols; ++j) {
            double d = (H[k * cols + j] - H_old[k * cols + j]);
            threadDiffSum += d*d;
            double oldval = H_old[k * cols + j];
            threadOldSum += oldval*oldval;
        }
    }

    // Riduzione interna al blocco
    sdiff[local_tid] = threadDiffSum;
    sold[local_tid]  = threadOldSum;
    __syncthreads();

    // Riduzione parallela lungo la dimensione x (righe)
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        // Riduzione
        if (local_tid < s) {
            sdiff[local_tid] += sdiff[local_tid + s];
            sold[local_tid]  += sold[local_tid + s];
        }
        __syncthreads();
    }

    // Salva risultati parziali
    if (local_tid == 0) {
        diffOut[blockIdx.x] = sdiff[0];
        oldOut[blockIdx.x]  = sold[0];
    }
}

__global__ void update_dict_k(double* W1, const double* A, const double* B, int d, int r) {
    // Indici globali
    int j = blockIdx.x;
    int i = threadIdx.x;
    // Se l'indice i è maggiore della dimensione d, non fare nulla
    if (j >= r) return;
    // Allocazione della shared memory
    extern __shared__ double shared[];
    // Aggiorna W1(i,j)
    double updated_val = 0.0;

    if (i < d) {
        double sum = 0.0;
        // Calcolo (W1 * A(:,j))(i) = sum_{k=0}^{r-1} W1(i,k) * A(k,j)
        for (int k = 0; k < r; k++) {
            // W1(i,k): memoria in col-major => indice = i + k*d
            // A(k,j): matrice A (dimensione r x r) in col-major => indice = k + j*r
            sum += W1[i + k * d] * A[k + j * r];
        }
        // B è considerata come matrice di dimensioni (r x d) in col-major
        // e B(j,i) si ottiene come B[j + i*r]
        double B_val = B[j + i * r];
        double temp = sum - B_val;
        // Calcolo denom = A(j,j) + 1.0, con A(j,j) = A[j + j*r]
        double denom = A[j + j * r] + 1.0;
        updated_val = W1[i + j * d] - (1.0 / denom) * temp;
        if (updated_val < 0.0) {
            updated_val = 0.0;
        }
        // Aggiorna l'elemento
        W1[i + j * d] = updated_val;
    }
    
    // Riduzione per calcolare la norma (somma dei quadrati) nella shared memory
    double val_s = (i < d) ? updated_val * updated_val : 0.0;
    shared[i] = val_s;
    __syncthreads();
    
    // Riduzione parallela lungo la dimensione i (righe)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) {
            shared[i] += shared[i + s];
        }
        __syncthreads();
    }
    // Aggiorna la norma
    double norm = 0.0;
    // Il thread 0 calcola la norma
    if (i == 0) {
        norm = sqrt(shared[0]);
        if (norm < 1.0) {
            norm = 1.0;
        }
        shared[0] = norm;
    }
    __syncthreads();
    
    // Normalizza la colonna j-esima
    norm = shared[0];
    if (i < d) {
        W1[i + j * d] /= norm;
    }
}

//Kernel per l'aggiornamento di A1 = 1.0 - (-beta)^t * A + (-beta)^t * (H1 * H1^T)
__global__ void update_A1(double* A1, const double *A, const double *H1, int rows, int cols, double alpha, double beta) {
    // Indici globali
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        // Calcola la somma per la riga row-esima di H1 e la colonna col-esima di H1
        double sum = 0.0;
        for (int k = 0; k < cols; k++) {
            // H1(row, k) * H1(col, k) (H1 traslato in col-major)
            sum += H1[row + k * rows] * H1[col + k * rows];
        }
        // Aggiorna A1
        A1[row + col * rows] = A[row + col * rows] + alpha * (sum - beta * A[row + col * rows]);
    }
}

//Kernel per l'aggiornamento di B1 (1 - (-beta)^t) * B + (-beta)^t * (H1 *X^T)
__global__ void update_B1(double *B1, const double *B,const double *H1, const double *X, int rows,int cols,int K,double alpha,double beta) {
    // Indici globali
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            // Calcola la somma per la riga row-esima di H1 e la colonna col-esima di X translato
            sum += H1[row + k * rows] * X[col + k * cols];
        }
        // Aggiorna B1
        B1[row + col * rows] = beta * B[row + col * rows] + alpha * sum;
    }
}


//Costruttore
Online_NMF::Online_NMF(const MatrixXd &X, int n_components,int iterations,int batch_size, const MatrixXd &ini_dict,const MatrixXd &ini_A,const MatrixXd &ini_B,
    const MatrixXd &ini_C,int history,double alpha,double beta,bool subsample)
{

    this->X = X;
    this->n_components = n_components;
    this->batch_size = batch_size;
    this->iterations = iterations;
    this->subsample = subsample;
    this->initial_dict = ini_dict;
    this->initial_A = ini_A;
    this->initial_B = ini_B;
    this->initial_C = ini_C;
    this->history = history;
    this->alpha = (alpha < 0 ? 0 : alpha);
    this->beta = (beta < 0 ? 1 : beta);
    // Initialize code as zero matrix with shape (n_components, X.cols())
    this->code = MatrixXd::Zero(n_components, X.cols());
}

//Funzione per il calcolo della matrice sparsa H
MatrixXd Online_NMF::sparse_code(const MatrixXd &X, const MatrixXd &W) {
    if (DEBUG) {
        cout << "sparse_code" << endl;
        cout << "X.shape: (" << X.rows() << ", " << X.cols() << ")" << endl;
        cout << "W.shape: (" << W.rows() << ", " << W.cols() << ")" << endl << endl;
    }
    //assegnazione di alpha
    double a = (this->alpha);
    //Chiamata alla funzione update_code_within_radius
    MatrixXd H = update_code_within_radius(X, W, MatrixXd(), 0.0, a, 10, 0.01);

    return H;
}

// Funzione per l'aggiornamento di A1 , B1, W1 e H1
tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> Online_NMF::step(const MatrixXd &X, const MatrixXd &A, const MatrixXd &B, const MatrixXd &C, const MatrixXd &W, double t) {
    //Calcolo della matrice sparsa H1
    MatrixXd H1 = sparse_code(X, W);
    if (DEBUG) {
        cout << "H1: " << H1.rows() << "x" << H1.cols() << endl;
    }

    double betaVal = this->beta;

    //Call funzioni update A1 e B1
    MatrixXd A1 = A;
    MatrixXd B1 = B;

    //Scalari A1 e B1
    double alpha = pow(t, -betaVal);
    double beta = 1.0 - alpha;

    //Aggiornamento di A1 tramite kernel
    int rows = A1.rows();
    int cols = A1.cols();

    //Kernel info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    //Dimensioni dei blocchi
    int maxThreads = prop.maxThreadsPerBlock;
    int dim = (int)sqrt((double)maxThreads);
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //Allocazione GPU
    double *A1_dev, *A_dev, *H1_dev;
    cudaMalloc((void**)&A1_dev, A1.size() * sizeof(double));
    cudaMalloc((void**)&A_dev, A.size() * sizeof(double));
    cudaMalloc((void**)&H1_dev, H1.size() * sizeof(double));

    //Copia dati su GPU
    cudaMemcpy(A1_dev, A1.data(), A1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(A_dev, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(H1_dev, H1.data(), H1.size() * sizeof(double), cudaMemcpyHostToDevice);

    update_A1<<<numBlocks, threadsPerBlock>>>(A1_dev, A_dev, H1_dev, rows, cols, alpha, beta);\

    //Copia dati da GPU
    cudaMemcpy(A1.data(), A1_dev, A1.size() * sizeof(double), cudaMemcpyDeviceToHost);

    //Rilascio memoria
    cudaFree(A1_dev);
    cudaFree(A_dev);

    //Aggiornamento di B1 tramite kernel
    int rows_B = B1.rows();
    int cols_B = B1.cols();
    int K = H1.cols();

   //Allocazione GPU
    double *B1_dev, *B_dev, *X_dev;
    cudaMalloc((void**)&B1_dev, B1.size() * sizeof(double));
    cudaMalloc((void**)&B_dev, B.size() * sizeof(double));
    cudaMalloc((void**)&X_dev, X.size() * sizeof(double));

    //Copia dati su GPU
    cudaMemcpy(B1_dev, B1.data(), B1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(X_dev, X.data(), X.size() * sizeof(double), cudaMemcpyHostToDevice);

    //Kernel info
    dim3 threadsPerBlock_B(16, 16);
    dim3 numBlocks_B((rows_B + threadsPerBlock_B.x - 1) / threadsPerBlock_B.x, (cols_B + threadsPerBlock_B.y - 1) / threadsPerBlock_B.y);
    update_B1<<<numBlocks_B, threadsPerBlock_B>>>(B1_dev,B_dev,H1_dev,X_dev,rows_B,cols_B,K,alpha,beta);

    //Copia dati da GPU
    cudaMemcpy(B1.data(), B1_dev, B1.size() * sizeof(double), cudaMemcpyDeviceToHost);

    //Rilascio memoria
    cudaFree(B1_dev);
    cudaFree(B_dev);
    cudaFree(X_dev);
    cudaFree(H1_dev);

    //Aggiornamento di W1 tramite kernel
    MatrixXd W1 = W;
    double *d_W1, *d_A, *d_B;

    //Allocazione GPU
    cudaMalloc((void**)&d_W1, W1.size() * sizeof(double));
    cudaMalloc((void**)&d_A, A.size() * sizeof(double));
    cudaMalloc((void**)&d_B, B.size() * sizeof(double));

    //Copia dati su GPU
    cudaMemcpy(d_W1, W1.data(), W1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A1.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B1.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice);

    //Kernel info
    dim3 grid(W1.cols());
    dim3 block(W1.rows());
    size_t shared_size = W1.rows() * sizeof(double);

    //Chiamata kernel
    update_dict_k<<<grid, block, shared_size>>>(d_W1, d_A, d_B, W1.rows(), W1.cols());
    cudaDeviceSynchronize();

    //Copia dati da GPU
    cudaMemcpy(W1.data(), d_W1, W1.size() * sizeof(double), cudaMemcpyDeviceToHost);
    
    this-> history = int(t)+1;
    //Rilascio memoria
    cudaFree(d_W1);
    cudaFree(d_A);
    cudaFree(d_B);

    return make_tuple(H1, A1, B1, W1);
}

//Funzione per l'addestramento del dizionario
tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> Online_NMF::train_dict() {
    /*
    Learns a dictionary matrix W with n_components number of columns based
    on a fixed data matrix X
    args:
        X (MatrixXd): data matrix with dimensions: data_dim (d) x samples (n)
    return:
        tuple: W, A, B, code
    */
    int d = this->X.rows();
    int n = this->X.cols();
    int r = this->n_components;
    // use class member code matrix
    // If initial dictionary not provided: initialize randomly and A, B with zeros.
    MatrixXd W, A, B, C;
    int t0 = this->history;
    if (this->initial_dict.size() == 0) {
        // initialize dictionary matrix W with random values
        W = MatrixXd::Random(d, r).cwiseAbs();
        A = MatrixXd::Zero(r, r);
        B = MatrixXd::Zero(r, d);
        C = MatrixXd::Zero(d, d);
    } else {
        W = this->initial_dict;
        A = this->initial_A;
        B = this->initial_B;
        C = this->initial_C;
    }
    // Iterative update
    for (int i = 1; i < this->iterations; i++) {
        // choose indices
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        if (this->subsample) {
            // random batch sampling
            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<> dis(0, n - 1);
            vector<int> temp;
            for (int k = 0; k < this->batch_size; k++) {
                temp.push_back(dis(gen));
            }
            idx = temp;
        }
        // Form X_batch from selected columns
        MatrixXd X_batch(d, idx.size());
        for (size_t j = 0; j < idx.size(); j++) {
            X_batch.col(j) = this->X.col(idx[j]);
        }
        // Update W, A, B using step
        MatrixXd H;
        tie(H, A, B, W) = this->step(X_batch, A, B, C, W, double(t0 + i));
        // Update code matrix for indices:
        for (size_t j = 0; j < idx.size(); j++) {
            this->code.col(idx[j]) = H.col(j);
        }
    }
    return make_tuple(W, A, B, this->code);
}

//Funzione per l'aggiornamento di H_hat
MatrixXd update_code_within_radius(const MatrixXd &X, const MatrixXd &W, const MatrixXd &H0_in, double r, double alpha,
                                       int sub_iter, double stopping_diff) {
        /*
        Find H_hat = argmin_H ( | X - W*H| + alpha|H| ) within radius r from H0
        Use row-wise projected gradient descent
        Do NOT sparsecode the whole thing and then project -- instable
        12/5/2020 Lyu
        Using custom sparsecoding strategy.
        */
        //Calcolo A e B
        MatrixXd A(W.cols(), W.cols());
        MatrixXd B(W.cols(), X.cols());
        //Richiamo Kernel
        double *d_W, *d_X;
        double *d_A, *d_B;
        //Alloco memoria
        cudaMalloc(&d_W, W.rows() * W.cols() * sizeof(double));
        cudaMalloc(&d_X, X.rows() * X.cols() * sizeof(double));
        cudaMalloc(&d_A, W.cols() * W.cols() * sizeof(double));
        cudaMalloc(&d_B, W.cols() * X.cols() * sizeof(double));
        //Copio i dati
        cudaMemcpy(d_W, W.data(), W.rows() * W.cols() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X, X.data(), X.rows() * X.cols() * sizeof(double), cudaMemcpyHostToDevice);
        //Definisco le dimensioni
        dim3 BlockDims(16, 16);
        dim3 numBlocks((W.cols() + BlockDims.x - 1) / BlockDims.x,
                       (X.cols() + BlockDims.y - 1) / BlockDims.y);
        //Chiamo il kernel
        matMulTransposeKernel<<<numBlocks, BlockDims>>>(d_W, d_W, d_A, W.rows(), W.cols(), W.cols());
        matMulTransposeKernel<<<numBlocks, BlockDims>>>(d_W, d_X, d_B, W.rows(), W.cols(), X.cols());
        //Copio i dati
        cudaMemcpy(A.data(), d_A, W.cols() * W.cols() * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(B.data(), d_B, W.cols() * X.cols() * sizeof(double), cudaMemcpyDeviceToHost);
        //Calcolo H e H0
        MatrixXd H0 = (H0_in.size() == 0)
                        ? MatrixXd::Random(W.cols(), X.cols()).cwiseAbs()
                        : H0_in;
        MatrixXd H = H0;
        //kernel H
        int threadsPerBlock = 256;
        int blocks = (H.rows() + threadsPerBlock - 1) / threadsPerBlock;
        size_t sharedMemSize = 2 * threadsPerBlock * sizeof(double);
        //Alloco memoria
        double *d_H, *d_H0;
        double* d_H_old = nullptr;
        double* d_diffOut = nullptr;
        double* d_oldOut  = nullptr;
        //CudaMalloc
        cudaMalloc(&d_H, H.rows() * H.cols() * sizeof(double));
        cudaMalloc(&d_H0, H0.rows() * H0.cols() * sizeof(double));
        cudaMalloc(&d_diffOut, blocks * sizeof(double));
        cudaMalloc(&d_oldOut,  blocks * sizeof(double));
        cudaMalloc(&d_H_old, H.size() * sizeof(double));
        //Copia dati
        cudaMemcpy(d_H, H.data(), H.rows() * H.cols() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_H0, H0.data(), H0.rows() * H0.cols() * sizeof(double), cudaMemcpyHostToDevice);
        //Avvio kernel
        int i = 0;
        double dist = 1.0;
        while (i < sub_iter && dist > stopping_diff) {
            cudaMemcpy(d_H_old, d_H, H.size() * sizeof(double), cudaMemcpyDeviceToDevice);updateAndDistanceKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
                d_A, d_B, d_H, d_H0, d_H_old,
                d_diffOut, d_oldOut,
                H.rows(), H.cols(),
                A.cols(), alpha, r, i
            );
            cudaDeviceSynchronize();
             // Copia i risultati parziali in host
            vector<double> h_diffOut(blocks), h_oldOut(blocks);
            cudaMemcpy(h_diffOut.data(), d_diffOut, blocks * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_oldOut.data(),  d_oldOut,  blocks * sizeof(double), cudaMemcpyDeviceToHost);

            // Somma e calcola dist
            double sumDiff = 0.0, sumOld = 0.0;
            for (int b = 0; b < blocks; b++) {
                sumDiff += h_diffOut[b];
                sumOld  += h_oldOut[b];
            }
            double sqrtOld = sqrt(sumOld);
            dist = (sqrtOld > 1e-9) ? sqrt(sumDiff) / sqrtOld : 0.0;
            i++;
        }
        //Copia i dati
        cudaMemcpy(H.data(), d_H, H.rows() * H.cols() * sizeof(double), cudaMemcpyDeviceToHost);

        //Dealloco memoria
        cudaFree(d_W);
        cudaFree(d_X);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_H);
        cudaFree(d_H0);
        cudaFree(d_H_old);
        cudaFree(d_diffOut);
        cudaFree(d_oldOut);

        return H;
    }
