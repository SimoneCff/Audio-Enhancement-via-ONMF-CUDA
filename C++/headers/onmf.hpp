#ifndef ONMF_HPP
#define ONMF_HPP

#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

#include <random>

// Using Eigen namespace for matrix operations
using namespace Eigen;
using namespace std;

// Global debug flag
const bool DEBUG = false;

// Forward declaration of custom sparsecoder function
MatrixXd update_code_within_radius(const MatrixXd &X, const MatrixXd &W, const MatrixXd &H0, double r, double alpha, int sub_iter, double stopping_diff);

class Online_NMF {
    public:
        MatrixXd X;
        int n_components;
        int iterations;
        int batch_size;
        bool subsample;
        MatrixXd initial_dict;
        MatrixXd initial_A;
        MatrixXd initial_B;
        MatrixXd initial_C;
        int history;
        double alpha;
        double beta;
        MatrixXd code; // dimensions: n_components x number of samples
    
        // Constructor
        Online_NMF(const MatrixXd &X,
                   int n_components = 100,
                   int iterations = 500,
                   int batch_size = 20,
                   const MatrixXd &ini_dict = MatrixXd(),
                   const MatrixXd &ini_A = MatrixXd(),
                   const MatrixXd &ini_B = MatrixXd(),
                   const MatrixXd &ini_C = MatrixXd(),
                   int history = 0,
                   double alpha = -1, // negative means not provided
                   double beta = -1,
                   bool subsample = false)
        {
            /* 
            X: data matrix
            n_components (int): number of columns in dictionary matrix W where each column represents one topic/feature
            iterations (int): number of iterations where each iteration is a call to step(...)
            batch_size (int): number of random columns of X that will be sampled during each iteration
            */
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
    
        MatrixXd sparse_code(const MatrixXd &X, const MatrixXd &W) {
            /*
            Given data matrix X and dictionary matrix W, find
            code matrix H such that W*H approximates X
            args:
                X (MatrixXd): data matrix with dimensions: features (d) x samples (n)
                W (MatrixXd): dictionary matrix with dimensions: features (d) x topics (r)
            returns:
                H (MatrixXd): code matrix with dimensions: topics (r) x samples(n)
            */
            if (DEBUG) {
                cout << "sparse_code" << endl;
                cout << "X.shape: (" << X.rows() << ", " << X.cols() << ")" << endl;
                cout << "W.shape: (" << W.rows() << ", " << W.cols() << ")" << endl << endl;
            }
            double a = (this->alpha);
            MatrixXd H = update_code_within_radius(X, W, MatrixXd(), 0.0, a, 10, 0.01);
            // transpose not needed because our update_code returns the code matrix directly analogous to Python.
            return H;
        }
    
        MatrixXd update_dict(const MatrixXd &W, const MatrixXd &A, const MatrixXd &B) {
            /*
            Updates dictionary matrix W using new aggregate matrices A and B
            args:
                W (MatrixXd): dictionary matrix with dimensions: data_dim (d) x topics (r)
                A (MatrixXd): aggregate matrix with dimensions: topics (r) x topics(r)
                B (MatrixXd): aggregate matrix with dimensions: topics (r) x data_dim (d)
            returns:
                W1 (MatrixXd): updated dictionary matrix with dimensions: features (d) x topics (r)
            */
            int d = W.rows();
            int r = W.cols();
            MatrixXd W1 = W;
            for (int j = 0; j < r; j++) {
                // W1(:,j) = W1(:,j) - (1/(A(j,j)+1))*(W1 * A(:,j) - B.transpose()( :,j))
                VectorXd Aj = A.col(j);
                VectorXd temp = (W1 * Aj) - B.transpose().col(j);
                double denom = A(j,j) + 1;
                W1.col(j) = W1.col(j) - (1.0 / denom) * temp;
                // element-wise maximum with zero
                for (int i = 0; i < d; i++) {
                    W1(i, j) = max(W1(i, j), 0.0);
                }
                // normalize column: divide by max(1, norm)
                double norm = W1.col(j).norm();
                norm = max(norm, 1.0);
                W1.col(j) = (1.0 / norm) * W1.col(j);
            }
            return W1;
        }
    
        tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> step(const MatrixXd &X, const MatrixXd &A, const MatrixXd &B, const MatrixXd &C, const MatrixXd &W, double t) {
            /*
            Performs a single iteration of the online NMF algorithm.
            args:
                X (MatrixXd): data matrix with dimensions: data_dim (d) x samples (n)
                A (MatrixXd): aggregate matrix with dimensions: topics (r) x topics(r)
                B (MatrixXd): aggregate matrix with dimensions: topics (r) x data_dim (d)
                W (MatrixXd): dictionary matrix with dimensions: data_dim (d) x topics (r)
                t (double): current iteration of the online algorithm
            returns:
                tuple of H1, A1, B1, W1
            */
            int d = X.rows();
            int n = X.cols();
            int r = W.cols();
    
            // Compute H1 by sparse coding X using dictionary W
            MatrixXd H1 = this->sparse_code(X, W);
            if (DEBUG) {
                cout << "H1.shape: (" << H1.rows() << ", " << H1.cols() << ")" << endl;
            }
            // Update aggregate matrices A and B
            double betaVal = this->beta;
            MatrixXd A1 = (1.0 - pow(t, -betaVal)) * A + pow(t, -betaVal) * (H1 * H1.transpose());
            MatrixXd B1 = (1.0 - pow(t, -betaVal)) * B + pow(t, -betaVal) * (H1 * X.transpose());
            // C update is commented out as in Python
    
            // Update dictionary matrix
            MatrixXd W1 = this->update_dict(W, A, B);
            this->history = int(t) + 1;
            return make_tuple(H1, A1, B1, W1);
        }
    
        tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> train_dict() {
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
    };
    
    // ---------------------------
    // custom sparsecoder
    // ---------------------------
    MatrixXd update_code_within_radius(const MatrixXd &X, const MatrixXd &W, const MatrixXd &H0_in, double r, double alpha,
                                       int sub_iter, double stopping_diff) {
        /*
        Find H_hat = argmin_H ( | X - W*H| + alpha|H| ) within radius r from H0
        Use row-wise projected gradient descent
        Do NOT sparsecode the whole thing and then project -- instable
        12/5/2020 Lyu
        Using custom sparsecoding strategy.
        */
        MatrixXd A = W.transpose() * W;
        MatrixXd B = W.transpose() * X;
        MatrixXd H0 = (H0_in.size() == 0)
                        ? MatrixXd::Random(W.cols(), X.cols()).cwiseAbs()
                        : H0_in;
        MatrixXd H = H0;
    
        int i = 0;
        double dist = 1.0;
        while (i < sub_iter && dist > stopping_diff) {
            MatrixXd H_old = H;
            double step_size = 1.0 / (A.diagonal().array() + 1).sqrt().mean();
            for (int k = 0; k < H.rows(); k++) {
                RowVectorXd grad = (A.row(k) * H) - B.row(k) + alpha * RowVectorXd::Ones(H.cols());
                H.row(k) = (H.row(k).array() - step_size * grad.array()).max(0.0);
            }
            if (r > 0) {
                double dnorm = (H - H0).norm();
                if (dnorm > r) {
                    H = H0 + (r / dnorm) * (H - H0);
                }
            }
            dist = (H - H_old).norm() / (H_old.norm() + 1e-12);
            i++;
        }
        return H;
    }
    

#endif