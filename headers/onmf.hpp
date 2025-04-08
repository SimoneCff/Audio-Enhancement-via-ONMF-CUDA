#ifndef ONMF_HPP
#define ONMF_HPP

//Librerie Eigen per la manipolazione di matrici
#include <Eigen/Dense>

//Librerie C++
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <random>

using namespace Eigen;
using namespace std;


const bool DEBUG = false;

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
                   double alpha = -1,
                   double beta = -1,
                   bool subsample = false);
        
    
        MatrixXd sparse_code(const MatrixXd &X, const MatrixXd &W);
    
        tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> step(const MatrixXd &X, const MatrixXd &A, const MatrixXd &B, const MatrixXd &C, const MatrixXd &W, double t);
    
        tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> train_dict();
    };
    
//Funzione per l'aggiornamento del codice entro un raggio
MatrixXd update_code_within_radius(const MatrixXd &X, const MatrixXd &W, const MatrixXd &H0_in, double r, double alpha,
                                       int sub_iter, double stopping_diff);

#endif