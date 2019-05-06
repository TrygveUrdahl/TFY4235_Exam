#include <iostream>
#include <cmath>
#include <armadillo>
#include <chrono>
#include <omp.h>
#include <hdf5.h>

#include "utils.hpp"
#include "montecarlo.hpp"

int main(int argc, char** argv) {
  arma::arma_rng::set_seed_random();
  if (argc < 3) {
    std::cout << "Requires argument(s): " << std::endl;
    std::cout << "\tint N: problemsize in 1st dir" << std::endl;
    std::cout << "\tint M: problemsize in 2nd dir" << std::endl;
    throw std::logic_error("Wrong program arguments! ");
  }
  const int N = atoi(argv[1]);
  const int M = atoi(argv[2]);
  int iterations = 100;
  if (argc > 3) {
    iterations = std::atoi(argv[3]);
  }
  double xA = 0.5;
  if (argc > 4) {
    xA = std::atof(argv[4]);
  }
  const double beta = 10.0;
  // const int numAtoms = N * M;
  // arma::sp_mat Htot = generateHtot(N, M, xA);
  // arma::vec eigvals;
  // solveSystem(eigvals, Htot);
  // std::cout << arma::mat(Htot) << std::endl;
  // std::cout << eigvals << std::endl;
  arma::uvec bestConfig = monteCarloBestShuffle(N, M, xA, beta, iterations);
  //arma::vec enthalpys = getEnthalpyChanges(N, M, beta, iterations);

  arma::umat bestConfigMat = arma::reshape(arma::umat(bestConfig.memptr(), bestConfig.n_elem, 1, false), N, M);
  // enthalpys.save("../output/enthalpys.h5", arma::hdf5_binary);
  bestConfigMat.save("../output/config.h5", arma::hdf5_binary);
  return 0;
}
