#include <iostream>
#include <cmath>
#include <armadillo>
#include <chrono>
#include <omp.h>
#include <hdf5.h>

#include "utils.hpp"

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
  const double xA = 0.5;
  const int numAtoms = N * M;
  arma::vec neighbourList = generateNeighbourVec(N, M);
  arma::sp_mat Htot = generateHtot(N, M, xA);
  std::cout << arma::mat(Htot) << std::endl;


  return 0;
}
