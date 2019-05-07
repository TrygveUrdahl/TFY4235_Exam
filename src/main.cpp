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
    std::cout << "\tint job: which job to do" << std::endl;
    throw std::logic_error("Wrong program arguments! ");
  }
  const int N = std::atoi(argv[1]);
  const int M = std::atoi(argv[2]);
  const int job = std::atoi(argv[3]);
  int iterations = 1000;
  if (argc > 4) {
    iterations = std::atoi(argv[4]);
  }
  double xA = 0.5;
  if (argc > 5) {
    xA = std::atof(argv[5]);
  }
  const double beta = 10.0;
  const int numAtoms = N * M;

  if (job == 0) {
    std::cout << "Calculate eigenenergys of a system. " << std::endl;
    arma::uvec neighbourVec = generateNeighbourVec(N, M);
    arma::uvec atomType = generateAtomTypeVec(numAtoms, xA);
    arma::sp_mat Htot = generateHtot(N, M, xA, atomType, neighbourVec);
    arma::vec eigvals;
    solveSystem(eigvals, Htot);
    eigvals.save("../output/eigvals.h5", arma::hdf5_binary);
  }
  else if (job == 1) {
    std::cout << "Find a best configuration to minimize free energy" << std::endl;
    arma::uvec bestConfig = monteCarloBestShuffleParallel(N, M, xA, beta, iterations);
    std::cout << bestConfig << std::endl;
    arma::umat bestConfigMat = arma::reshape(arma::umat(bestConfig.memptr(), bestConfig.n_elem, true, false), N, M);
    bestConfigMat.save("../output/config.h5", arma::hdf5_binary);
  }
  else if (job == 2) {
    std::cout << "Get enthalpy evolution of systems" << std::endl;
    arma::uvec iterationCount;
    arma::vec enthalpys = getEnthalpyChanges(N, M, beta, iterations, iterationCount);
    enthalpys.save("../output/enthalpys.h5", arma::hdf5_binary);
    iterationCount.save("../output/iterationCount.h5", arma::hdf5_binary);
  }

  return 0;
}
