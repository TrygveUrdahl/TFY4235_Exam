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
    std::cout << "---WRONG PROGRAM ARGUMENTS GIVEN--- " << std::endl;
    std::cout << "Requires argument(s): " << std::endl;
    std::cout << "\tint N: problemsize in 1st dir" << std::endl;
    std::cout << "\tint M: problemsize in 2nd dir" << std::endl;
    std::cout << "\tint job: which job to do (0, 1, or 2)" << std::endl;
    std::cout << "Voluntary argument(s): " << std::endl;
    std::cout << "\tint iterations: how many iterations to maximally do (default 1000)" << std::endl;
    std::cout << "\tfloat xA: how many of the atoms are of type A (default 0.5)" << std::endl;
    std::cout << "\tfloat r: atom spacing (default 1.3)" << std::endl;
    std::cout << "\tfloat beta: beta value (default 10.0)" << std::endl;

    throw std::logic_error("Wrong program arguments! ");
  }
  // Get program argument variables
  const int N = std::atoi(argv[1]);
  const int M = std::atoi(argv[2]);
  const int job = std::atoi(argv[3]);
  const int numAtoms = N * M;
  int iterations = 1000;
  if (argc > 4) {
    iterations = std::atoi(argv[4]);
  }
  double xA = 0.5;
  if (argc > 5) {
    xA = std::atof(argv[5]);
  }
  double r = 1.3;
  if(argc > 6) {
    r = std::atof(argv[6]);
  }
  double beta = 10.0;
  if(argc > 7) {
    beta = std::atof(argv[7]);
  }
  std::cout << "Initializations complete, starting job. " << std::endl;
  // Select which job to run
  if (job == 0) {
    std::cout << "Calculate eigenenergys of a system. " << std::endl;
    const arma::uvec neighbourVec = generateNeighbourVec(N, M);
    const arma::uvec atomType = generateAtomTypeVec(numAtoms, xA);
    const arma::sp_mat Htot = generateHtot(N, M, xA, atomType, neighbourVec, r);
    arma::vec eigvals;
    solveSystem(eigvals, Htot);
    eigvals.save("../output/eigvals.h5", arma::hdf5_binary);
  }
  else if (job == 1) {
    std::cout << "Find a best configuration to minimize free energy. " << std::endl;
    int iterationsDone = 0;
    arma::uvec bestConfig = monteCarloBestShuffle(N, M, xA, beta, iterationsDone, iterations, r);
    const arma::umat bestConfigMat = arma::reshape(arma::umat(bestConfig.memptr(), bestConfig.n_elem, true, false), N, M);
    bestConfigMat.save("../output/config.h5", arma::hdf5_binary);
  }
  else if (job == 2) {
    std::cout << "Get enthalpy evolution of systems. " << std::endl;
    arma::uvec iterationCount;
    const arma::vec enthalpys = getEnthalpyChanges(N, M, beta, iterations, iterationCount, r);
    enthalpys.save("../output/enthalpys.h5", arma::hdf5_binary);
    iterationCount.save("../output/iterationCount.h5", arma::hdf5_binary);
  }
  std::cout << "Program finished! " << std::endl;
  return 0;
}
