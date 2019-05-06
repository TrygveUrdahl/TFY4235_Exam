#include <iostream>
#include <cmath>
#include <armadillo>
#include <chrono>
#include <omp.h>

#include "utils.hpp"

arma::uvec monteCarloBestShuffle(int N, int M, double xA, double beta, int iterations) {
  const double numAtoms = N * M;
  arma::uvec bestAtomConfig;
  double bestEnergy = - std::numeric_limits<double>::max();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < iterations; i++) {
    const arma::uvec atomType = generateAtomTypeVec(numAtoms, xA);
    const arma::uvec neighbourList = generateNeighbourVec(N, M);
    arma::sp_mat Htot = generateHtot(N, M, xA, atomType, neighbourList);
    arma::vec eigvals;
    solveSystem(eigvals, Htot);
    double freeEnergy = getFreeEnergy(beta, eigvals);
    #pragma omp critical
    if (freeEnergy > bestEnergy) {
      bestEnergy = freeEnergy;
      bestAtomConfig = atomType;
    }
  }
  // std::cout << "bestEnergy: " << bestEnergy << std::endl;
  return bestAtomConfig;
}
