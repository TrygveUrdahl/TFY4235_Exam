#include <iostream>
#include <cmath>
#include <armadillo>
#include <chrono>
#include <omp.h>

#include "utils.hpp"

arma::uvec monteCarloBestShuffle(int N, int M, double xA, double beta, int &iterations, int maxIterations) {
  const double numAtoms = N * M;
  const arma::uvec neighbourList = generateNeighbourVec(N, M);
  arma::uvec atomType = generateAtomTypeVec(numAtoms, xA);
  bool converge = false;
  iterations = 0;
  arma::uvec bestAtomConfig;
  double bestEnergy = std::numeric_limits<double>::max();
  int notChanged = 0;
  for (int i = 0; i < maxIterations * 0.5; i++) {
    iterations++;
    atomType = generateAtomTypeVec(numAtoms, xA);
    arma::sp_mat Htot = generateHtot(N, M, xA, atomType, neighbourList);
    arma::vec eigvals;
    solveSystem(eigvals, Htot);
    double freeEnergy = getFreeEnergy(beta, eigvals);
    if (freeEnergy < bestEnergy) {
      bestEnergy = freeEnergy;
      bestAtomConfig = atomType;
    }
  }
  atomType = bestAtomConfig;
  arma::uvec nextUse = atomType;
  for (int i = 0; i < maxIterations * 0.5; i++) {
    iterations++;
    atomType = permuteAtomTypeVec(nextUse, neighbourList);
    arma::sp_mat Htot = generateHtot(N, M, xA, atomType, neighbourList);
    arma::vec eigvals;
    solveSystem(eigvals, Htot);
    double freeEnergy = getFreeEnergy(beta, eigvals);
    double deltaEnergy = freeEnergy - bestEnergy;
    if (deltaEnergy <= 0.0) {
      // std::cout << " delta < 0: "<< deltaEnergy << std::endl;
      bestEnergy = freeEnergy;
      bestAtomConfig = atomType;
      notChanged = 0;
      nextUse = atomType;
    }
    else if(std::exp(-deltaEnergy * beta) >= arma::randu<double>()) {
      // std::cout << std::exp(-deltaEnergy * beta) << std::endl;
      nextUse = atomType;
      notChanged++;
    }
    else {
      notChanged++;
    }
    if (notChanged > maxIterations * 0.1) {
      converge = true;
    }
    if (converge) {
      return bestAtomConfig;
    }
  }
  // std::cout << "bestEnergy: " << bestEnergy << std::endl;
  return bestAtomConfig;
}

arma::uvec monteCarloBestShuffleParallel(int N, int M, double xA, double beta, int maxIterations) {
  const double numAtoms = N * M;
  arma::uvec bestAtomConfig;
  double bestEnergy = std::numeric_limits<double>::max();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < maxIterations; i++) {
    const arma::uvec atomType = generateAtomTypeVec(numAtoms, xA);
    const arma::uvec neighbourList = generateNeighbourVec(N, M);
    arma::sp_mat Htot = generateHtot(N, M, xA, atomType, neighbourList);
    arma::vec eigvals;
    solveSystem(eigvals, Htot);
    double freeEnergy = getFreeEnergy(beta, eigvals);
    #pragma omp critical
    if (freeEnergy < bestEnergy) {
      bestEnergy = freeEnergy;
      bestAtomConfig = atomType;
    }
  }
  std::cout << "bestEnergy: " << bestEnergy << std::endl;
  return bestAtomConfig;
}
