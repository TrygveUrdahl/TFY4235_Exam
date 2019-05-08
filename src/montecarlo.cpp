#include <iostream>
#include <cmath>
#include <armadillo>
#include <chrono>
#include <omp.h>

#include "utils.hpp"

arma::uvec monteCarloBestShuffle(int N, int M, double xA, double beta, int &iterations, int maxIterations, double r) {
  const double numAtoms = N * M;
  const arma::uvec neighbourList = generateNeighbourVec(N, M);
  arma::uvec atomType = generateAtomTypeVec(numAtoms, xA);
  bool converge = false;
  iterations = 0;
  arma::uvec bestAtomConfig;
  double bestEnergy = std::numeric_limits<double>::max();
  int notChanged = 0;
  for (int i = 0; i < maxIterations * 0.1; i++) {
    atomType = generateAtomTypeVec(numAtoms, xA);
    arma::sp_mat Htot = generateHtot(N, M, xA, atomType, neighbourList, r);
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
  double lastEnergy = bestEnergy;
  for (int i = 0; i < maxIterations; i++) {
    iterations++;
    atomType = permuteAtomTypeVec(nextUse, neighbourList);
    arma::sp_mat Htot = generateHtot(N, M, xA, atomType, neighbourList, r);
    arma::vec eigvals;
    solveSystem(eigvals, Htot);
    double freeEnergy = getFreeEnergy(beta, eigvals);
    double deltaEnergy = freeEnergy - lastEnergy;
    if (deltaEnergy <= 0.0) {
      if (freeEnergy <= bestEnergy) {
        bestEnergy = freeEnergy;
        bestAtomConfig = atomType;
      }
      notChanged = 0;
      nextUse = atomType;
      lastEnergy = freeEnergy;
    }
    else if(std::exp(-deltaEnergy * beta) >= arma::randu<double>()) {
      nextUse = atomType;
      lastEnergy = freeEnergy;
      notChanged++;
    }
    else {
      notChanged++;
    }
    if (notChanged > maxIterations * 0.1) {
      converge = true;
    }
    if (converge) {
      std::cout << "bestEnergy: " << bestEnergy << std::endl;
      return bestAtomConfig;
    }
  }
  std::cout << "bestEnergy: " << bestEnergy << std::endl;
  return bestAtomConfig;
}
