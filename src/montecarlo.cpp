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
  arma::uvec bestAtomConfig, worstAtomConfig;
  double bestEnergy = std::numeric_limits<double>::max();
  double worstEnergy = -std::numeric_limits<double>::max();
  int notChanged = 0;
  // Generate a good initial state from random shuffling
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
    //if (freeEnergy > worstEnergy) {
    //  worstEnergy = freeEnergy;
    //  worstAtomConfig = atomType;
    //}
  }
  std::cout << "Best energy from random generation: " << bestEnergy << std::endl;
  atomType = bestAtomConfig;
  arma::uvec nextUse = atomType;
  double lastEnergy = bestEnergy;
  // Start swapping atoms in the best initial state to find the best state
  for (int i = 0; i < maxIterations; i++) {
    iterations++;
    atomType = permuteAtomTypeVec(nextUse, neighbourList);
    arma::sp_mat Htot = generateHtot(N, M, xA, atomType, neighbourList, r);
    arma::vec eigvals;
    solveSystem(eigvals, Htot);
    double freeEnergy = getFreeEnergy(beta, eigvals);
    //if (freeEnergy > worstEnergy) {
    //  worstEnergy = freeEnergy;
    //  worstAtomConfig = atomType;
    //}
    double deltaEnergy = freeEnergy - lastEnergy;
    if (deltaEnergy < 0.0) {
      if (freeEnergy < bestEnergy) {
        bestEnergy = freeEnergy;
        bestAtomConfig = atomType;
        notChanged = 0;
      }
      else {
        notChanged++;
      }
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
    if (notChanged > maxIterations * 0.2) {
      std::cout << "Converge! " << std::endl;
      converge = true;
    }
    if (converge) {
      std::cout << "Best energy after flipping: " << bestEnergy << std::endl;
      std::cout << "Iterations run: " << iterations << std::endl;
      return bestAtomConfig;
    }
  }
  std::cout << "Best energy after flipping: " << bestEnergy << std::endl;
  std::cout << "Iterations run: " << iterations << std::endl;
  return bestAtomConfig;
}
