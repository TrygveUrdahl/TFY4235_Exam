#include <iostream>
#include <cmath>
#include <armadillo>
#include <chrono>
#include <omp.h>

#include "utils.hpp"

// Run the Metropolis algorithm and return the best atom configuration.
arma::uvec monteCarloBestShuffle(int N, int M, double xA, double beta, int &iterations, int maxIterations, double r) {
  const double numAtoms = N * M;
  const arma::uvec neighbourList = generateNeighbourVec(N, M);
  arma::uvec atomType = generateAtomTypeVec(numAtoms, xA);
  bool converge = false;
  iterations = 0;
  arma::uvec bestAtomConfig, worstAtomConfig;
  double bestEnergy = std::numeric_limits<double>::max();
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
  }
  std::cout << "\tBest energy from random generation: " << bestEnergy << std::endl;
  atomType = bestAtomConfig;
  arma::uvec nextUse = atomType;
  double lastEnergy = bestEnergy;
  // Start swapping atoms in the best initial state to find the best state.
  for (int i = 0; i < maxIterations; i++) {
    iterations++;
    atomType = permuteAtomTypeVec(nextUse, neighbourList);
    arma::sp_mat Htot = generateHtot(N, M, xA, atomType, neighbourList, r);
    arma::vec eigvals;
    solveSystem(eigvals, Htot);
    double freeEnergy = getFreeEnergy(beta, eigvals);
    double deltaEnergy = freeEnergy - lastEnergy;
    if (deltaEnergy < 0.0) {
      if (freeEnergy < bestEnergy) {
        // New best configuration found!
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
      // Maintain ergodic hypothesis.
      nextUse = atomType;
      lastEnergy = freeEnergy;
      notChanged++;
    }
    else {
      notChanged++;
    }
    if (notChanged > maxIterations * 0.2) {
      std::cout << "\tConvergence criterion reached. " << std::endl;
      converge = true;
    }
    if (converge) {
      std::cout << "\tBest energy after flipping: " << bestEnergy << std::endl;
      std::cout << "\tIterations run: " << iterations << std::endl;
      return bestAtomConfig;
    }
  }
  std::cout << "\tBest energy after flipping: " << bestEnergy << std::endl;
  std::cout << "\tIterations run: " << iterations << std::endl;
  return bestAtomConfig;
}


// Calculate the enthalpy $\Delta F$ of a system returned from a Monte Carlo
// simulation. Could have been made shorter, but this works well.
double getEnthalpy(int N, int M, double beta, double xA, int &iterations, int maxIterations, double r) {
  const int numAtoms = N * M;
  // Calculate fA and fB for enthalpy
  arma::vec eigvalsA, eigvalsB, eigvalsBest;
  static const arma::uvec atomTypesA = generateAtomTypeVec(numAtoms, 1.0);
  static const arma::uvec atomTypesB = generateAtomTypeVec(numAtoms, 0.0);
  static const arma::uvec neighbourList = generateNeighbourVec(N, M);
  static const arma::sp_mat HtotA = generateHtot(N, M, 1.0, atomTypesA, neighbourList, r);
  static const arma::sp_mat HtotB = generateHtot(N, M, 0.0, atomTypesB, neighbourList, r);
  solveSystem(eigvalsA, HtotA);
  solveSystem(eigvalsB, HtotB);
  static const double fA = getFreeEnergy(beta, eigvalsA);
  static const double fB = getFreeEnergy(beta, eigvalsB);

  // Get the best (or close to best) F for a system
  const arma::uvec bestShuffle = monteCarloBestShuffle(N, M, xA, beta, iterations, maxIterations, r);
  const arma::sp_mat HtotBest = generateHtot(N, M, xA, bestShuffle, neighbourList, r);
  solveSystem(eigvalsBest, HtotBest);
  const double fBest = getFreeEnergy(beta, eigvalsBest);

  return (fBest - fA*xA - fB * (1.0 - xA));
}

// Calculate enthalpy changes for all x_A values and iteration counters for the
// different x_A values and return them. Also changes the variable
// iterationCount as it is passed by reference.
arma::vec getEnthalpyChanges(int N, int M, double beta, int maxIterations, arma::uvec &iterationCount, double r) {
  const int points = N * M;
  const int averageIterations = 1; // Change if averageing is wanted
  iterationCount.resize(points);
  iterationCount.fill(0);
  arma::vec enthalpys(points);
  arma::vec enthalpysAveraged(points, arma::fill::zeros);
  arma::vec xAs = arma::linspace(0, 1, points);
  for (int ii = 0; ii < averageIterations; ii++) {
    arma::uvec iterationCountLocal(points);
    #pragma omp parallel for // Main source of parallelization of the program
    for (int i = 0; i < points; i++) {
      int iterations = 0;
      enthalpys(i) = getEnthalpy(N, M, beta, xAs(i), iterations, maxIterations, r);
      iterationCountLocal(i) = iterations;
    }
    iterationCount += iterationCountLocal;
    enthalpysAveraged += enthalpys;
  }
  iterationCount /= averageIterations;
  enthalpysAveraged /= averageIterations;
  return enthalpysAveraged;
}
