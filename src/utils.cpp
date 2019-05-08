#include <iostream>
#include <cmath>
#include <armadillo>
#include <chrono>
#include <omp.h>

#include "montecarlo.hpp"

// Generate a randomly shuffled list of all atoms in the system corresponding to
// a given x_A. Value 0 for A, 1 for B.
arma::uvec generateAtomTypeVec(int numAtoms, double xA) {
  const int numA = std::round(numAtoms * xA);
  arma::uvec atomType(numAtoms);
  int usedA = 0;

  for (int i = 0; i < numAtoms; i++) {
    if (usedA < numA) {
      usedA++;
      atomType(i) = 0;
    }
    else {
      atomType(i) = 1;
    }
  }
  return arma::shuffle(atomType);
}

// Swap two atoms in the atomType vector to create a small permutation.
arma::uvec permuteAtomTypeVec(const arma::uvec &atomType, const arma::uvec &neighbourVec) {
  arma::uvec atomTypePermuted = atomType;
  int index = arma::randi<int>(arma::distr_param(0, atomType.n_elem - 1));
  int swap_index = arma::randi<int>(arma::distr_param(0, 3));
  std::swap(atomTypePermuted(index), atomTypePermuted(neighbourVec(4*index + swap_index)));
  return atomTypePermuted;
}

// Holds value 0 for A and 1 for B
arma::uvec generateNeighbourVec(int N, int M) {
  const int numAtoms = N * M;
  const int numNeighbours = 4 * numAtoms;
  arma::uvec neighbourVec(numNeighbours, arma::fill::zeros);

  for (int i = 0; i < numAtoms; i++) {
    neighbourVec(4 * i + 0) = i - M; // Top
    neighbourVec(4 * i + 1) = i + 1; // Right
    neighbourVec(4 * i + 2) = i + M; // Bottom
    neighbourVec(4 * i + 3) = i - 1; // Left

    if ((i - M) < 0) { // Top loop-around
      neighbourVec(4 * i + 0) = i + numAtoms - M;
    }
    if ((i + 1) % M == 0) { // Right loop-around
      neighbourVec(4 * i + 1) = (i - M + 1);
    }
    if ((i + M) > numAtoms - 1) { // Bottom loop-around
      neighbourVec(4 * i + 2) = i % M;
    }
    if ((((i + M) - 1) % M) == M - 1) { // Left loop-around
      neighbourVec(4 * i + 3) = (i + M - 1);
    }
  }
  return neighbourVec;
}

// Return the potential between two atoms
double V(double r, int atom1, int atom2) {
  if (atom1 == 0 && atom2 == 0) {
    constexpr double epsilonAA = 5.0;
    constexpr double gammaAA = 0.80;
    constexpr double etaAA = 1.95;
    const double value = epsilonAA * (std::pow(gammaAA/r, 6)  - std::exp(-r/etaAA));
    return value; // V_AA
  }
  else if (atom1 == 1 && atom2 == 1) {
    constexpr double epsilonBB = 6.0;
    constexpr double gammaBB = 0.70;
    constexpr double etaBB = 0.70;
    const double value = epsilonBB * (std::pow(gammaBB/r, 6)  - std::exp(-r/etaBB));
    return value; // V_BB
  }
  else {
    constexpr double epsilonAB = 4.5;
    constexpr double gammaAB = 0.85;
    constexpr double etaAB = 1.90;
    const double value = epsilonAB * (std::pow(gammaAB/r, 6)  - std::exp(-r/etaAB));
    return value; //V_AB
  }
}

// Return the Hamiltonian of atom A or B
double H(int atom1) {
  if (atom1 == 0) {
    return 0.20; // H_A
  }
  else {
    return 0.25; // H_B
  }
}

// Set up the Hamiltonian matrix
arma::sp_mat generateHtot(int N, int M, double xA, const arma::uvec &atomType,
                          const arma::uvec &neighbourList, double r) {
  const int numAtoms = N * M;
  arma::sp_mat Htot(numAtoms, numAtoms);
  // #pragma omp parallel for
  for (int i = 0; i < numAtoms; i++) {
    const int currentAtom = atomType(i);
    Htot(i,                        i) = H(currentAtom); // H_n
    Htot(i, neighbourList(4 * i + 0)) = V(r, currentAtom, atomType(neighbourList(4 * i + 0))); // V_nn
    Htot(i, neighbourList(4 * i + 1)) = V(r, currentAtom, atomType(neighbourList(4 * i + 1))); //V_nn
    Htot(i, neighbourList(4 * i + 2)) = V(r, currentAtom, atomType(neighbourList(4 * i + 2))); //V_nn
    Htot(i, neighbourList(4 * i + 3)) = V(r, currentAtom, atomType(neighbourList(4 * i + 3))); //V_nn
  }
  return Htot;
}

// Solve the Hamiltonian system and find its eigenvalues
void solveSystem(arma::vec &eigvals, const arma::sp_mat &Htot) {
  if (Htot.is_symmetric()) {
    arma::eig_sym(eigvals, arma::mat(Htot));
  }
  else {
    throw std::logic_error("Hamiltonian is not symmetric! ");
  }
}

// Calculate the free energy of a system
double getFreeEnergy(double beta, const arma::vec &eigvals) {
  double freeEnergy = 0;
  // #pragma omp parallel for schedule(static) reduction(+:freeEnergy)
  for (int i = 0; i < eigvals.n_elem; i++) {
    freeEnergy += std::log(1.0 + std::exp(-beta * eigvals(i)));
  }
  freeEnergy *= -1.0/beta;
  return freeEnergy;
}

// Calculate the enthalpy of one system returned from a Monte Carlo-run.
double getEnthalpy(int N, int M, double beta, double xA, int &iterations, int maxIterations, double r) {
  const int numAtoms = N * M;
  arma::vec eigvalsA, eigvalsB, eigvalsBest;
  const arma::uvec atomTypesA = generateAtomTypeVec(numAtoms, 1.0);
  const arma::uvec atomTypesB = generateAtomTypeVec(numAtoms, 0.0);
  const arma::uvec neighbourList = generateNeighbourVec(N, M);
  const arma::sp_mat HtotA = generateHtot(N, M, 1.0, atomTypesA, neighbourList, r);
  const arma::sp_mat HtotB = generateHtot(N, M, 0.0, atomTypesB, neighbourList, r);
  solveSystem(eigvalsA, HtotA);
  solveSystem(eigvalsB, HtotB);
  const double fA = getFreeEnergy(beta, eigvalsA);
  const double fB = getFreeEnergy(beta, eigvalsB);

  const arma::uvec bestShuffle = monteCarloBestShuffle(N, M, xA, beta, iterations, maxIterations, r);
  const arma::sp_mat HtotBest = generateHtot(N, M, xA, bestShuffle, neighbourList, r);
  solveSystem(eigvalsBest, HtotBest);
  const double fBest = getFreeEnergy(beta, eigvalsBest);

  return (fBest - fA*xA - fB * (1.0 - xA));
}

// Calculate enthalpy changes for all x_A values and iteration counters for the
// different x_A values and return them.
arma::vec getEnthalpyChanges(int N, int M, double beta, int maxIterations, arma::uvec &iterationCount, double r) {
  const int points = N * M;
  const int averageIterations = 1;
  iterationCount.resize(points);
  iterationCount.fill(0);
  arma::vec enthalpys(points);
  arma::vec enthalpysAveraged(points, arma::fill::zeros);
  arma::vec xAs = arma::linspace(0, 1, points);
  for (int ii = 0; ii < averageIterations; ii++) {
    arma::uvec iterationCountLocal(points);
    #pragma omp parallel for
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
  std::cout << enthalpysAveraged << std::endl;
  return enthalpysAveraged;
}

// Calculate the Hamming distance between two atom configurations. Considered
// to use this for the convergence criterion, but it did not work out as I hoped.
int hammingDistance(const arma::uvec &atomState1, const arma::uvec &atomState2) {
  int dist = 0;
  if (atomState1.n_elem != atomState2.n_elem) {
    return std::numeric_limits<int>::max();
  }
  for (int i = 0; i <atomState1.n_elem; i++) {
    if (atomState1(i) != atomState2(i)) {
      dist++;
    }
  }
  return dist;
}
