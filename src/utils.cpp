#include <iostream>
#include <cmath>
#include <armadillo>
#include <chrono>
#include <omp.h>

// value 0 for A, 1 for B
arma::vec generateAtomTypeVec(int numAtoms, double xA) {
  int numA = numAtoms * xA;
  arma::vec atomType(numAtoms);
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

// Holds value 0 for A and 1 for B
arma::vec generateNeighbourVec(int N, int M) {
  int numAtoms = N * M;
  int numNeighbours = 4 * numAtoms;
  arma::vec neighbourVec(numNeighbours, arma::fill::zeros);

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

double V(double r, int atom1, int atom2) {
  if (atom1 == 0 & atom2 == 0) {
    constexpr double epsilonAA = 5.0;
    constexpr double gammaAA = 0.80;
    constexpr double etaAA = 1.95;
    const double value = epsilonAA * (std::pow(gammaAA/r, 6)  - std::exp(-r/etaAA));
    return value; // V_AA
  }
  else if (atom1 == 1 & atom2 == 1) {
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

double H(int atom1) {
  if (atom1 == 0) {
    return 0.20; // H_A
  }
  else {
    return 0.25; // H_B
  }
}

arma::sp_mat generateHtot(int N, int M, const arma::vec &atomType,
                          const arma::vec &neighbourList) {
  int numAtoms = N * M;
  double r = 1.3;
  arma::sp_mat Htot(numAtoms, numAtoms);

  for (int i = 0; i < numAtoms; i++) {
    const int currentAtom = atomType(i);
    Htot(i,                    i) = H(currentAtom); // H_n
    Htot(neighbourList(i + 0), i) = V(r, currentAtom, atomType(neighbourList(i + 0))); // V_nn
    Htot(neighbourList(i + 1), i) = V(r, currentAtom, atomType(neighbourList(i + 1))); //V_nn
    Htot(neighbourList(i + 2), i) = V(r, currentAtom, atomType(neighbourList(i + 2))); //V_nn
    Htot(neighbourList(i + 3), i) = V(r, currentAtom, atomType(neighbourList(i + 3))); //V_nn
  }
  return Htot;
}
