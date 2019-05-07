#pragma once

arma::uvec generateNeighbourVec(int N, int M);

arma::uvec generateAtomTypeVec(int numAtoms, double xA);

arma::uvec permuteAtomTypeVec(const arma::uvec &atomType, const arma::uvec &neighbourVec);

arma::sp_mat generateHtot(int N, int M, double xA, const arma::uvec &atomType,
                          const arma::uvec &neighbourList);

void solveSystem(arma::vec &eigvals, const arma::sp_mat &Htot);

double getFreeEnergy(double beta, const arma::vec &eigvals);

double getEnthalpy(int N, int M, double beta, double xA, int &iterations, int maxIterations);

arma::vec getEnthalpyChanges(int N, int M, double beta, int maxIterations, arma::uvec &iterationCount);

int hammingDistance(const arma::uvec &atomState1, const arma::uvec &atomState2);
