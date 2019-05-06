#pragma once

arma::uvec generateNeighbourVec(int N, int M);

arma::uvec generateAtomTypeVec(int numAtoms, double xA);

arma::sp_mat generateHtot(int N, int M, double xA, const arma::uvec &atomType,
                          const arma::uvec &neighbourList);
void solveSystem(arma::vec &eigvals, const arma::sp_mat &Htot);

double getFreeEnergy(double beta, const arma::vec &eigvals);

double getEnthalpy(int N, int M, double beta, double xA, int iterations);

arma::vec getEnthalpyChanges(int N, int M, double beta, int iterations);
