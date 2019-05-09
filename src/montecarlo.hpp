#pragma once

// Run the Metropolis algorithm and return the best atom configuration
arma::uvec monteCarloBestShuffle(int N, int M, double xA, double beta, int &iterations, int maxIterations, double r);

// Calculate enthalpy changes for all x_A values and iteration counters for the
// different x_A values and return them.
arma::vec getEnthalpyChanges(int N, int M, double beta, int maxIterations, arma::uvec &iterationCount, double r);
