//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

mat compute_utility (const vec &P,
                     const mat &X,
                     const double &alpha,
                     const mat &beta_i) {
  
  int I = beta_i.n_rows;
  int N = X.n_rows;
  int K = X.n_cols;
  
  mat out(I, N);
  for (int n = 0; n < N; n++) {
    double price_part = alpha * P[n]; // I vector
    mat x_part(I, K);
    for (int k = 0; k < K; k++) {
      x_part.col(k) = beta_i.col(k) * X(n, k);
    }
    out.col(n) = sum(x_part, 1) - price_part;
  }
  return out;
}

mat logit_choice (const mat &U,
                  const vec &market_id) {
  
  // U -> (I x N) matrix
  // market_id -> N vector
  //
  // OUTPUT -> I x N matrix
  
  // Logit choice
  int I = U.n_rows;
  int T = max(market_id);
  mat out(I, U.n_cols);
  for (int t = 0; t < T; t++) {
    uvec t_index = find(market_id == t + 1);
    mat U_t = exp(U.cols(t_index));
    out.cols(t_index) = U_t.each_col() / (1 + sum(U_t, 1));
  }
  
  return out;
}

//[[Rcpp::export]]
vec calc_share (const vec &P,
                const mat &X,
                const double &alpha,
                const mat &beta_i,
                const vec &market_id) {
  
  // OUTPUT -> N vector
  
  mat U = compute_utility(P, X, alpha, beta_i);
  mat logit_p = logit_choice(U, market_id);
  rowvec share = mean(logit_p, 0);
  
  return share.t();
}