beta <- 0.5 
rho <- 0.9 
B <- 10
M <- 6

S <- seq(0, B + M)

u <- function(x, beta=0.5) {
  return(x^beta)
}

phi <- function(z) {
  return(1/(B + 1))
}

Gamma <- function(x) {
  return(seq(0, min(x, M)))
}

dist <- function(v0, v1) {
  return(max(abs(v0 - v1)))
}

Tv <- function(v) {
  ret_val <- c()
  for (x in 0:(B + M)) {
    feasible_a <- Gamma(x)
    max_val <- -Inf
    for (a in feasible_a) {
      tmp_val <- u(x - a)
      for (z in 0:B) {
        tmp_val <- tmp_val + rho * v[a + z + 1] * phi(seq(0, z))
      }
      if (tmp_val > max_val)
        max_val <- tmp_val
    }
    ret_val[x + 1] <- max_val
  }
  return(ret_val)
}

compute_vstar <- function(v0, max_iter=1e3, tol=1e-5, verbose=TRUE) {
  success <- FALSE
  iter <- 0
  while (TRUE) {
    v1 <- Tv(v0)
    d <- dist(v0, v1)
    if (d < tol) {
      success <- TRUE
      break
    }
    if (iter > max_iter)
      break
    if (verbose & iter %% 10 == 0)
      print(paste0("Iter:", iter, ". Distance: ", d))
    iter = iter + 1
    v0 <- v1
  }
  ret_list <- list(v_final = v1, v_prev = v0, success = success, iter = iter)
  return(ret_list)
}



