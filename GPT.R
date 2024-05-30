# AUTHOR: Alexandre Galvão Patriota
# IME-USP

GPT <- torch::nn_module(
  initialize = function(block_size,block_size_out,n_embd, N_Layers, nvoc, Head, p0 , p1) {
    self$N <- N_Layers
    self$block_size <- block_size
    self$block_size_out <- block_size_out
    self$wpe_out <- torch::nn_embedding(block_size+block_size_out, n_embd)
    self$wte_out <- torch::nn_embedding(nvoc, n_embd, padding_idx = 1)
    self$MH_dec <- torch::nn_module_list(lapply(
      1:N_Layers,
      function(x) torch::nn_multihead_attention(n_embd, Head, dropout = p1)
    ))
    self$scale1_dec <- torch::nn_module_list(lapply(
      1:N_Layers,
      function(x) torch::nn_layer_norm(n_embd)
    ))
    self$scale2_dec <- torch::nn_module_list(lapply(
      1:N_Layers,
      function(x) torch::nn_layer_norm(n_embd)
    ))
    self$scale3 <- torch::nn_layer_norm(n_embd, elementwise_affine = TRUE)
    self$FFN_dec <- torch::nn_module_list(lapply(
      1:N_Layers,
      function(x) {
        torch::nn_sequential(
          torch::nn_linear(n_embd, 4 * n_embd),
          torch::nn_gelu(),
          torch::nn_linear(4 * n_embd, n_embd),
          torch::nn_dropout(p0)
        )
      }
    ))
    self$ln_f <- torch::nn_linear(n_embd, nvoc, bias = FALSE)
    self$drop0_out <- torch::nn_dropout(p = p0)
  },
  forward = function(x,y) {
    y1 <- torch::torch_arange(1, x$size(2) + y$size(2), dtype = torch::torch_int(),device = y$device)$unsqueeze(1)
    wei <- matrix(0, x$size(2) + y$size(2), x$size(2) + y$size(2))
    aux = (x$size(2)+1):( x$size(2) + y$size(2))
    wei[aux,aux ][upper.tri(wei[aux, aux])] = 1
    wei[1:x$size(2),aux[-1]] = 1
    wei = torch::torch_tensor(wei, dtype = torch::torch_bool(), device = x$device)
    output <- torch_cat(list(x, y),2)
    output <- self$wte_out(output) + self$wpe_out(y1)
    output <- self$drop0_out(output)
    
    for (j in 1:self$N) {
      Q <- torch::torch_transpose(self$scale1_dec[[j]](output), 1, 2)
      output <- output + torch::torch_transpose(self$MH_dec[[j]](Q, Q, Q, attn_mask = wei,need_weights = FALSE)[[1]], 1, 2)
      output <- output + self$FFN_dec[[j]](self$scale2_dec[[j]](output))
    }
    output <- self$ln_f(self$scale3(output))
    output[, (output$size(2)-y$size(2)+1):output$size(2),]
  }
)
