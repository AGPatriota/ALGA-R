# AUTHOR: Alexandre Galvão Patriota
# IME-USP

config <- list(
  digits = 1000,
  train = FALSE, 
  iter=2000,
  block_size = 5,
  block_train = 5,
  block_size_out = 4, 
  max = 5,
  n_embd = 64, 
  N_Layers = 2,     
  n_head = 2,     
  lr =  0.0005,
  batch_size = 16*32,  
  epochs = 20,
  p0 = 0.2,         
  p1 = 0.2
)
