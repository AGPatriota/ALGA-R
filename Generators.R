# AUTHOR: Alexandre Galvão Patriota
# IME-USP

#Encoding numbers to Tokens
Encoder = function(File = File0, Vocabulary = Voc){
  File = unlist(strsplit(File, ""))
  FileX = numeric(length(File))
  for(i in 1:length(Vocabulary)){
    FileX[File == Vocabulary[i]] <- i 
  }
  return(FileX)  
}

#Decoding tokens to numbers
Decoder = function(File = File1, Vocabulary = Voc){
  FileX = File
  for(i in 1:length(Vocabulary)){
    FileX[File == i] <- Vocabulary[i]
  }
  return(FileX)  
}


#Generating Tokens
Generate = function(idx, Model, block_size_out , max_new_tokens = 3, temperature=0.7, top_k = NULL, device00=device0, print = TRUE){
  if(print)
  cat("\n \n===================== Generating Tokens =====================\n \n")
  aa = Decoder(idx)
  aa = aa[aa != "P"]
  if(print)
  cat(paste0("Prompt:\n", paste(aa,collapse=""), "\n", collapse=""))
  idx = torch::torch_tensor(idx, dtype=torch::torch_int(), device=device00)
  idx = torch::torch_unsqueeze(idx, 1)
 
  if(print)
  cat(paste0("Output:\n", paste(Decoder(c(13)), collapse=""), collapse=""))
  y = torch::torch_tensor(c(13), dtype=torch::torch_int(), device=device00)
  y = y$unsqueeze(1)
  y0= 13
  torch::with_no_grad({
    for(i in 1:max_new_tokens){
      if(y$size(2) <= block_size_out){ 
        y_cond = y
      } else{
        k1=y$size(2)-block_size_out+1; k2 =y$size(2)
        y_cond = y[,k1:k2]}
      logits = Model$eval()(idx, y_cond)
      q = min(i, logits$size(2))
      logits = logits[,  q, ] / temperature
      if(!is.null(top_k)){
        logits = logits$topk(top_k)
        probs = torch::nnf_softmax(logits[[1]],-1)
        selected = torch::torch_multinomial(probs, num_samples=1)
        y_next <- logits[[2]][,selected$item()]$unsqueeze(1)
      }
      if(is.null(top_k)){
        y_next = torch::torch_max(logits, -1)[[2]]$unsqueeze(1)
      }
      if(y_next$item()==2) break
      
      y  = torch::torch_cat(list(y, y_next), 2)
      if(print)
      cat(Decoder(as.integer(y_next$cpu())))
    }
    if(print)
    cat("\n")
    y  = y$to(device = 'cpu')
    y  = as.integer(y)
    return(y)
  })
}


Intercalar = function(input){
	num = strsplit(unlist(strsplit(input, "\\+")), "")
	na = length(num[[1]])
	nb = length(num[[2]])
	num[[1]] = c(rep("0", max(na,nb)-na), num[[1]])
	num[[2]] = c(rep("0", max(na,nb)-nb), num[[2]])
	return(paste(paste(num[[1]], num[[2]], sep=""), collapse=""))
}


Add <- function(num1) {
  num1 = as.numeric(unlist(strsplit(as.character(num1), "")))
  soma = sum(num1)
  return( list(soma, ifelse(soma>=10, 1, 0)))
}

Add1 = function(num1, num2=NULL){
	if(is.null(num2)){
		y = Add(num1)[[1]]
	       	return(list(x = num1, y=paste(c("\n",y, "S"), collapse="")))
	}
	if(!is.null(num2)){
		if(nchar(num2)<2) num2 = paste(c(0,num2), collapse="")
			x <- Add(num1)
			num21= as.numeric(substr(num2,1,1))
			num22= as.numeric(substr(num2,2,2))
			return(list(x = paste(c(x[[1]], "C", num2), collapse=""), y = paste(c("\n",num21+num22+x[[2]][1], "S"), collapse="")))
	}
}

Gen = function(batch = config$batch_size) {
    x = vector("list", batch)
    y = vector("list", batch)
    for(l in 1:batch) {
        num1 = paste0(sample(0:9, 2, replace = TRUE), collapse = "")
	num2 = NULL
	if(runif(1)< 0.5) num2 = paste0(sample(0:9, 2, replace = TRUE), collapse = "")
        z = Add1(num1, num2)
        if(min(nchar(z$y)) < config$block_size_out) {
            num_chars_neededy = config$block_size_out - nchar(z$y)
            z$y = paste0(z$y, strrep("P", num_chars_neededy))
        }
        if(min(nchar(z$x)) < config$block_size) {
            num_chars_neededx = config$block_size - nchar(z$x)
            z$x = paste0(z$x, strrep("P", num_chars_neededx))
        }
        x[[l]] = Encoder(z$x)
        y[[l]] = Encoder(z$y)
    }
    x_tensor = torch::torch_tensor(do.call(rbind, x), dtype = torch::torch_int(), device = device0)
    y_tensor = torch::torch_tensor(do.call(rbind, y), dtype = torch::torch_int(), device = device0)
    return(list(x = x_tensor, y = y_tensor))
}
