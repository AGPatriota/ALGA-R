# AUTHOR: Alexandre Galvão Patriota
# IME-USP

options(scipen=999)
#require('stringdist')
#require('stringr')
require('torch')
source('config.R')
source('GPT.R')
source('Generators.R')
library(gmp)


#Change here to set the number of digits
config$digits = 100
n0 = config$digits

#Print the results?
Print=TRUE
#comment for testing with different numbers 
set.seed(10)

#NUmbers
x = paste0(sample(0:9,n0, replace=TRUE), collapse="")
y = paste0(sample(0:9,n0, replace=TRUE), collapse="")



model_save = "Model.pt"

#Vocabulary
Voc = c("P", "S", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "\n", "C")

device0= if (torch::cuda_is_available()) "cuda" else "cpu"

#MODEL
model = GPT(block_size = config$block_size,
            block_size_out = config$block_size_out,
            n_embd = config$n_embd, 
            N_Layers = config$N_Layers,
            Head = config$n_head,
            nvoc = length(Voc),p0 =config$p0, p1=config$p1
)


if (file.exists(model_save)){
	model$load_state_dict(state_dict = torch::torch_load(model_save),  .refer_to_state_dict = TRUE )
}

model  = model$to(device=device0)


x0 = as.bigz(ifelse(as.numeric(x)==0, x, sub("^0+", "",x)))
y0 = as.bigz(ifelse(as.numeric(y)==0, y, sub("^0+", "",y)))
real = x0+y0
num = Intercalar(paste(x,"+", y, sep=""))
temp = list()
num0 =  strsplit(num, "")[[1]]
N = abs(2-nchar(num))
num1 = c(num0[(length(num0)-2+1):length(num0)],rep("P", config$block_size-2))
num1 =  c(Encoder(num1))
temp[[1]] = Generate(num1,Model=model,block_size_out=config$block_size_out,max_new_tokens=config$max, print=FALSE)
s=0
if(N>=2){
for(l in (1:(N/2))*2) {
	s = s+1		
	num1 = num0[(length(num0)-l-1):(length(num0)-l)]
	num1 = c(temp[[s]][-1],14, Encoder(num1))
	nn = length(num1)
	num1 = c(num1,rep(1, config$block_size-nn))
	temp[[s+1]] = Generate(num1,Model=model,block_size_out=config$block_size_out,max_new_tokens=config$max, print=FALSE)
	cli::cli_progress_message(paste(" Digit #",s+1, " ",  sep=""))
if(l%%100000==0){
	gc()
	gc()
}
}
}
aux = function(s) temp[[s]][length(temp[[s]])]
ind = length(temp):1
a = temp[[length(temp)]]
a = a[length(a)-1]
if(a %in% c(1,2,3,13,14)){
	pred = paste(Decoder(c(sapply(ind,aux))), collapse="")
} else{
	pred = paste(Decoder(c(a,sapply(ind,aux))), collapse="")
}
num1 = as.numeric(unlist(strsplit(as.character(num), "")))
pred = ifelse(pred==0, pred, sub("^0+", "", pred))
cat("\n")
print(paste("The output is ",real==pred, sep=""))
if(Print){
	print(paste("x=", x, sep=""))
	print(paste("y=", y, sep=""))
	print(paste("Real x+y=", real, sep=""))
	print(paste("Pred x+y=", pred, sep=""))
}
