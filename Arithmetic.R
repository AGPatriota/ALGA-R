# AUTHOR: Alexandre Galvão Patriota
# IME-USP

options(scipen=999)
require('stringdist')
require('stringr')
require('torch')
source('config.R')
source('GPT.R')
source('Generators.R')

model_save = "Model.pt"
metrics_save= "Model.RData"

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

loss_save = NULL
if (file.exists(metrics_save)){
	load(metrics_save)
}

#set.seed(1)
test = list()
for(L in 1:10){
	test[[L]] = Gen()
}

set.seed(NULL)

loss0 = nn_cross_entropy_loss(reduction="none",ignore_index=0)
optimizer <- optim_adamw(model$parameters, lr = config$lr)
loss_store = 0

if(config$train){
#for(i in 1:config$epochs){
  for(j in 1:config$iter){
    Z = Gen()
    FIT = model$train()(Z$x, Z$y[,1:(Z$y$size(2)-1)]  )
    loss = loss0(FIT$flatten(end_dim = 2), Z$y[,2:Z$y$size(2)]$flatten(end_dim = -1))
    loss$mean()$backward()
    optimizer$step()
    optimizer$zero_grad()
    loss_store = ((j-1)* config$batch_size * loss_store + loss$mean()$item() * config$batch_size)/( j*config$batch_size )
    cli::cli_progress_message(paste(" Epoca: ",i, " ", " Iter: ",j, " ", "loss: ",loss_store*100, " loss Atual: ",  loss$mean()$item()*100, sep=""))
  }

torch::torch_save(model$parameters, model_save)

loss_store_eval = 0
with_no_grad({
for(k in 1:length(test)){
	aux = (1+ config$batch_size*(k-1)):( config$batch_size *k)
	FIT = model$train()(test[[k]]$x, test[[k]]$y[,1:(test[[k]]$y$size(2)-1)]  )
	loss_eval = loss0(FIT$flatten(end_dim = 2), test[[k]]$y[,2:test[[k]]$y$size(2)]$flatten(end_dim = -1))
	loss_store_eval = loss_store_eval + mean(loss_eval)/length(test)
}
})

xx = test[[1]]$x
for(k in 2:length(test)){
	xx = torch_cat(list(xx,test[[k]]$x), 1)
}
yy= test[[1]]$y
for(k in 2:length(test)){
	yy = torch_cat(list(yy,test[[k]]$y), 1)
}

w=LV = numeric()
for(k in 1:xx$size(1)){
	x = xx[k,]
        output = Generate(as.numeric(x$cpu()),Model=model,block_size_out=config$block_size_out,max_new_tokens=config$max ,print= F) 
	output = gsub("P", "",output)
	output = sub("S.*", "", paste(Decoder(output), collapse = ""))
	Real = sub("S.*", "",(paste(Decoder(as.numeric(yy[k,]$to(device="cpu"))), collapse = "")))
	Real = gsub("P", "", Real) 
	Real = substr(Real, 1,config$max+1)
	w[k] = output == Real
	LV[k] = sum(stringdist(output,Real, method = 'lv'))
	if(k%%20==0){
		cat("\nPrompt:")
		cat(paste(Decoder(as.numeric(xx[k,]$to(device="cpu"))), collapse = ""))
		cat("\nOutput:\n")
		cat(output)
		cat(Real)
	}
}
cat(paste("\nAccuracy: ", mean(w)*100, "%\n", sep=""))
#}
cat(paste("Eval loss: ",round(loss_store_eval$mean()$item()*100,4) , sep=""),"\n")

loss_save = rbind(loss_save, matrix(c(i,loss_store,loss_store_eval$mean()$item(),mean(w)*100,mean(LV)),nrow= 1))
save(loss_save,file=metrics_save)
cat(paste(" LV: ",mean(LV),  "\n", sep="")	)

colnames(loss_save)=c("epoch", "Train loss (%)", "Eval loss (%)", "Accuracy (%)", "Levenshtein (edit) distance")
jpeg("Train-Loss.jpg",width = 482*3.2*2, height = 480*1.5*2, quality = 100, res=170)
  par(mfrow=c(2,2))
  plot(loss_save[,c(1,2)], type="l", lwd =2)
  plot(loss_save[,c(1,3)], type="l", lwd =2)
  plot(loss_save[,c(1,4)], ylim=c(0,100), type="l", lwd =2)
  abline(h=100, lty=2)
  plot(loss_save[,c(1,5)], type="l", lwd =2)
dev.off()
}


############################################
###Testing after training config$train=FALSE
############################################

if(!config$train){
library(gmp)

Intercalar = function(input){
	num = strsplit(unlist(strsplit(input, "\\+")), "")
	na = length(num[[1]])
	nb = length(num[[2]])
	num[[1]] = c(rep("0", max(na,nb)-na), num[[1]])
	num[[2]] = c(rep("0", max(na,nb)-nb), num[[2]])
	return(paste(paste(num[[1]], num[[2]], sep=""), collapse=""))
}

ss = numeric()
n0 = config$digits
for(i in 1:1000){
	x = paste0(sample(0:9,n0, replace=TRUE), collapse="")
	y = paste0(sample(0:9,n0, replace=TRUE), collapse="")
	num = Intercalar(paste(x,"+", y, sep=""))
	temp=list()
	num0 =  strsplit(num, "")[[1]]
        N = abs(2-nchar(num))
	num1 = c(num0[(length(num0)-2+1):length(num0)],rep("P", config$block_size-2))
	num1 =  c(Encoder(num1))
	temp[[1]]= Generate(num1,Model=model,block_size_out=config$block_size_out,max_new_tokens=config$max, print=FALSE)
	s=0
	for(l in (1:(N/2))*2) {
		s=s+1		
		num1 = num0[(length(num0)-l-1):(length(num0)-l)]
		num1 = c(temp[[s]][-1],14, Encoder(num1))
		nn = length(num1)
		num1 = c(num1,rep(1, config$block_size-nn))
		temp[[s+1]]= Generate(num1,Model=model,block_size_out=config$block_size_out,max_new_tokens=config$max, print=FALSE)
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
	#print(paste(as.bigz(sub("^0+", "",x)) + as.bigz(sub("^0+", "",y))))
	#print(pred)

	ss[i] = paste(as.bigz(sub("^0+", "",x)) + as.bigz(sub("^0+", "",y)))==pred
	cli::cli_progress_message(paste(" Epoca: ",i, " ", " Positive: ", sum(ss), sep=""))
}
}
