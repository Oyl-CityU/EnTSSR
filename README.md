EnTSSR: a weighted ensemble learning method to impute single-cell RNA sequencing data
===========

Contents of this archive
------------------------

(1) EnTSSR_adaptive_weight.py: the scripts of EnTSSR
(2) EnTSSR_conda_env.yml :environment profile

# Configuring the environment
Run the following code on the command line:
'''
conda env create -f EnTSSR_conda_env.yml
'''

# Step 1. Install the EnImpute package from GitHub.
'''
install.packages("devtools")
library("devtools")
install_github("Zhangxf-ccnu/EnImpute", subdir="pkg")
'''

# Step 2. Obtain base imputed results of various base imputation methods by EnImpute package, and save at directory: "./data/res/".
'''
datasets = c("baron","manno","chen","zeisel","guo2","pollen","iPSC","loh","usoskin","deng","chu")
methods =c("ALRA", "DCA", "MAGIC","SAVER", "scImpute", "scRMD")
for (d in data)
	count <- utils::readcsv(paste0(".\data\input\",d,"_exp_obs.csv"), header = TRUE, row.names = 1)
	count.imputed = EnImpute(count, ALRA = TRUE, DCA = TRUE, DrImpute = FALSE, MAGIC = TRUE, SAVER = TRUE,
                    scImpute = TRUE, scRMD = TRUE, Seurat = FALSE)
	exp <- count.imputed$count.imputed.individual
	res <- count.imputed$count.imputed.individual.rescaled
	for (n in 1:6){
		write.csv(exp[,,n],file = paste0(".\data\res\",d,"_exp_",methods[n],".csv"),row.names = F,quote = F)
		write.csv(res[,,n],file = paste0(".\data\res\",d,"_res_",methods[n],".csv"),row.names = F, quote = F)
	}
}
'''
# Step 3. Run EnTSSR model ,and find the imputed result at: "./data/res/xxx_exp_EnTSSR" .
'''
python EnTSSR_adaptive_weight.py
'''

# Data are collected from the following sources:
https://github.com/mohuangx/SAVER-paper/tree/master/SAVER-data
https://github.com/gongx030/scDatasets
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748