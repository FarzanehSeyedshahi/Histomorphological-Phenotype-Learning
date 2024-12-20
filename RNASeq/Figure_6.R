

library(biomaRt)
library(GSVA)
library(matrixStats)
library(readr)
library(readxl)
library(ComplexHeatmap)
library(MCPcounter)
library(circlize)
#normalized TCGA MESO expression was retrieved from UCSC (https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Mesothelioma%20(MESO)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)
norm_expr_MESO <- read.delim("TCGA-MESO.htseq_fpkm-uq.tsv")

#load HPC annotation of TCGA samples
leiden2p0fold0_v2 <- read.csv("TCGA_MESO_clr_leiden_2p0_fold4_WSI.csv")

#load HPC annotations
hpc_labels <- read.csv("hpc_labels.csv")
#load superclusters annotation
Supercluster_v3=read_xlsx("Superclusters v3.xlsx")
#converting ensembl gene ID to symbol
mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
genes <- sapply(norm_expr_MESO$Ensembl_ID,function(x){substr(x,1,15)})
G_list <- getBM(filters= "ensembl_gene_id", attributes= c("ensembl_gene_id",
                                                          "hgnc_symbol", "description"),values=genes,mart= mart)

G_list=G_list[!duplicated(G_list$ensembl_gene_id), ]
rownames(G_list)=G_list$ensembl_gene_id
norm_expr_MESO$Ensembl_ID=genes
norm_expr_MESO=norm_expr_MESO[norm_expr_MESO$Ensembl_ID%in%G_list$ensembl_gene_id, ]


#removing genes detected in less than a third of samples
rss=rowSums(norm_expr_MESO[,2:87]>0)
filtered_norm_expr_MESO=norm_expr_MESO[rss>86/3, ]

rownames(filtered_norm_expr_MESO)=filtered_norm_expr_MESO$Ensembl_ID
filtered_norm_expr_MESO=filtered_norm_expr_MESO[,c(-1)]
#aggregating expression of genes with same symbol ID
filtered_norm_expr_MESO$Symbol=G_list[rownames(filtered_norm_expr_MESO),"hgnc_symbol"]
final_expr_MESO=aggregate(filtered_norm_expr_MESO[,1:86],by=list(filtered_norm_expr_MESO$Symbol),FUN = sum)

final_expr_MESO=final_expr_MESO[c(-1),]

rownames(final_expr_MESO)=final_expr_MESO$Group.1
final_expr_MESO=final_expr_MESO[,c(-1)]

#calculating ssgsea score for signature of HALLMARKS database version 7.4 retrieved from MSIGDB website 
#and KEGG human database download from enrichR page (https://maayanlab.cloud/Enrichr/#libraries)
Paths=loadDB(c("h.all.v7.4.symbols.gmt","KEGG_2021_Human.txt"))
gsva.es <- gsva(param=ssgseaParam(as.matrix(final_expr_MESO),
                                  Paths,
                                  minSize=10,
                                  normalize = T), verbose=FALSE)


TCGA_MESO_cluster_clr <- leiden2p0fold0_v2

#restricting gene expression matrix and WSI HPC annotation to the samples that are included in both
short_expr_ID=gsub("\\.","-",substr(colnames(final_expr_MESO),1,12))
short_img_ID=substr(TCGA_MESO_cluster_clr$slides,1,12)
colnames(final_expr_MESO)=short_expr_ID
colnames(gsva.es)=short_expr_ID
final_gsva_mat=gsva.es[,(short_img_ID[short_img_ID%in%short_expr_ID])]
matrix_hpc=TCGA_MESO_cluster_clr[short_img_ID%in%short_expr_ID,3:49]
matrix_hpc=as.data.frame(matrix_hpc)
matrix_hpc_ID=substr(unlist(TCGA_MESO_cluster_clr[short_img_ID%in%short_expr_ID,2]),1,12)

#calulating correlation coefficient and p-value between hpc composition and pathways ssGSEA score
cluster_gsva_corr=matrix(NA,ncol=ncol(matrix_hpc),nrow = nrow(final_gsva_mat))
for(i in 1:ncol(matrix_hpc)){
  for(j in 1:nrow(final_gsva_mat)){
    cluster_gsva_corr[j,i]=cor.test(matrix_hpc[,i],final_gsva_mat[j,])$estimate
  }
}

cluster_gsva_corr=as.data.frame(cluster_gsva_corr)
colnames(cluster_gsva_corr)=colnames(matrix_hpc)
rownames(cluster_gsva_corr)=rownames(final_gsva_mat)

cluster_gsva_pval=matrix(NA,ncol=ncol(matrix_hpc),nrow = nrow(final_gsva_mat))
for(i in 1:ncol(matrix_hpc)){
  for(j in 1:nrow(final_gsva_mat)){
    cluster_gsva_pval[j,i]=cor.test(matrix_hpc[,i],final_gsva_mat[j,])$p.value
  }
}


cluster_gsva_pval=as.data.frame(cluster_gsva_pval)
colnames(cluster_gsva_pval)=colnames(matrix_hpc)
rownames(cluster_gsva_pval)=rownames(final_gsva_mat)

#restricting correlation coefficient matrix to the ones with a p-value <= 0.01
cluster_gsva_corrsignif=cluster_gsva_corr
for(i in 1:ncol(cluster_gsva_pval)){
  for(j in 1:nrow(cluster_gsva_pval)){
    if(cluster_gsva_pval[j,i]>0.01){
      cluster_gsva_corrsignif[j,i]=0
    }
  }
}
dim(cluster_gsva_corrsignif[rowSums(cluster_gsva_corrsignif)>0,])
cluster_gsva_corrsignif=cluster_gsva_corrsignif[rowSums(cluster_gsva_corrsignif)>0.3,]
dim(cluster_gsva_corrsignif)
colnames(cluster_gsva_corrsignif)=Book2$short_annot
rownames(cluster_gsva_corrsignif)=sapply(rownames(cluster_gsva_corrsignif), function(x){strsplit(x,"~")[[1]][2]})


#######Heatmaps annotations and legends
col_fun = colorRamp2(c(-0.5, 0, 0.5), c("#67a9cf", "white", "#ef8a62"))
col_fun_sub = colorRamp2(c(-0.5, 0, 0.5), c("#d8b365", "white", "#5ab4ac"))
col_fun_surv=colorRamp2(c(0.9,1,1.1), c("#af8dc3", "white", "#7fbf7b"))
col_fun_inflammation=colorRamp2(c(0,300), c("white", "#998ec3"))

sign_OR=ifelse(hpc_labels$P_value_subtype<0.05,"*",NA)
sign_surv=ifelse(hpc_labels$P_value_survival<0.05,"*",NA)




hc=ComplexHeatmap::HeatmapAnnotation(Supercluster=Supercluster_v3$Cluster,
                                     Survival = anno_simple(hpc_labels$HR_survival, col = col_fun_surv, pch = sign_surv),
                                     Subtype = anno_simple(hpc_labels$LogOR_subtype, col = col_fun_sub, pch = sign_OR),
                                     Inflammation = anno_simple(hpc_labels$Inflammation, col = col_fun_inflammation),
                                     col = list("Supercluster"= c("Blood" = "#E15759","Complex nested epithelioid"="#F28E2B",
                                                                  "Collagen"="#A0CBE8", "Connective tissue"="#499894",
                                                                  "Inflamed" = "#9D7660", "Inflamed malignancy"="#8CD17D",
                                                                  "Lung"="#FFBE7D", "Muscle"="#FF9D9A","Necrosis" = "#BAB0AC",
                                                                  "Nested epithelioid"="#B07AA1", "Nested epithelioid, cold"="#F1CE63",
                                                                  "Packed malignant cells"="#4E79A7", "Packed spindle cells"="#59A14F",
                                                                  "Solid"="#D37295", "Spindle cells and collagen"="#79706E",
                                                                  "Talc"="#D7B5A6", "vessels"="#FABFD2")))

lgd_survival = Legend(title = "Survival: HR", col_fun = col_fun_surv, at = c(0.9, 1, 1.1))
lgd_subtype = Legend(title = "Subtype: logOR", col_fun = col_fun_sub, at = c(-0.5, 0, 0.5))
lgd_inflammation = Legend(title = "Inflammation", col_fun = col_fun_inflammation, at = c(0, 175, 350, 525, 700))
# and one for the significant p-values
lgd_sig = Legend(pch = "*", type = "points", labels = "< 0.05")


#####MCP counter and Ki67 (a)
#Ki67 correlation
ki67corr_estimate=sapply(1:ncol(matrix_hpc),function(i){
  cor.test(matrix_hpc[,i],as.numeric(final_expr_MESO["MKI67",(short_img_ID[short_img_ID%in%short_expr_ID])]))$estimate})

ki67corr_pval=sapply(1:ncol(matrix_hpc),function(i){
  cor.test(matrix_hpc[,i],as.numeric(final_expr_MESO["MKI67",(short_img_ID[short_img_ID%in%short_expr_ID])]))$p.value})

ki67corr_estimate_signif=ki67corr_estimate
ki67corr_estimate_signif[ki67corr_pval>0.05]=0

#MCPcounter estimation and correlation
mcp_patient=MCPcounter.estimate(final_expr_MESO,featuresType = "HUGO_symbols")

final_mcp_mat=mcp_patient[,(short_img_ID[short_img_ID%in%short_expr_ID])]

cluster_MCP_corr=matrix(NA,ncol=ncol(matrix_hpc),nrow = nrow(final_mcp_mat))
for(i in 1:ncol(matrix_hpc)){
  for(j in 1:nrow(final_mcp_mat)){
    cluster_MCP_corr[j,i]=cor.test(matrix_hpc[,i],final_mcp_mat[j,])$estimate
  }
}


cluster_MCP_corr=as.data.frame(cluster_MCP_corr)
colnames(cluster_MCP_corr)=colnames(matrix_hpc)
rownames(cluster_MCP_corr)=rownames(final_mcp_mat)

cluster_MCP_pval=matrix(NA,ncol=ncol(matrix_hpc),nrow = nrow(final_mcp_mat))
for(i in 1:ncol(matrix_hpc)){
  for(j in 1:nrow(final_mcp_mat)){
    cluster_MCP_pval[j,i]=cor.test(matrix_hpc[,i],final_mcp_mat[j,])$p.value
  }
}

cluster_MCP_corrsignif=cluster_MCP_corr
for(i in 1:ncol(cluster_MCP_pval)){
  for(j in 1:nrow(cluster_MCP_pval)){
    if(cluster_MCP_pval[j,i]>0.05){
      cluster_MCP_corrsignif[j,i]=0
    }
  }
}



matrixheatmap=rbind(cluster_MCP_corrsignif,ki67corr_estimate_signif)
colnames(matrixheatmap)=as.character(0:46)
ht1=Heatmap(matrixheatmap,top_annotation = hc, name="MCPCounter and Ki67 correlation",
            row_labels = c(rownames(cluster_MCP_corrsignif), "Ki67"), col = col_fun,
            rect_gp = gpar(col = "darkgrey", lwd = 1),cluster_row_slices = F, column_names_rot = 360,
            column_names_centered = T, row_split = c(rep("MCP",times=10),"KI67"),
            row_names_max_width = max_text_width(
              rownames(matrixheatmap), 
              gp = gpar(fontsize = 12)
            ))
draw(ht1,heatmap_legend_side="left",annotation_legend_side="left",
     annotation_legend_list = list(lgd_subtype, lgd_survival, lgd_inflammation,
                                   lgd_sig), merge_legend = T)


#######KEGG heatmap (b)
kegg.cluster_gsva_corr=cluster_gsva_corr[grep("KEGG",rownames(cluster_gsva_corr)), ]
rownames(kegg.cluster_gsva_corr)=sapply(rownames(kegg.cluster_gsva_corr),function(x){strsplit(x = x, split = "~",fixed = T)[[1]][2]})
kegg.cluster_gsva_pval=cluster_gsva_pval[grep("KEGG",rownames(cluster_gsva_pval)), ]
rownames(kegg.cluster_gsva_pval)=sapply(rownames(kegg.cluster_gsva_pval),function(x){strsplit(x = x, split = "~",fixed = T)[[1]][2]})
for(i in 1:ncol(kegg.cluster_gsva_pval)){
  for(j in 1:nrow(kegg.cluster_gsva_pval)){
    if(kegg.cluster_gsva_pval[j,i]>0.05){
      kegg.cluster_gsva_corr[j,i]=0
    }
  }
}
kegg.cluster_gsva_corr=kegg.cluster_gsva_corr[rowSums(kegg.cluster_gsva_corr)>0,]


kegg_paths=c("Focal adhesion", "Cell adhesion molecules", "cGMP-PKG signaling pathway",
             "Rap1 signaling pathway", "Pathways in cancer", "Ras signaling pathway",
             "Leukocyte transendothelial migration", "MAPK signaling pathway", "PI3K-Akt signaling pathway",
             "Notch signaling pathway", "T cell receptor signaling pathway")
matrixheatmap=kegg.cluster_gsva_corr[kegg_paths,]
colnames(matrixheatmap)=as.character(0:46)
ht1=Heatmap(matrixheatmap,top_annotation = hc, name="KEGG correlation",
            row_labels = kegg_paths, col = col_fun, rect_gp = gpar(col = "darkgrey", lwd = 1),
            cluster_row_slices = F, column_names_rot = 360, column_names_centered = T,
            row_names_max_width = max_text_width(
              rownames(matrixheatmap), 
              gp = gpar(fontsize = 12)
            ))
draw(ht1,heatmap_legend_side="left",annotation_legend_side="left",
     annotation_legend_list = list(lgd_subtype, lgd_survival, lgd_inflammation, lgd_sig), merge_legend = T)


#######HALLMARKS (c)
hk.cluster_gsva_corr=cluster_gsva_corr[grep("h.all",rownames(cluster_gsva_corr)), ]
rownames(hk.cluster_gsva_corr)=sapply(rownames(hk.cluster_gsva_corr),function(x){substring(strsplit(x = x, split = "~",fixed = T)[[1]][2],10)})
hk.cluster_gsva_pval=cluster_gsva_pval[grep("h.all",rownames(cluster_gsva_pval)), ]
rownames(hk.cluster_gsva_pval)=sapply(rownames(hk.cluster_gsva_pval),function(x){substring(strsplit(x = x, split = "~",fixed = T)[[1]][2],10)})
for(i in 1:ncol(hk.cluster_gsva_pval)){
  for(j in 1:nrow(hk.cluster_gsva_pval)){
    if(hk.cluster_gsva_pval[j,i]>0.05){
      hk.cluster_gsva_corr[j,i]=0
    }
  }
}
hallmarks_paths=c("MTORC1_SIGNALING", "MITOTIC_SPINDLE", "E2F_TARGETS",
                  "G2M_CHECKPOINT", "KRAS_SIGNALING_UP", "NOTCH_SIGNALING",
                  "EPITHELIAL_MESENCHYMAL_TRANSITION", "APICAL_JUNCTION",
                  "TGF_BETA_SIGNALING", "HEDGEHOG_SIGNALING", "GLYCOLYSIS",
                  "OXIDATIVE_PHOSPHORYLATION", "ANGIOGENESIS")

matrixheatmap=hk.cluster_gsva_corr[hallmarks_paths,]
colnames(matrixheatmap)=as.character(0:46)
ht1=Heatmap(matrixheatmap,top_annotation = hc, name="HALLMARKS correlation",
            row_labels = hallmarks_paths, col = col_fun, rect_gp = gpar(col = "darkgrey", lwd = 1),
            cluster_row_slices = F, column_names_rot = 360, column_names_centered = T,
            row_names_max_width = max_text_width(
              rownames(matrixheatmap), 
              gp = gpar(fontsize = 12)
            ))
draw(ht1,heatmap_legend_side="left",annotation_legend_side="left",
     annotation_legend_list = list(lgd_subtype, lgd_survival, lgd_inflammation, lgd_sig), merge_legend = T)


#####################Utilities
loadDB<-function(file_list=listDB()){
  DB=NULL
  lengthfl=length(file_list)
  filetype=unlist(lapply(file_list, tools::file_ext))
  for (file in file_list[filetype=="" | filetype=="txt"]){
    # Read in the data
    x <- scan(file, what="", sep="\n")
    # Separate elements by one or more whitepace
    y <- strsplit(x, "\t+")
    # Extract the first vector element and set it as the list element name concatenated with file name separated by "~"
    if(lengthfl>1){
      names(y) <- paste(file,sapply(y, `[[`, 1),sep="~")
    }else{
      names(y) <- sapply(y, `[[`, 1)
    }
    # Remove the first vector element from each list element
    y <- lapply(y, `[`, -1)
    DB=c(DB,y)
  }
  
  for (file in file_list[filetype=="gmt"]){
    y <- qusage::read.gmt(file)
    if(lengthfl>1){
      names(y)<-paste0(file,"~",names(y))
    }
    DB=c(DB,y)
  }
  
  DB <- lapply(DB,function(z) sort(unique(setdiff(z,c(NA,"","---")))))
  DB <- DB[unique(names(DB))]
  
  invisible(DB)
  
}

