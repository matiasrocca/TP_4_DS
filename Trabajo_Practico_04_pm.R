library(xgboost)
library(dplyr)

rm(list = ls())

###func para feature engenieering
one_hot_sparse <- function(data_set) {

    require(Matrix)
    created <- FALSE

    if (sum(sapply(data_set, is.numeric)) > 0) {  # Si hay, Pasamos los numéricos a una matriz esparsa (sería raro que no estuviese, porque "Price"  es numérica y tiene que estar sí o sí)
        out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.numeric), drop = FALSE]), "dgCMatrix")
        created <- TRUE
    }

    if (sum(sapply(data_set, is.logical)) > 0) {  # Si hay, pasamos los lógicos a esparsa y lo unimos con la matriz anterior
        if (created) {
            out_put_data <- cbind2(out_put_data,
                                   as(as.matrix(data_set[,sapply(data_set, is.logical), drop = FALSE]), "dgCMatrix"))
        } else {
            out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.logical), drop = FALSE]), "dgCMatrix")
            created <- TRUE
        }
    }

    # Identificamos las columnas que son factor (OJO: el data.frame no debería tener character)
    fact_variables <- names(which(sapply(data_set, is.factor)))

    # Para cada columna factor hago one hot encoding
    i <- 0

    for (f_var in fact_variables) {

        f_col_names <- levels(data_set[[f_var]])
        f_col_names <- gsub(" ", ".", paste(f_var, f_col_names, sep = "_"))
        j_values <- as.numeric(data_set[[f_var]])  # Se pone como valor de j, el valor del nivel del factor
        
        if (sum(is.na(j_values)) > 0) {  # En categóricas, trato a NA como una categoría más
            j_values[is.na(j_values)] <- length(f_col_names) + 1
            f_col_names <- c(f_col_names, paste(f_var, "NA", sep = "_"))
        }

        if (i == 0) {
            fact_data <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                      x = rep(1, nrow(data_set)),
                                      dims = c(nrow(data_set), length(f_col_names)))
            fact_data@Dimnames[[2]] <- f_col_names
        } else {
            fact_data_tmp <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                          x = rep(1, nrow(data_set)),
                                          dims = c(nrow(data_set), length(f_col_names)))
            fact_data_tmp@Dimnames[[2]] <- f_col_names
            fact_data <- cbind(fact_data, fact_data_tmp)
        }

        i <- i + 1
    }

    if (length(fact_variables) > 0) {
        if (created) {
            out_put_data <- cbind(out_put_data, fact_data)
        } else {
            out_put_data <- fact_data
            created <- TRUE
        }
    }
    return(out_put_data)
}

### Cargamos los datos

setwd('')
train_data <- read.csv("train.csv", header=TRUE, sep=",", stringsAsFactors = TRUE)
train_data$train_sample <- TRUE #creamos una variable pa reconocerlo dsps
train_data$ROW_ID <- seq.int(nrow(train_data)) #Generamos un ID

eval_data <- read.csv("test.csv", header=TRUE, sep=",", stringsAsFactors = TRUE) #mi único problema con esto es q el row id empieza en 0, no en 1
eval_data$train_sample <- FALSE
eval_data$conversion <- NA #lo q tenemos q predecir... lo agregamos pa poder mergear

### Unimos train y eval en un único data.frame ("train_sample" me permite volver a identificarlos)
all_data <- rbind(train_data, eval_data)
all_data$conversion <- ifelse(all_data$conversion == 'False', 0, 1) #buena practica(?)
rm(train_data, eval_data)
gc()

### Acá aplicamos la ingeniería de atributos

summary(all_data) #entendemos preeliminarmente nuestros datos

all_data= select(all_data, -benefit, -etl_version, -accepts_mercadopago, -site_id) #eliminamos las variables que nos piden que eliminemos
#eliminar rn tmbn?? (hay q cambiarle el nombre, este df sirve para jugar)

all_data$date = as.POSIXct(all_data$date,format="%Y-%m-%d")
#all_data$weekend #= #la hacemo?? #creamos variable EsFinDeSemana que creemos puede sumar información
#factor(format(all_data$date, "%w"))#?--> habría q hacerla bool

#Escalamos las variables numéricas usando
all_data <- all_data %>% mutate_at(c("avg_gmv_item_sel", "total_si_domain_30days", "available_quantity", "avg_gmv_seller_bday", "original_price", "offset", "price", "qty_items_sel", "sold_quantity", "total_asp_item_domain_30days", "total_gmv_domain_bday", "total_gmv_item_30days", "total_items_domain", "total_items_seller", "total_orders_domain_30days", "total_orders_item_30days","total_orders_sel_30days", "total_visits_domain", "total_visits_item", "total_visits_seller"), ~(scale(.) %>% as.vector))

# Hacemos one hot encoding
all_data <- one_hot_sparse(all_data)

# Volvemos a separar training y evaluation
train_data <- all_data[all_data[, "train_sample"] == TRUE,]
eval_data <- all_data[all_data[, "train_sample"] == FALSE,]
rm(all_data)
gc()

# Separamos un conjunto de validación
val_index <- sample(1:nrow(train_data), 30000)
val_data <- train_data[val_index,]
train_data <- train_data[-val_index,]

# Pasamos los datos a formato de xgboost
dtrain <- xgb.DMatrix(data = train_data[,setdiff(colnames(train_data), "conversion")],
                      label = train_data[,"conversion"])

dvalid <- xgb.DMatrix(data = val_data[,setdiff(colnames(val_data), "conversion")],
                      label = val_data[,"conversion"])

### Entrenamos un modelo de xgboost con fine tuning de hiperparametros

random_grid <- function(size,
                        min_nrounds, max_nrounds,
                        min_max_depth, max_max_depth,
                        min_eta, max_eta,
                        min_gamma, max_gamma,
                        min_colsample_bytree, max_colsample_bytree,
                        min_min_child_weight, max_min_child_weight,
                        min_subsample, max_subsample) {
    
    rgrid <- data.frame(nrounds = if (min_nrounds == max_nrounds) {
        rep(min_nrounds, size)
    } else {
        sample(c(min_nrounds:max_nrounds),
               size = size, replace = TRUE)
    },
    max_depth = if (min_max_depth == max_max_depth) {
        rep(min_max_depth, size)
    } else {
        sample(c(min_max_depth:max_max_depth),
               size = size, replace = TRUE)
    },
    eta = if (min_eta == max_eta) {
        rep(min_eta, size)
    } else {
        round(runif(size, min_eta, max_eta), 5)
    },
    gamma = if (min_gamma == max_gamma) {
        rep(min_gamma, size)
    } else {
        round(runif(size, min_gamma, max_gamma), 5)
    },
    colsample_bytree = if (min_colsample_bytree == max_colsample_bytree) {
        rep(min_colsample_bytree, size)
    } else {
        round(runif(size, min_colsample_bytree, max_colsample_bytree), 5)
    },
    min_child_weight = if (min_min_child_weight == max_min_child_weight) {
        rep(min_min_child_weight, size)
    } else {
        round(runif(size, min_min_child_weight, max_min_child_weight), 5)
    },
    subsample = if (min_subsample == max_subsample) {
        rep(min_subsample, size)
    } else {
        round(runif(size, min_subsample, max_subsample), 5)
    })
    
    return(rgrid)
}


rgrid <- random_grid(size = 5,
                     min_nrounds = 50, max_nrounds = 300,
                     min_max_depth = 2, max_max_depth = 12,
                     min_eta = 0.001, max_eta = 0.125,
                     min_gamma = 0, max_gamma = 1,
                     min_colsample_bytree = 0.5, max_colsample_bytree = 1,
                     min_min_child_weight = 0, max_min_child_weight = 2,
                     min_subsample = 0.5, max_subsample = 1)


train_xgboost <- function(data_train, data_val, rgrid) {
    
    watchlist <- list(train = data_train, valid = data_val)
    
    predicted_models <- list()
    
    for (i in seq_len(nrow(rgrid))) {
        print(i)
        print(rgrid[i,])
        trained_model <- xgb.train(data = data_train,
                                   params=as.list(rgrid[i, c("max_depth",
                                                             "eta",
                                                             "gamma",
                                                             "colsample_bytree",
                                                             "subsample",
                                                             "min_child_weight")]),
                                   nrounds = rgrid[i, "nrounds"],
                                   watchlist = watchlist,
                                   objective = "binary:logistic",
                                   eval.metric = "auc",
                                   print_every_n = 10)
        
        perf_tr <- tail(trained_model$evaluation_log, 1)$train_auc
        perf_vd <- tail(trained_model$evaluation_log, 1)$valid_auc
        print(c(perf_tr, perf_vd))
        
        predicted_models[[i]] <- list(results = data.frame(rgrid[i,],
                                                           perf_tr = perf_tr,
                                                           perf_vd = perf_vd),
                                      model = trained_model)
        rm(trained_model)
        gc()
    }
    
    return(predicted_models)
}

predicted_models <- train_xgboost(dtrain, dvalid, rgrid) #entreno modelos como dios manda
# Veo los resultados
result_table <- function(pred_models, higher_is_better = TRUE) {
    res_table <- data.frame()
    i <- 1
    for (m in pred_models) {
        res_table <- rbind(res_table, data.frame(i = i, m$results))
        i <- i + 1
    }
    
    hib <- if (higher_is_better) -1 else 1
    
    res_table <- res_table[order(hib *res_table$perf_vd),]
    return(res_table)
}
results <- result_table(predicted_models)
#siendo
predicted_models[[results[1, "i"]]]$model #el mejor modelo entre los q busqué... (EL NUMERO, indica el orden en q quedaron en los round)

### Hacemos las predicciones en eval y generamos el archivo para subir en Kaggle

deval <- xgb.DMatrix(data = eval_data[,setdiff(colnames(eval_data), "conversion")],
                     label = eval_data[,"conversion"])

preds <- predict(predicted_models[[results[1, "i"]]]$model, deval) #lo q vamos a subir y lo q se nos pide// lo q dsps kaggle nos va a decir q tan bn nos fue

#write.csv?
options(scipen = 999)  # Para evitar que se guarden valores en formato científico
write.csv(data.frame(ROW_ID = eval_data[, "ROW_ID"], conversion = preds),
            "predicciones_primer_intento_modif.csv", sep = ",", row.names=FALSE, quote=FALSE) #acá estas creando un archivo q es un df q está guardando lo q le estas pidiendo; se guarda en el directorio q le pasas--> ESO ES LO Q TENÉS Q SUBIR 
options(scipen=0, digits=7)  # Para volver al comportamiento tradicional
