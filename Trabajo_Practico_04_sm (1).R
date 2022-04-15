library(caret)
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
all_data$weekend #= #la hacemo?? #creamos variable EsFinDeSemana que creemos puede sumar información
#factor(format(all_data$date, "%w"))#?--> habría q hacerla bool

all_data$print_server_timestamp #???
as.Date(all_data$print_server_timestamp)

strptime(all_data$print_server_timestamp,format="%Y-%m-%dT%H:%M:%OS-%z") #formato raro que tenemos q poder cambiar
as.POSIXct(all_data$print_server_timestamp,format="%Y-%m-%d %H:%M:%OS")
#el 0400 del dataframe og q es???--> %z?

#SE PUEDE APLICAR EL PREPROCESAMIENTO DE CARET

all_data$is_pdp  #blanco= NA ?

#estandarizar el resto


# Hacemos one hot encoding?
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




### Entrenamos un modelo de ??

"##El más complejo y lento es un modelo que necesita correr muchos modelos 
para componerse... eg el Trabajo_Practico_04_pm y cualquiera de ensamble.
Cuando tenes uno tipo k fold (estas construyendo solo un modelo...) podes jugar
con otras técnicas/esquemas de validación, para estar SEGURO de q tu medida
de confianza es muy robusta. IE u are + likely 2 use a measure like hold-out set
for models that take long to create, pero eso no es así para modelos tipo knn, 
donde por ej podés buscar medidas más robustas para estar SEGURO de q tu hiperp 
elegido es muy muy bueno y el mejor de todos. A su vez, podés elegir algo tipo 
grid search (en lugar de random search) para encontar los hiperp...

En síntesis, tenés más cintura para probar."

fitControl <- trainControl(method =
                               "de validac")

modelo_1 <- train(conversion ~ . "?", 
                  data = train_data, 
                  method = "algoritmo", preProcess(), 
                  trControl = fitControl, metric = "AUC")

predicc_modelo_1 = predict(modelo_1, eval_data)
confusionMatrix(predicc_modelo_1, eval_data$conversion) #????--> no podemo realmente saber esto


### Hacemos las predicciones en eval

preds <- predict(base_model, deval)

### Generamos el archivo para subir en Kaggle

options(scipen = 999)  # Para evitar que se guarden valores en formato científico
write.csv(data.frame(ROW_ID = eval_data[, "ROW_ID"], conversion = preds),
            "predicciones_segundo_intento.csv", sep = ",", row.names=FALSE, quote=FALSE) #acá estas creando un archivo q es un df q está guardando lo q le estas pidiendo; se guarda en el directorio q le pasas--> ESO ES LO Q TENÉS Q SUBIR 
options(scipen=0, digits=7)  # Para volver al comportamiento tradicional
