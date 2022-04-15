library(dplyr)

rm(list = ls())

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

zup= select(all_data, -benefit, -etl_version) #eliminamos las variables que nos piden que eliminemos
#eliminar rn tmbn?? (hay q cambiarle el nombre, este df sirve para jugar)

all_data$date = as.POSIXct(all_data$date,format="%Y-%m-%d")
all_data$weekend #= #la hacemo?? #creamos variable EsFinDeSemana que creemos puede sumar información
#factor(format(all_data$date, "%w"))#?--> habría q hacerla bool







all_data$print_server_timestamp #???
strptime(all_data$print_server_timestamp,format="%Y-%m-%dT%H:%M:%OS-%z") #formato raro que tenemos q poder cambiar
as.POSIXct(all_data$print_server_timestamp,format="%Y-%m-%d %H:%M:%OS")
#el 0400 del dataframe og q es???--> %z?

#SE PUEDE APLICAR EL PREPROCESAMIENTO DE CARET



all_data$available_quantity  #estandarizar
#?avg_gmv_item_domain_30days
all_data$avg_gmv_item_sel
all_data$avg_gmv_seller_bday

all_data$is_pdp  #blanco= NA ?

#all_data$qty_items_dom  #no me pa

#estandarizar
#?original_price?
avg_qty_orders_item_sel_30days #binarizar(?) --> >50 no sirve
#?price
#?print_position

all_data$qty_items_dom  #estandarizar
all_data$qty_items_sel  
all_data$sold_quantity

#llegué a total_asp_item_domain_30days




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

### Entrenamos un modelo



#write.csv?
options(scipen = 999)  # Para evitar que se guarden valores en formato científico
write.table(data.frame(ROW_ID = eval_data[, "ROW_ID"], conversion = preds),
            "predicciones_primer_intento.txt", sep = ",", row.names=FALSE, quote=FALSE) #acá estas creando un archivo q es un df q está guardando lo q le estas pidiendo; se guarda en el directorio q le pasas--> ESO ES LO Q TENÉS Q SUBIR 
options(scipen=0, digits=7)  # Para volver al comportamiento tradicional
