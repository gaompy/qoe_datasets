install.packages("mice")
require("mice")

for (j in 1:7) {
    input_csv_file = paste(j,"/data.csv",sep="")
    output_csv_file = paste(j,"/data_filled_r.csv",sep="")

    mydata <- read.csv(input_csv_file, header = TRUE)

    mydata$ID <- seq.int(nrow(mydata))

    filter_not2fill = (names(mydata) %in% c("X_id", "X_rev", "app_foreground", "devid", "ts_from", "ts_to"))
    filter_2fill = !filter_not2fill
    filter_not2fill = (names(mydata) %in% c("X_id", "X_rev", "app_foreground", "devid", "ts_from", "ts_to", "ID"))
    
    df1 = mydata[filter_2fill]
    df2 = mydata[filter_not2fill]

    df1 <- mice(df1, method = "cart")
    df1 <- complete(df1)
    
    mydatafilled = merge(df1, df2, by="ID")

    towrite = mydatafilled[!(names(mydatafilled) %in% c("ID"))]

    write.csv(towrite, output_csv_file, row.names = F)
}
