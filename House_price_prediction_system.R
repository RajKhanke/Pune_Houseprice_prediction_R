setwd("C:/Users/rajvk/OneDrive/Documents/r language")

#importing libraries -------------------------------------------------------------------------------------------------------

library(shiny)
library(shinydashboard)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)  
library(glmnet)
library(boot)   
library(rpart)
library(e1071)
library(randomForest)
library(plotly)

#Data cleaning and Preprocessing---------------------------------------------------------------------------------------------------------------------------

f=read.csv("punehouseprediction.csv")
View(f)   
print(dim(f)) 

# give tabular format of count of different area type as output
area_type_count = table(f$area_type)
print(area_type_count)

#removing the unnecessary columns like availability,area_type and society_name from dataset
# %in% means whether one vector exists in another or not
#creating subset by processing from original data frame

f1 = f[, !colnames(f) %in% "availability"] 
f2 = f1[, !colnames(f1) %in% "area_type"]
f3 = f2[, !colnames(f2) %in% "society"]

cat("\n")

#is.na(f) calculate the missing or blanked values for every column and output is generated in tabular format
missing_counts=colSums(is.na(f3))
print(missing_counts)

#removing all rows having missing values using na.omit function in r
f4= na.omit(f3)


#there are errors in size(like 3bhk,3bedroom etc,so identifying total uniques values of house sizes)
unique_BHK_values=unique(f4$size)
print(unique_BHK_values) #printing unique values


# %>% is a pipeline operator which is used to pass f4 to as series of manipulation processes
#mutate is used to create new column bhk
# strsplit splits the words and size column and sapply then apply method to access the first word
#as.integer convert the obtained value to integer

f4 = f4 %>%mutate(size = as.integer(sapply(strsplit(size, " "), "[[", 1)))

#bhk values greater than 10 creates ambiguity, we will tackle in the next parts
bhkgreaterthan10_f4=subset(f4, size > 10)
print(bhkgreaterthan10_f4)
f4 <- f4[f4$size <= 10, ]



#this will compute all unique values in the total sqft and we will observe for any type of ambiguities like ranged sq ft values,non-floating values etc.
unique_sqft=unique(f$total_sqft)
print(unique_sqft)

#this is is_float function,it will check whether the value is numeric or not and handle the exceptions using try catch and print NA as error message in case of any errors
is_float = function(x) 
{
  result = tryCatch({
    as.numeric(x)
  }, error = function(e) {
    NA  
  })
  !is.na(result)
}

#pipeline operator is again used to do manipulations on f4 total sqft column in which function is_float is checked to check for floating values and non-floating filtered values are separated in a data frame filtered_sq_ft_f5
filtered_sq_ft_f5 = f4[!f4$total_sqft %>% sapply(is_float), ]
#(we found many errors in Sqft like 1234sq yard ,2345sq feets etc as well as ranges like 345-3456 etc)


# so to remove range based sq_ft,all sqft values are converted to num using function convert_sqft_to_num
#token variable splits the two range numbers,before and after hyphen(lower and upper bounds)
#now if length of tokens is 2,lower and upper bounds both will be converted to numeric and their average value should be taken
#if value of sqft is not numeric ,it will convert to numeric else if unable to convert ,it will show NA
convert_sqft_to_num = function(x) {
  tokens = unlist(strsplit(x, " - "))
  
  if (length(tokens) == 2) {
    lower_bound = as.numeric(tokens[1])
    upper_bound = as.numeric(tokens[2])
    if (!any(is.na(c(lower_bound, upper_bound)))) {
      return((lower_bound + upper_bound) / 2)
    }
  }
  
  numeric_value = as.numeric(x)
  
  if (!is.na(numeric_value)) {
    return(numeric_value)
  }
  
  cat("Unable to convert:", x, "\n")  # Print the value that caused the issue
  return(NA)
}


#remove non numeric  and range hyphen sqft function is applied to f4 data frame
f4$total_sqft = sapply(f4$total_sqft, convert_sqft_to_num)

#removing all rows where sqft value is NA
f5=f4[!is.na(f4$total_sqft), ]

f5$total_sqft <- f5$total_sqft / 2
# feature engineering and outlier Removal ------------------------------------------------------------------------------


#price per sq ft column is added to f5 data frame (multiplied by 1lakh becuases prices were in lakhs) 
f5$price_per_sqft = (f5$price * 100000) / f5$total_sqft


#finding total number of unique locations and printing them to take an idea
unique_location=unique(f5$location)
print(unique_location)

#counting every location among 97 occured how many times in dataset in tabular format and sorting them afterwards

location_counts = table(f5$location)
location_counts = sort(location_counts)
print(location_counts)

#removing location that appeared less,but every location appears on average 100-120 times except one place where location is not mentioned so removing that blancked place
f6 = f5[f5$location != "", ]

#less than 300sqft per bedroom is unusual things in house price market,so removing such type of outliers
f7= f6[!(f6$total_sqft / f6$size < 250), ]

#describing the properties of price_per_sqft like min,max,std.deviation,mean etc.
print(summary(f7$price_per_sqft))

#for data outlier removing, using traditional technique to keep only one standard deviation below and above data from mean
#f7_out is output data initialized with data frame inside the function
#f7 datasets grouped by location and price-per_sqft are binded rows and output f7_out is generated

remove_pps_outliers = function(f7) {
  f7_out = data.frame()
  f7_grouped = f7 %>%
    group_by(location)
  f7_grouped = f7_grouped %>%
    mutate(mean_pps = mean(price_per_sqft),
           std_pps = sd(price_per_sqft))
  f7_filtered = f7_grouped %>%
    filter(price_per_sqft > (mean_pps - std_pps) & price_per_sqft <= (mean_pps + std_pps))
  f8_out = bind_rows(f7_out, f7_filtered)
  return(f8_out)
}
f8 = remove_pps_outliers(f7)

print(dim(f))

#plotting the scatter plot of location vs price_per_sqft to observe the outliers
#bhk2 and bhk3 data are initialized with f8 dataset with bhk 2 and 3 bedrooms and location parameter
#scatter plot is plotted using ggplot(),with size,shape being adjusted,x and y labels are given as total sq feet area and lak indian rupees
#minimal theme is choosen with legend at top
#minimal theme being predefined theme and legend is info of symbols on top
plot_scatter_chart = function(f8, location)
{
  
  bhk2 = f8[f8$location == location & f8$size == 2, ]
  bhk3 = f8[f8$location == location & f8$size == 3, ]
  
  p = ggplot() +
    geom_point(data = bhk2, aes(x = total_sqft, y = price), color = 'blue', size = 3, shape = 19) +
    geom_point(data = bhk3, aes(x = total_sqft, y = price), color = 'green', size = 3, shape = 3) +
    labs(x = "Total Square Feet Area", y = "Price (Lakh Indian Rupees)", title = location) +
    theme_minimal() +
    theme(legend.position = "top") +
    scale_shape_manual(values = c(19, 3), name = "BHK", labels = c("2 BHK", "3 BHK")) +
    scale_color_manual(values = c("blue", "green"), name = "BHK", labels = c("2 BHK", "3 BHK"))
  
  print(p)
}

#plotting the scatter_plots of bibwewadi and katraj areas to test,detect and observe outlier points
plot_scatter_chart(f8, "Katraj")
plot_scatter_chart(f8, "Bibvewadi")


#custom function to remove outlliers and such rows where mean price_per_sqft of 1 bhk is greater than price per sq_feet of 2 bhk
#f8 is group by location and multiple manipulations are done on it using pipeline operator
#then it will check for mean of 1 bhk price per sqfeet and remove/ungroup all those rows where it is less than price per sqft for 2 bhk
remove_bhk_outliers = function(f8) 
{
  f8 %>%
    group_by(location) %>%
    mutate(mean_1bhk_pps = mean(price_per_sqft[size == 1])) %>%
    filter(!(size== 2 & price_per_sqft < mean_1bhk_pps)) %>%
    ungroup()
}

# Call the function to remove outliers
f9 = remove_bhk_outliers(f8)

#replotting scatter plots for katraj and bibvewadi regions for difference identification in outlier removal
plot_scatter_chart(f9, "Katraj")
plot_scatter_chart(f9, "Bibvewadi")

# adjust the dimensions of the histogram such as height and the weight
options(repr.plot.width=20, repr.plot.height=10)

# Create a histogram having intervals of 20,on price_per_sqft whose name in the main,x & y lab as x and y axis names,color and range adjustments
hist(f9$price_per_sqft, breaks = 20, main = " count of Price Per Square Feet Histogram",
     xlab = "Price Per Square Feet", ylab = "Count", col = "red", border = "black", xlim = c(0, max(f9$price_per_sqft)))

#printing all unique no of bathroom values to look for outliers
unique_bath_values=unique(f9$bath)
print(unique_bath_values)

# Set the plot size (width,height and dimensions,name(main),color,border etc of bathroom counts)
options(repr.plot.width = 8, repr.plot.height = 6)

#histogram for bathrooms is just for outliers observation
hist(f9$bath, breaks = 20, main = "Number of Bathrooms Histogram",
     xlab = "Number of Bathrooms", ylab = "Count", col = "blue", border = "black", xlim = c(0, max(f9$bath)))

#searching for rows where bath greater than 10 to observe ambiguities
filtered_f9_bath = f9[f9$bath > 10, ]

#it is unusual to have two more bathrooms than number of bedrooms,so removing such type of outliers
filtered_bath_f9final = f9[f9$bath > f9$size + 2, ]

f10 = f9[f9$bath < f9$size + 2, ]

#searching for balcony outliers

#usually galleries not greater than number of rooms,so removing such outliers
filtered_f10_balcony = f10[f10$balcony > f10$size+1, ]
#no outliers as filtered_f10_balcony has no values

#removing size and price_per_sqft as they are useles now after outlier removal

df = f10 %>%
  select(-price_per_sqft,-mean_pps,-std_pps,-mean_1bhk_pps)
View(df)
print(dim(df)) 

#---------------------------------------------------------------------------------------------------

#  1.Multiple linear regression model on the dataset---------------------------------------------------------

#declaring our feature input matrix or independent variable x and dependent or target variable y

#x will include all values in dataset df2 and exclude the target column price
X = df[, !names(df) %in% c("price")]

#y will form a vector of only target value price
Y = df$price

#checking for observations first few values of x and y and their dimensions

print(head(X, 3))
colnames(X) = gsub("df.location", "", colnames(X))

print(head(Y, 3))


print(dim(X))

#get length of y vector
print(length(Y))



# Split the data into training and testing sets
set.seed(52)
# the seed for the model is set to seed value 52.This ensures that accuracy value or predicted values generated should be reproducible

# a training index variable is declared using function createDatapartition inbuilt function having targeted value y,80% trained dataset and 20% tested dataset
#list=false ensures that we will get vector or index as output
#times indicate total number of times the data is splitted
trainIndex = createDataPartition(Y, p = .8,  list = FALSE, times = 1)


#show training and testing data and their dimensions

Training_data=df[trainIndex, ]
Testing_data=df[-trainIndex, ]

print(dim(Training_data))
print(dim(Testing_data))

#3909 trained and 909 tested-------


#this line contains the variable X_train where all training independent data(80%)is stored and 
#y-train where all output trained data is stored
X_train = X[trainIndex,]
y_train = Y[trainIndex]
#xtest and y test contains all data except traindataset that is test data
X_test  = X[-trainIndex,]
y_test  = Y[-trainIndex]

# Train the Linear Regression model(lm is function used to execute linear ml model)
#linear regression is executed on Xtrain and output as y_train,"."indicates all variables in x are used as predictors
Multiple_linear_regression <- lm(y_train ~ ., data = X_train)

# Check the the summary of performance of linear reggression model
print(summary(Multiple_linear_regression))

# predicting the house price for testing dataset using linear regression model
predicted_houseprice = predict(Multiple_linear_regression, X_test)


#evaluation matrix for the model (R^2 value,MAE(mean absolute error),MSE(mean squared error))
#calculating the r-squared value to determine the model's accuracy

cat("\n \n evaluation matrix for Multiple linear regression : \n \n")
R2 = 1 - sum((y_test - predicted_houseprice)^2) / sum((y_test - mean(y_test))^2)
cat("\n accuracy for linear regression model:",R2)
# Calculate Mean Absolute Error (MAE)
MAE = mean(abs(y_test - predicted_houseprice))
cat("\n Mean Absolute Error (MAE):", MAE)

# Calculate Mean Squared Error (MSE)
MSE = mean((y_test - predicted_houseprice)^2)
cat("\n Mean Squared Error (MSE):", MSE)

# Calculate Mean Squared Error (MSE)
MSE = mean((y_test - predicted_houseprice)^2)
cat("\n Mean Squared Error (MSE):", MSE)

# Calculate Root Mean Squared Error (RMSE)
RMSE = sqrt(mean((y_test - predicted_houseprice)^2))
cat("\n Root Mean Squared Error (RMSE):", RMSE)

# Calculate Mean absolute percentage Error (MAPE)
MAPE = mean(abs((y_test - predicted_houseprice) / y_test)) * 100
cat("\nMean absolute percentage Error (MAPE):", MAPE)

cat("\n\n\n")

#2.decision trees----------------------------------------------------------------------------------------------

# Train the Decision Tree model
decision_tree <- rpart(y_train ~ ., data = X_train)
# Predict house prices using Decision Tree model
predicted_houseprice_tree <- predict(decision_tree,X_test )

# Calculate R-squared
R2_tree <- 1 - sum((y_test - predicted_houseprice_tree)^2) / sum((y_test - mean(y_test))^2)

# Calculate Mean Absolute Error (MAE)
MAE_tree <- mean(abs(y_test - predicted_houseprice_tree))

# Calculate Mean Squared Error (MSE)
MSE_tree <- mean((y_test - predicted_houseprice_tree)^2)

# Calculate Root Mean Squared Error (RMSE)
RMSE_tree <- sqrt(mean((y_test - predicted_houseprice_tree)^2))

# Calculate Mean absolute percentage Error (MAPE)
MAPE_tree <- mean(abs((y_test - predicted_houseprice_tree) / y_test)) * 100

# Print Decision Tree model evaluation metrics
cat("\nEvaluation matrix for Decision Tree:\n\n\n")
cat("Accuracy for Decision Tree model (R-squared):", R2_tree, "\n")
cat("Mean Absolute Error (MAE):", MAE_tree, "\n")
cat("Mean Squared Error (MSE):", MSE_tree, "\n")
cat("Root Mean Squared Error (RMSE):", RMSE_tree, "\n")
cat("Mean Absolute Percentage Error (MAPE):", MAPE_tree, "\n\n\n")


# 3.support vector machine (SVM0----------------------------------------------------------------------------------------



# Assuming you have already split your data into training and testing sets (X_train, y_train, X_test)
# Train the SVR model
svm_model <- svm(y_train ~ ., data = X_train, kernel = "linear")

# Predict house prices for the test dataset
predicted_house_prices <- predict(svm_model, X_test)

# Calculate R-squared
R2_svm <- 1 - sum((y_test - predicted_house_prices)^2) / sum((y_test - mean(y_test))^2)

# Calculate Mean Absolute Error (MAE)
MAE_svm <- mean(abs(y_test - predicted_house_prices))

# Calculate Mean Squared Error (MSE)
MSE_svm <- mean((y_test - predicted_house_prices)^2)

# Calculate Root Mean Squared Error (RMSE)
RMSE_svm <- sqrt(mean((y_test - predicted_house_prices)^2))

# Calculate Mean Absolute Percentage Error (MAPE)
MAPE_svm <- mean(abs((y_test - predicted_house_prices) / y_test)) * 100

# Print SVM model evaluation metrics


cat("Evaluation matrix for Support Vector Machine (SVM):\n\n")
cat("Accuracy for SVM model (R-squared):", R2_svm, "\n")
cat("Mean Absolute Error (MAE):", MAE_svm, "\n")
cat("Mean Squared Error (MSE):", MSE_svm, "\n")
cat("Root Mean Squared Error (RMSE):", RMSE_svm, "\n")
cat("Mean Absolute Percentage Error (MAPE):", MAPE_svm, "\n")



#4.Random_Forest-------------------------------------------------------------------------------------------------------------------

# Train the Random Forest model
random_forest_model <- randomForest(y_train ~ ., data = X_train)

# Predict house prices for the test dataset
predicted_houseprice_rf <- predict(random_forest_model, X_test)

# Calculate R-squared
R2_rf <- 1 - sum((y_test - predicted_houseprice_rf)^2) / sum((y_test - mean(y_test))^2)

# Calculate Mean Absolute Error (MAE)
MAE_rf <- mean(abs(y_test - predicted_houseprice_rf))

# Calculate Mean Squared Error (MSE)
MSE_rf <- mean((y_test - predicted_houseprice_rf)^2)

# Calculate Root Mean Squared Error (RMSE)
RMSE_rf <- sqrt(mean((y_test - predicted_houseprice_rf)^2))

# Calculate Mean Absolute Percentage Error (MAPE)
MAPE_rf <- mean(abs((y_test - predicted_houseprice_rf) / y_test)) * 100

# Print Random Forest model evaluation metrics
cat("\n\nEvaluation matrix for Random Forest:\n\n")
cat("Accuracy for Random Forest model (R-squared):", R2_rf, "\n")
cat("Mean Absolute Error (MAE):", MAE_rf, "\n")
cat("Mean Squared Error (MSE):", MSE_rf, "\n")
cat("Root Mean Squared Error (RMSE):", RMSE_rf, "\n")
cat("Mean Absolute Percentage Error (MAPE):", MAPE_rf, "\n")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#data visualization through graphs---------------------------------------------------------------------------------------------------------------------------------------------


#Create data frames for plotting
mlr_plot_data <- data.frame(Actual = y_test, Predicted = predicted_houseprice, Model = "Multiple Linear Regression")
dt_plot_data <- data.frame(Actual = y_test, Predicted = predicted_houseprice_tree, Model = "Decision Tree")
svm_plot_data <- data.frame(Actual = y_test, Predicted = predicted_house_prices, Model = "SVM")
Rf_plot_data <- data.frame( Actual = y_test, Predicted = predicted_houseprice_rf, Model = "Random Forest")

# Combine the data frames
combined_plot_data <- rbind(mlr_plot_data, dt_plot_data, svm_plot_data,Rf_plot_data)

# Create the plot
predicted_actual_plot <- ggplot(combined_plot_data, aes(x = Actual, y = Predicted, color = Model)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed") +
  labs(title = "Predicted vs. Actual House Prices", x = "Actual Prices", y = "Predicted Prices") +
  theme_minimal()

# Convert the ggplot object to a plotly object
predicted_actual_plot <- ggplotly(predicted_actual_plot)


# Display the plot
print(predicted_actual_plot)

# Create a data frame for model accuracy comparison
accuracy_data <- data.frame(Model = c("Multiple Linear Regression", "Decision Tree", "SVM","Random Forest"),
                            Accuracy = c(R2, R2_tree, R2_svm,R2_rf))

# Create the accuracy comparison plot
accuracy_plot <- ggplot(accuracy_data, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "R-squared Accuracy") +
  theme_minimal() +
  ylim(0, 1)  # Set y-axis limits

# Display the accuracy comparison plot
print(accuracy_plot)


# so Multiple linear regression is found most accurate with 86% accuracy followed by svm with 84%
#let's perform k-fold(10 fold) cross validation on multiple linear regression model and svm model


#k-fold cross validation----------------------------------------------------------------------------------------------------------------------------------------



#k-fold(10 folds) cross validation to check validation of accuracy of linear regression model
cat("\n\n\n")

# Set the number of folds and test data size(0.2 = 20%)
k = 10
test_size = 0.2

#declaring empty vector to store cross validation results
cv_results = numeric(k)

#seed is set using seed value 52 to minimize randomness and provide accurate results
# Set seed for reproducibility
set.seed(52)

# Perform cross-validation by loop iterating from i=1 to 10
for (i in 1:k) 
{
  # again creating Multiple linear regression model for cross validation,p--trained size=total-test_size)
  trainIndex = createDataPartition(Y, p = 1 - test_size, list = FALSE)
  X_train = X[trainIndex,]
  y_train = Y[trainIndex]
  X_test = X[-trainIndex,]
  y_test = Y[-trainIndex]
  Multiple_linear_regression = lm(y_train ~ ., data = X_train)
  predicted_houseprice = predict(Multiple_linear_regression, newdata = X_test)
  R2 = 1 - sum((y_test - predicted_houseprice)^2) / sum((y_test - mean(y_test))^2)
  
  # Store the result
  cv_results[i] = R2 # each r-squared value accuracy index is stored in ith index of cross validation vector
}

# Print cross-validation results
cat("Cross-validation results:\n", cv_results, "\n")
cat("Mean R-squared:", mean(cv_results), "\n")
cat("\n\n\n")
#mean accuracy obtained on 10-fold cross validation of linear regression is 81%
#--------------------------------------------------------------------------------------------------------


# Set the number of folds and test data size (0.2 = 20%)
k = 10
test_size = 0.2

# Declare an empty vector to store cross-validation results
cv_results_svm = numeric(k)

# Perform cross-validation using a loop
for (i in 1:k) {
  # Create training and testing data subsets for the current fold
  trainIndex = createDataPartition(Y, p = 1 - test_size, list = FALSE)
  X_train = X[trainIndex,]
  y_train = Y[trainIndex]
  X_test = X[-trainIndex,]
  y_test = Y[-trainIndex]
  
  # Train the SVM model on the training data
  svm_model = svm(y_train ~ ., data = X_train, kernel = "linear")
  
  # Predict house prices for the test data
  predicted_house_prices_svm = predict(svm_model, X_test)
  
  # Calculate the R-squared value for the current fold
  R2_svm = 1 - sum((y_test - predicted_house_prices_svm)^2) / sum((y_test - mean(y_test))^2)
  
  # Store the result in the cross-validation results vector
  cv_results_svm[i] = R2_svm
}

# Print cross-validation results
cat("Cross-validation results for SVM model:\n", cv_results_svm, "\n")
cat("Mean R-squared for SVM:", mean(cv_results_svm), "\n")
#mean accuracy for svm is found to be 78%


#predicting house prices with user input

# Create a data frame with default values
custom_data <- data.frame(
  size = numeric(1),
  total_sqft = numeric(1),
  bath = numeric(1),
  balcony = numeric(1),
  location = character(1)
)


#GUI using shinny--------------------------------------------------------------------------------------------------------------------------------

# Load necessary libraries
library(shiny)
library(shinydashboard)

# Define UI for the Shiny app
ui <- dashboardPage(
  dashboardHeader(title = "Pune House Price Prediction"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Home", tabName = "home", icon = icon("home")),
      menuItem("Predict Price", tabName = "predict", icon = icon("calculator"))
    )
  ),
  
  dashboardBody(
    tabItems(
      tabItem(
        tabName = "home",
        fluidRow(
          box(
            title = "Welcome to Pune House Price Prediction",
            width = 12,
            height = 300,
            background = "purple",
            "This is a Shiny dashboard for predicting house prices in Pune.\n Created By Raj Khanke (CSE (AIML)) from VIT PUNE."
            
          )
        )
      ),
      
      tabItem(
        tabName = "predict",
        fluidRow(
          box(
            title = "Custom Input",
            width = 6,
            status = "primary",
            solidHeader = TRUE,
            textInput("size", "Number of Bedrooms", value = ""),
            textInput("total_sqft", "Total Square Feet Area", value = ""),
            textInput("bath", "Number of Bathrooms", value = ""),
            textInput("balcony", "Number of Balconies", value = ""),
            selectInput("location", "Location", choices = unique_location, selected = ""),
            actionButton("predictBtn", "Predict", class = "btn-primary"),
            actionButton("clearBtn", "Clear", class = "btn-default")
          ),
          box(
            title = "Prediction Result",
            width = 6,
            status = "success",
            solidHeader = TRUE,
            verbatimTextOutput("predictedPrice")
          )
        )
      )
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  # Function to predict house price
  predictPrice <- function(size, total_sqft, bath, balcony, location) {
    custom_data <- data.frame(
      size = as.numeric(size),
      total_sqft = as.numeric(total_sqft),
      bath = as.numeric(bath),
      balcony = as.numeric(balcony),
      location = location
    )
    
    # Replace with your actual Multiple linear regression model
    predicted_price <- predict(Multiple_linear_regression, newdata = custom_data)
    return(predicted_price)
  }
  
  # Predict and update the output when the "Predict" button is clicked
  observeEvent(input$predictBtn, {
    predicted_price <- predictPrice(
      input$size,
      input$total_sqft,
      input$bath,
      input$balcony,
      input$location
    )
    
    # Convert the predicted price to lakh rupees
    predicted_price_in_lakh <- predicted_price 
    output$predictedPrice <- renderText({
      paste(predicted_price_in_lakh, "Lakh Rupees")
    })
  })
  
  # Observe the "Clear" button click
  observeEvent(input$clearBtn, {
    # Reset or clear the input fields
    updateTextInput(session, "size", value = "")
    updateTextInput(session, "total_sqft", value = "")
    updateTextInput(session, "bath", value = "")
    updateTextInput(session, "balcony", value = "")
    updateSelectInput(session, "location", selected = "")
    output$predictedPrice <- renderText({""})
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)
