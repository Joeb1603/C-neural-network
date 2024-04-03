#include <stdio.h>
#include <math.h>
#define e 2.718281828459045
#define EXP_A 184
#define EXP_C 16249 
#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr)[0]))
#define PRINT_ARR(arr) print_array(arr, LEN(arr)) //Does not work for pointers
#define PRINT_2DARR(arr) print_2d_array(LEN(training_data), LEN(training_data[0]), arr) //Does not work for pointers
#define PRINTD(double) printf("%f\n",double)
#define ARRAY_MULTIPLIER(arr, num) array_multiplier(arr, LEN(arr), num)
double dot_product(double vector1[], double vector2[], int len);
//double pow_func(double x, double y);
double exp_func(double x);
double sigmoid(double x);
double mse(double x, double y);
double sigmoid_deriv(double x);
int print_2d_array(int len_rows, int len_cols, double x[len_rows][len_cols]);
int print_array(double x[], int len);
double * array_multiplier(double array[],int len, double scaler);
double * subtract_arrays(double arr1[], double arr2[], int len);



int main() {



	FILE *training_output;
	training_output = fopen("training_output.txt","w+");
	
	
	
	//Setting up data
	double training_data[4][3] = {
		{0,0,0},
	  	{0,0,1},
	  	{0,1,0},
	  	{0,1,1}	
	 };
	 double testing_data[4][3] = {
	  	{1,0,0},
	  	{1,0,1},
	  	{1,1,0},
	  	{1,1,1}	
	 };
	double training_targets[] = {0,1,0,1};
	double testing_targets[] = {0,1,0,1};
	
	
	//Setting up initial weights and bias values
	double weights[] = {0.9056419025125352, 0.38798302348487035,0.05958744949098482};
	double bias = 0;
	double training_rate = 0.1;
	int total_epochs = 20000;
	
	
	
	
	//Setting up first batch (Batches don't do anything yet)
	double (*batch)[LEN(training_data[0])] = training_data; //Each batch consists of whole dataset
	int batch_rows_len = LEN(training_data);
	int batch_cols_len = LEN(training_data[0]);
	
	double running_error;
	
	int epoch;
	for(epoch=0;epoch<total_epochs+1;epoch++){
	
		running_error=0;
		//Test current weights
		int current_index;
		for(current_index=0;current_index<batch_rows_len;current_index++){ //for each data item in batch
		
			double layer1 = dot_product(batch[current_index],weights,batch_cols_len)+bias;
			double prediction = sigmoid(layer1);
			
			double error = prediction - training_targets[current_index];
			//printf("Error --- %f\n",error);
			running_error+=error;
			double cost = mse(prediction,training_targets[current_index]);
			
			
			//Find derivatives of cost in relation to weight and bias
			//Cost -> sigmoid -> {weight, bias}
		
			//Derivatives needed for both:
			double cost_deriv = error; //Deriv of cost is 2*error, can ignore the 2x
			double prediction_deriv = sigmoid_deriv(layer1);

			//Derivs for weight and bias
			double (*weight_deriv) = batch[current_index];
			double bias_deriv = 1;
			
			
			//Update weight and bias using the relational derivs 
			double *cost_weight_deriv = array_multiplier(weight_deriv, batch_cols_len, cost_deriv*prediction_deriv);
			//double *new_weights = subtract_arrays(weights,cost_weight_deriv, batch_cols_len);
			
		
			
			int i;
			for(i = 0;i<LEN(weights);i++){
				weights[i]-=(cost_weight_deriv[i]*training_rate);
			}
			
			double cost_bias_deriv = cost_deriv*prediction_deriv*bias_deriv;
			bias-=cost_bias_deriv*training_rate;
			
			
		}
		
		fprintf(training_output,"%f\n",running_error/4);
   		
		
		if (epoch%5000==0){
			printf("Epoch %d complete\n",epoch);
			printf("Error: %f\n\n\n",running_error/4);
			
			//print_array(array_multiplier(weight_deriv,3,0.125000),3);
		}
		
	}
	fclose(training_output);
	
	
	
	int rounded_result;
	int i;
	for(i=0;i<LEN(testing_data);i++){
		double test_layer1 = dot_product(testing_data[i],weights,batch_cols_len)+bias;
		double prediction = sigmoid(test_layer1);
		printf("Test #%d\n",i);
		printf("Prediction: %f\n", prediction);
		printf("Expected result: %f\n", testing_targets[i]);
		if(prediction>=0.5){
			rounded_result = 1;
		}
		else{
			rounded_result=0;
		}
		if(rounded_result==testing_targets[i]){
			printf("Test Passed\n\n");
		}
		else{
			printf("Test Failed\n\n");
		}
	}
	
	
	
}

//The functions are a bit of a mess

double * subtract_arrays(double arr1[], double arr2[], int len){
	static double output2[3]; //bad
	
	int i;
	for(i=0;i<len;i++){
		output2[i] = arr1[i]-arr2[i];
	}
	
	return output2;
}

double * array_multiplier(double array[],int len, double scaler){
	
	static double output[3]; // bad
	int i;
	for(i=0;i<len;i++){
		output[i] = array[i]*scaler;
	}
	return output;
}

int print_array(double x[], int len){
	int i;
	printf("(");
	for(i=0;i<len;i++){
		if (i+1<len){
			printf("%f, ",x[i]);
			
		}else{
			printf("%f)\n",x[i]);
		}
	}
	
}



int print_2d_array(int len_rows, int len_cols, double x[len_rows][len_cols]){
	
	
	printf("%d Rows ",len_rows);
	printf("and %d columns\n\n", len_cols);
	
	int outer_i;
	int inner_i;
	
	for (outer_i = 0; outer_i<len_rows; outer_i++){
		printf("(");
		
		for (inner_i = 0; inner_i<len_cols; inner_i++){
			
			if (inner_i+1<len_cols){
				printf("%f, ",x[outer_i][inner_i]);
			}
			else{
				printf("%f)\n",x[outer_i][inner_i] );
			}
		}
		
	}
	return 0;
}	

double dot_product(double vector1[], double vector2[], int len){

	double output = 0;
	int i;
	for(i=0;i<len;i++){
		output = output+vector1[i]*vector2[i];
	}
	return output;
}


double sigmoid(double x){
	return 1.0 / (1.0 + exp_func(-x));
}

double sigmoid_deriv(double x){
	return sigmoid(x)*(1-sigmoid(x));
}

double exp_func(double x){

	return powf(e,x);
}

double mse(double x, double y){
	return (powf(x-y,2));
}

/*
double pow_func(double x, double y){
	int i;
	double output = 1.0;
	
	if (y<0){
		y=-1*y;
		x=1/x;
	}
	
	for (i=0;i<y;i++){
		output *= x;
	}
	return output;
	
}
*/


