* Loading and inspecting the data;
proc import datafile='/folders/myfolders/sasuser.v94/data.csv'
	out=rawdata;
	getnames=yes;
run;
proc print data=rawdata (obs = 10);run;
proc contents data=rawdata;run;


* Dropping the unwanted columns;
data trimdata;
	Set rawdata (drop=var1 numclaims claimcst0 X_OBSTAT_);
run;
proc print data=trimdata (obs = 10);run;

* Inspecting the data for the possible imbalance with 
  respect to the response variable (clm);
proc freq data=trimdata;
	table clm;
run;


* balancing the data with respect to the response variable;
Proc sort data=trimdata  out=trimdata;
	by clm;
Run;
Proc surveyselect data=trimdata seed=12345  
n= 4624 out=trimdata_balanced;
Strata clm;
Run;

* splitting the data into train/validation and test set;
data train_valid test;
set trimdata_balanced;
p=ranuni(99999);
If (p<=0.7)then output train_valid;
else output test; 
Run;

data train_valid;
	set train_valid (drop=SelectionProb SamplingWeight p);
run;
data test;
	set test (drop=SelectionProb SamplingWeight p);
run;


* making sure that the resulting sets are still balanced;
proc freq data=train_valid;
	table clm;
run;
proc freq data=test;
	table clm;
run;

* inspecting the for correlations;
proc corr data=train_valid; run;

* Checking possible multicolinearity between predictors;
proc reg data=train_valid;
model veh_value= exposure veh_age/tol vif collin;
run; quit;

* using stepwise method to choose the significant variables. For this
  purpose all variables and their interactions are inluded;
proc logistic data=train_valid descending;
class veh_body gender area agecat;
model clm= veh_value exposure veh_body veh_age gender area agecat
	veh_value*exposure veh_value*veh_body veh_value*veh_age veh_value*gender veh_value*area veh_value*agecat
	exposure*veh_body exposure*veh_age exposure*gender exposure*area exposure*agecat
	veh_body*veh_age veh_body*gender veh_body*area veh_body*agecat
	veh_age*gender veh_age*area veh_age*agecat
	gender*area gender*agecat
	area*agecat/ selection=stepwise slentry=0.1
                     slstay=0.05 maxstep = 30;
run; quit;



* The macro that does the cross-validation;
%macro crossval(k,out_train,out_valid);
	data new_data;
		set train_valid;
		sample = mod(_n_,&k);
	Run;
			
	%do j=0 %to &k-1;

		data train validation;
			set new_data;
			if sample^=&j then output train;
			else output validation;
		run;
		
		proc logistic data=train descending;
			class area agecat;
			model clm= veh_value exposure area agecat/ stb;
			ods output Association=p_train;
			score data=validation out=score_temp;
		run; quit;
		
		proc logistic data=score_temp descending;
			model clm=p_1;
			ods output Association=p_valid;
		run; quit;
		
		data p_valid;
			set p_valid;
			c=nValue2;
			sample_v=&j;
			if _n_=4;
			keep sample_v c;
		run;
		
		proc append base=&out_valid data=p_valid;
		run;
		
		data p_train;
			set p_train;
			c=nValue2;
			sample_v=&j;
			if _n_=4;
			keep sample_v c;
		run;
		
		proc append base=&out_train data=p_train;
		run;
	%end;
	
	proc means data=&out_train;
		var c;
	run;
	
	proc means data=&out_valid;
		var c;
	run;
%mend;

* Applying a 5-fold cross-validation;	
%crossval(5,out_train,out_valid);


* applying the model to the test set and checking the result;
proc logistic data=train_valid descending;
	class area agecat;
	model clm= veh_value exposure area agecat/ stb;
	score data=test out=test_pred;
run; quit;
proc logistic data=test_pred descending;
	model clm=p_1;
run; quit;

proc print data=test_pred (obs=15);run;

data diffs;
set test_pred;
diff=abs(I_clm-F_clm);
run;

proc freq data=diffs;
 table diff / nocum BINOMIAL;
run;


