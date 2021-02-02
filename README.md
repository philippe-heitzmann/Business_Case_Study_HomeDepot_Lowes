# Business Case Study - Analyzing and predicting new store locations for Lowe's and Home Depot

## Background

##### In this case study, we will be put in the shoes of a consultant looking to understand some of the factors that Lowe's and Home Depot look to when deciding to build new stores. We will then use this strategy to advise a new competitor we will call 'Tool Time' on where this new entrant should build its next five stores based on this information on Home Depot and Lowe's.

## Case Study Outline 

1. **Exploratory Data Analysis & creating a map of stores**
1. **Identifying relationships between variables with linear regression model**
1. **Leveraging linear regression model to perform _new store location predictions_ for Lowe's, Home Depot and Tool Time**
1. **Comparing our results to realtor.com 'market hotness' data** 


## Questions:
1. **Perform Exploratory Data Analysis on the stores**
	1. **What are the total store counts of Home Depot and Lowes?**

		There are a total of 1706 Lowe's stores and 1952 Home Depot stores in this dataset 

		```python
		#reading in our data 
		hdlo = pd.read_csv("Home_Depot_Lowes_Data.csv", sep = ',')
		print('There are a total of {} HDSupply stores in this dataset'.format(np.sum(hdlo.HDcount)))
		print('There are a total of {} Lowe\'s stores in this dataset'.format(np.sum(hdlo.Lcount)))
		```

	1. **Create one dummy variable for Home Depot and one dummy variable for Lowes
that identifies whether or not the store is located in each county**

		```python
		hdlo['HD_dummy'] = (hdlo['HDcount'] > 0) * 1
		hdlo['Lo_dummy'] = (hdlo['Lcount'] > 0) * 1
		```

	1. **Which store is present in more counties?**

		As can be seen from the below, Lowe's is present in 924 counties while Home Depot is present in 785 counties in the United States 

		```python
		print('Lowe\'s stores are present in {} counties'.format(np.sum(hdlo.Lo_dummy)))
		print('HDSupply stores are present in {} counties'.format(np.sum(hdlo.HD_dummy)))
		print('HDSupply have a presence in a higher number of counties than do Lowe\'s stores')
		```

1. **Use a United States map with FIPS locations to plot the store locations of both Lowes
and Home Depot**

	```python
	import plotly.express as px
	fig = px.choropleth(hdlo, geojson=counties, locations='county', color='HDcount',
	                           color_continuous_scale="Viridis",
	                           range_color=(0, 48),
	                           scope="usa",
	                           labels={'HDcount':'Home Depot stores'},
	                           title = 'Home Depot Stores in the United States')
	# fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
	fig.show()
	#
	fig = px.choropleth(hdlo, geojson=counties, locations='county', color='Lcount',
                           color_continuous_scale="Viridis",
                           range_color=(0, hdlo.Lcount.max()),
                           scope="usa",
                           labels={'Lcount':'Lowe\'s stores'},
                           title = 'Lowe\'s Stores in the United States')
	fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
	fig.show()
	```
	1. **What observations can you make from the map?**

	There look to be high concentrations of Lowe's and Home Depot stores in metropolis areas and other high density population centers. 

1. **Create a linear regression model to identify the correlations among the variables.**
		
	```python
	#imputing our data
	hdlo['pct_U18_2000'].fillna(hdlo.pct_U18_2000.mean(), inplace = True)
	hdlo['pct_U18_2010'].fillna(hdlo.pct_U18_2000.mean(), inplace = True)
	hdlo['pctwhite_2000'].fillna(hdlo.pct_U18_2000.mean(), inplace = True)
	hdlo['pctwhite_2010'].fillna(hdlo.pct_U18_2000.mean(), inplace = True)
	hdlo['pctblack_2000'].fillna(hdlo.pct_U18_2000.mean(), inplace = True)
	hdlo['pctblack_2010'].fillna(hdlo.pct_U18_2000.mean(), inplace = True)
	#further cleaning and separating our data 
	hd_target = hdlo.HDcount
	lo_target = hdlo.Lcount
	hd_data = hdlo.copy()
	lo_data = hdlo.copy()
	hd_data.drop(['areaname','county','state','r1', 'r2', 'HD_dummy', 'Lo_dummy','HDcount','Lcount'], axis = 1, inplace = True)
	lo_data.drop(['areaname','county','state','r1', 'r2', 'HD_dummy', 'Lo_dummy','HDcount','Lcount'], axis = 1, inplace = True)
	#fitting our linear regression model
	from sklearn.linear_model import LinearRegression
	lr1 = LinearRegression(normalize=True).fit(hd_data, hd_target)
	print('Home Depot Regression Coefficients are:', lr1.coef_)
	lr2 = LinearRegression(normalize=True).fit(lo_data, lo_target)
	print('Lowe\'s Regression Coefficients are:', lr1.coef_)
	#showing cross validation scores
	from sklearn.model_selection import cross_val_score
	print(cross_val_score(lr1, hd_data, hd_target, cv = 5, n_jobs = -1))
	print(cross_val_score(lr2, lo_data, lo_target, cv = 5, n_jobs = -1))
	```

	1. **What customer demographic variables are most import to Lowes?**

		Lowe's is most interested in building new stores in areas where a large portion of the population is under the age of 18. This may be due to the fact that this may be an indicator that this county has high amounts of young families looking to potentially invest more down the road in home improvement. Lowe's is also interested in areas with high racial diversity and high rates of college education 
		```python
		hd_scores = list(zip(hd_data.columns, lr1.coef_))
		lo_scores = list(zip(lo_data.columns, lr2.coef_))	
		sorted(lo_scores, key = lambda x: x[1], reverse = True)
		```

	1. **What customer demographic variables are most import to Home Depot?**

		Home Depot is most interested in building new stores in areas with high degrees of racial diversity. Interestingly, Home Depot actually builds less stores all else equal in areas with large amounts of the population that is under 18 years old, in contract to Lowe's which chooses to build more houses in areas with more <18yo people as a percentage of the population all else equal 
		```python
		sorted(hd_scores, key = lambda x: x[1], reverse = True)
		```

	1. **How are the chains similar in their decision making?**

		As percent white and percent black variables appear in the top three variables of each regression analysis, it appears both chains place a heavy emphasis on areas with high degrees of racial diversity. Additionally, with pct_U18 and pctcollege appearing in the top five variables for each, counties with relatively higher amounts of young children, perhaps an indicator of families looking to further extend / renovate their homes, and relatively higher amounts of college educated people, perhaps an indicator of higher income households, also seem to be targets for new store construction for each chain. 	


	1. **How are they different?**

		Lowe's appears to be more interested in areas with larger amounts of the population being under 18 years of age. On the other hand, Home Depot appears to prioritize areas with relatively higher rates of homeownership than does Lowe's. 

1. **What are the top 5 towns/cities that can be predicted as potential candidates for new
locations for both Lowes and Home Depot?**
	
	Two ways to answer this question - one using a simple linear regression and one using a logistic regression regressing against the dummy variables for existence of a Lowe's or Home Depot store in a specific county. 

	**First way - using simple linear regression**

	```python 
	nohd = hdlo.loc[hdlo['HDcount'] == 0]
	nohd.drop(['areaname','county','state','r1', 'r2', 'HD_dummy', 'Lo_dummy','HDcount','Lcount'], axis = 1, inplace = True)
	answers = zip(hdlo.loc[hdlo['HDcount'] == 0]['areaname'], list(lr1.predict(nohd)))
	sorted(answers, key = lambda x: x[1], reverse = True)[:5]
	```

	Output:
	[('San Francisco', 2.9847174013397284),
	 ('Union', 1.8780228205575722),
	 ('Sangamon', 1.4803420877554014),
	 ('Tolland', 1.4637971918204253),
	 ('Johnson', 1.4608526781148954)]

	Our model predicts that the above five counties should be those that Home Depot should prioritize for new store construction and that San Francisco county should be the target of approximately 3 new HD stores, Union 2 stores, Sangamon 1 store, Tolland 1 store and Johnson 1 store.  



	 ```python
	 nolo = hdlo.loc[hdlo['Lcount'] == 0]
	nolo.drop(['areaname','county','state','r1', 'r2', 'HD_dummy', 'Lo_dummy','HDcount','Lcount'], axis = 1, inplace = True)
	answers = zip(hdlo.loc[hdlo['Lcount'] == 0]['areaname'], list(lr2.predict(nolo)))
	sorted(answers, key = lambda x: x[1], reverse = True)[:5]
	```
	
	Output:
	[('Westchester', 3.1453938279743636),
	 ('Dane', 2.716737183882178),
	 ('Denver', 2.3201024980996396),
	 ('Newton', 2.073841064867157),
	 ('Essex', 1.9865756377144366)]

	 Our model predicts that the above five counties should be those that Lowe's should prioritize for new store construction and that Westchester county should be the target of approximately 3 new HD stores, Dane 3 stores, Denver 2 stores, Newton 2 stores and Essex 2 stores.


	 **Second way - logistic regression**

	 ```python
	 from sklearn.model_selection import train_test_split
	#preparing our data 
	hd_x = hdlo.copy()
	hd_y = hdlo.HD_dummy
	lo_x = hdlo.copy()
	lo_y = hdlo.Lo_dummy
	hd_x.drop(['areaname','county','state','r1', 'r2', 'HD_dummy', 'Lo_dummy','HDcount','Lcount'], axis = 1, inplace = True)
	lo_x.drop(['areaname','county','state','r1', 'r2', 'HD_dummy', 'Lo_dummy','HDcount','Lcount'], axis = 1, inplace = True)
	hdxtrain, hdxtest, hdytrain, hdytest = train_test_split(hd_x, hd_y, test_size = 0.3)
	loxtrain, loxtest, loytrain, loytest = train_test_split(lo_x, lo_y, test_size = 0.33)
	#training our models 
	from sklearn.linear_model import LogisticRegressionCV
	lg1 = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', max_iter=100, random_state=24)
	lg2 = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', max_iter=100, random_state=24)
	lg1.fit(hdxtrain, hdytrain)
	lg2.fit(loxtrain, loytrain)
	#predict our test data using these fitted models 
	lg1.score(hdxtest, hdytest)
	lg2.score(loytest, loytest)
	#since these scores are rather good we can go ahead with training on our full data 
	lg3 = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', max_iter=100, random_state=24)
	lg4 = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', max_iter=100, random_state=24)
	lg3.fit(hd_x, hd_y)
	lg4.fit(lo_x, lo_y)
	#adding predicted probabilities to our dataframes to show areas with highest predicted store presence in areas where 
	# store count = 0 
	hd_x2 = hd_x.assign(HD_dummy = hdlo.HD_dummy)
	hd_x2 = hd_x2.assign(areaname = hdlo.areaname)
	hd_x2 = hd_x2.loc[hd_x2['HD_dummy'] == 0]
	hd_x2areaname = hd_x2.areaname
	hd_x2.drop(['HD_dummy','areaname'], axis = 1, inplace = True)
	hd_predictions = [x[1] for x in lg1.predict_proba_(hd_x2)]
	hdanswers = list(zip(hd_x2areaname, list(hd_predictions)))
	sorted(hdanswers, key = lambda x: x[1])[:5]
	#doing same for Lowes
	lo_x2 = lo_x.assign(Lo_dummy = hdlo.Lo_dummy)
	lo_x2 = lo_x2.assign(areaname = hdlo.areaname)
	lo_x2 = lo_x2.loc[lo_x2['Lo_dummy'] == 0]
	lo_x2areaname = lo_x2.areaname
	hd_x2.drop(['Lo_dummy','areaname'], axis = 1, inplace = True)
	lopredictions = [x[1] for x in lg4.predict_proba_(lo_x2)]
	zippedanswer = list(zip(lo_x2areaname, list(lopredictions)))
	sorted(zippedanswer, key = lambda x: x[1], reverse = True)[:5]
	```


	1. **Explain the rationale for your decision**

		As can be seen from the top five predicted counties by our logistic regression model we get a similar answer to our linear regression model, with 3 of the predicted counties from our linear regression figuring in the top 5 results for our logistic regression model, and all top5 counties for each model figuring in the top10 predicted counties by the other. Tool Time should therefore build new stores in counties that should be high on the list for both Home Depot and Lowe's given their similarity in target demographic customer base and that do not yet have any Lowe's or Home Depot stores built in them. 


1. **Where should “Tool Time” build its next 5 stores based on the Census Data on your
customers.**
	1. **Explain the rationale for your decision**

1. **Using the realator.com market hotness index report , create an additional variable to
segment the country into the following regions:**
	1. **Region 1 – NorthEast** 
		**1. Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Vermont, New Jersey, New York, and Pennsylvania**
	1. **Region 2 – MidWest**
		**1. Illinois, Indiana, Michigan, Ohio, Wisconsin, Iowa, Kansas, Minnesota, Missouri, Nebraska, North Dakota, and South Dakota**
	1. **Region 3 – South**
		1. **Delaware, Florida, Georgia, Maryland, North Carolina, South Carolina, Virginia, District of Columbia, and West Virginia, Alabama, Kentucky, Mississippi, Tennessee, Arkansas, Louisiana, Oklahoma, and Texas**
	1. **Region 4 – West**
		1. **Arizona, Colorado, Idaho, Montana, Nevada, New Mexico, Utah, and Wyoming, Alaska, California, Hawaii, Oregon, and Washington**

	1. **Exploratory Data Analysis for realator.com data**
		1. **Which Region of the country has the best “Demand Score”**

			```python
				# Keep only relevant columns from state_region
				state_region = pd.read_csv('state_region.csv', sep = ',')
				state_region.drop(['State', 'Division'], axis=1, inplace=True)
				#reading in our data
				rdc = pd.read_csv('RDC_MarketHotness_Monthly.csv', sep = ',')
				rdc['town'], rdc['state'] = rdc['ZipName'].str.split(',').str[0], rdc['ZipName'].str.split(', ').str[1]
				df1 = rdc.merge(state_region, how = 'left', left_on = 'state', right_on = 'State Code')
				df1.drop('State Code', axis = 1, inplace = True)
				df1.groupby('Region')['Demand Score'].mean()
			```
			
		1. **Which State in the country has the best “Demand Score”**

			```python
			df1.groupby('state')['Demand Score'].aggregate({'mean'}).sort_values(by = 'mean', ascending = False)[:3]
			```

			Massachusetts has the best mean "Demand Score."

		1. **Which metro area (Pop_2010 > 1million) has the best “Demand Score”**

			```python
			#merging our data from hdlo to extract population for each country
			hdlo2 = hdlo[['county','pop_2010']]
			df2 = df1.merge(hdlo2, how = 'inner', left_on = 'CountyFIPS', right_on = 'county')
			df2[df2.pop_2010 > 1e6][['CountyName', 'state','Demand Score']][:10]
			```

			Middlesex, MA has the best "Demand Score" of metro areas.



	1. **Compare and contrast these findings with your predicted new store findings.**
		1. **Describe your findings as they relate to the customer attributes and potential
		business opportunity that Lowes, Home Depot and/or Tool Time may have if they
		are or are not located in the areas that have high demands for real estate
		opportunities.**

			It appears that the Northeast, Texas, Ohio, California, and Florida are all good areas for stores to be loacted, according to "Demand Score." This somewhat agrees with the model predictions; there were a lot of predicted locations in California and the Northeast.

	1. **Add the following as features to the original HDLo data set and predict again where Tool
	Time should build its next 5 stores.**
	a. Feature Engineering
	i. Median.Listing.Price
	ii. Demand.Score
	iii. Hotness.Score
	iv. Nieleson.HH.Rank

		```python
		rdc2 = rdc[['CountyFIPS','Nielsen HH Rank','Demand Score','Hotness Score','Median Listing Price']]
		hd_data = hdlo.merge(rdc2, how = 'inner', left_on = 'county', right_on = 'CountyFIPS')
		# Copy data into training and testing sets, drop earlier added probability columns along with other irrelevant columns
		hd_data = hd_data.copy()
		l_data = hd_data.copy()

		hd_data.drop(['areaname', 'county', 'state', 'r1', 'r2', 'Lcount', 'HDcount', 'Lexists'], axis=1, inplace=True)
		l_data.drop(['areaname', 'county', 'state', 'r1', 'r2', 'Lcount', 'HDcount', 'HDexists'], axis=1, inplace=True)

		hd_X = hd_data.drop(['HD_dummy'], axis=1)
		hd_y = hd_data.HD_dummy

		l_X = l_data.drop(['Lo_dumm7'], axis=1)
		l_y = l_data.Lo_dummy

		# Split the data into train and test portions to test accuracy
		hd_X_train, hd_X_test, hd_y_train, hd_y_test = train_test_split(hd_X, hd_y, random_state=42)
		l_X_train, l_X_test, l_y_train, l_y_test = train_test_split(l_X, l_y, random_state=42)

		# Train the models
		hd_logit = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', max_iter=1000, random_state=42).fit(hd_X_train, hd_y_train)
		l_logit = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', max_iter=1000, random_state=42).fit(l_X_train, l_y_train)

		# Home Depot store predicition accuracy
		hd_logit.score(hd_X_test, hd_y_test)

		# Lowes store predicition accuracy
		l_logit.score(l_X_test, l_y_test)

		# Train on the full data
		hd_full_logit = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', max_iter=1500, random_state=42).fit(hd_X, hd_y)
		l_full_logit = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', max_iter=1500, random_state=42).fit(l_X, l_y)

		# Predict the target for all locations and extract the probability that a store should be built
		hd_store_prob = [x[1] for x in hd_full_logit.predict_proba(hd_X)]
		l_store_prob = [x[1] for x in l_full_logit.predict_proba(l_X)]

		# Add the probability features to the original dataset so we can find the best next locations to build stores
		added_data['hd_store_prob'] = hd_store_prob
		added_data['l_store_prob'] = l_store_prob

		# Create new column that is the sum of the two probability columns for sorting purposes
		added_data['prob_sum'] = added_data.hd_store_prob + added_data.l_store_prob

		# For Tool Time, show only locations where there are 1 or no stores for HD and Lowes,
		# sort by prob_sum descending and show top 5 areas predicted to be the best store locations
		added_data[(added_data.Lcount <= 1) & (added_data.HDcount <= 1)].sort_values(by='prob_sum', ascending=False).head(10)
		```

	5. **What are the top 5 new area names for which Tool Time should build their stores?**
		The first four stores remain the same, however Ottawa, MI overtook Ingham, MI in this version of the model.

	6. **Do these features increase the prediction accuracy for the new area predictions?**
		Based on the new scores, the new features seem to have decreased the accuracy of the models.
	7. **Does overlaying the realtor data set add value to the business strategy of Tool Time?**
		In this case, I would conclude that adding the new features **does not** add value to the business strategy that I would recommend.

	8. **Is there an alternative strategy that Tool Time should explore other than Census Data
	and Realtor data?**
		Tool Time might want to consider areas that have high demand for commerical real estate, as new businesses will have contractors in need of supplies. Tool Time might also want to look into trends for up-and-coming neighborhoods where real estate prices may increase in the future, and get ahead of competitors by opening stores newly desired neighborhoods.

				



