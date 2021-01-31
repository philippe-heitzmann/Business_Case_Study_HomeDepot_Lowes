# Business Case Study - Analyzing and predicting new store locations for Lowe's and Home Depot

### In this case study, we will be put in the shoes of a consultant looking to understand some of the factors that Lowe's and Home Depot look to when deciding to build new stores. We will then use this strategy to advise a new competitor we will call 'Tool Time' on where this new entrant should build its next five stores based on this information on Home Depot and Lowe's.

##### This case study will be split into the following subsections:

1. Exploratory Data Analysis & creating a map of stores
1. Identifying relationships between variables with linear regression model 
1. Leveraging linear regression model to perform _new store location predictions_ for Lowe's, Home Depot and Tool Time 
1. Comparing our results to realtor.com 'market hotness' data 


## Questions:
1. **Perform Exploratory Data Analysis on the stores**
	1. What are the total store counts of Home Depot and Lowes?
```python
reading in our data 
hdlo = pd.read_csv("Home_Depot_Lowes_Data.csv", sep = ',')
print('There are a total of {} Lowe\'s stores in this dataset'.format(np.sum(hdlo.Lcount)))
```
		1. Create one dummy variable for Home Depot and one dummy variable for Lowes
that identifies whether or not the store is located in each county
		1. Which store is present in more counties?

1. **Use a United States map with FIPS locations to plot the store locations of both Lowes
and Home Depot**
	1. What observations can you make from the map?

1. **Create a linear regression model to identify the correlations among the variables.**
	1. What customer demographic variables are most import to Lowes?
	1. What customer demographic variables are most import to Home Depot?
	1. How are the chains similar in their decision making?
	1. How are they different?

1. **What are the top 5 towns/cities that can be predicted as potential candidates for new
locations for both Lowes and Home Depot?**
	1. Explain the rationale for your decision

1. **Where should “Tool Time” build its next 5 stores based on the Census Data on your
customers.**
	1. Explain the rationale for your decision

