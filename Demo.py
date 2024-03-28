import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor


model = joblib.load('diamond_random_forest_regressor.pkl')

st.set_page_config(
	page_title="Diamond Prices 2022",
	page_icon = ":Gem:"
)

page = st.radio("Choose a page:",
		["Demo", "Explanation"], captions = ["Small demo app.","Written explanation on the creation of the model."])

if page =="Demo":
	st.header("💎 Diamond Price Prediction")
	st.write("Enter diamond characteristics below to get a price prediction:")
	
	depth = st.slider("Depth:", 43,79,30)
	table = st.slider("Table:", 43,95,30)

	col1, col2, col3 = st.columns(3)

	with col1:
		color_options = ["D","E","F","G","H","I","J"]
		color = st.selectbox("Colour of the diamond:",color_options)
	with col2:
		cut_options = ["Fair", "Good", "Ideal","Premium","Very Good"]
		cut = st.selectbox("Cut of the diamond:",cut_options)
	with col3:
		clarity_options = ["I1", "IF","SI1","SI2","VS1","VS2","VVS1","VVS2"]
		clarity = st.selectbox("Clarity of the diamond:", clarity_options)
	
	carat = st.slider("Carats:",
		0.2,5.0,2.0)
	
	st.write("Dimmensions (mm):")
	col1,col2,col3 = st.columns(3)

	with col1:
		X = st.slider("X dimmension:",
			1.0,11.0,2.0)

	with col2:
		Y = st.slider("Y dimmension:",
			1.0,58.0,2.0)

	with col3:
		Z =st.slider("Z dimmension:",
			1.0,31.0,2.0)

	if st.button("Predict Price"):
		cut_encoded = {f'cut_{option}': 0 for option in cut_options}
		cut_encoded[f'cut_{cut}'] = 1
		color_encoded = {f'color_{option}': 0 for option in color_options}
		color_encoded[f'color_{color}'] = 1
		clarity_encoded = {f'clarity_{option}': 0 for option in clarity_options}
		clarity_encoded[f'clarity_{clarity}'] = 1
		
		other_data= {
			"carat":carat,
			"depth": depth,
			"table":table,
			"x":X,
			"y":Y,
			"z":Z
		}

		all_data = {**other_data, **cut_encoded, **color_encoded, **clarity_encoded}

		df = pd.DataFrame([all_data])

		st.write(f"${int(model.predict(df))}")
		#predictions = model.predict(carrat,depth,table,x,y,z,)




elif page == "Explanation":
	st.header("Creation process for the model:")
	st.subheader("Step 1: Obtain and load the dataset.")
	data_link ="https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices"
	st.write("Link: [Diamond Prices 2022](%s)"%data_link)
	df = pd.read_csv("Diamonds Prices2022.csv", index_col=0)
	st.write(df.head())

	st.subheader("**Step 2: Explore the data**")
	st.write("Checking the distribution of categorical data columns with pie charts.")

	cuts = ("Fair", "Good", "Ideal", "Premium", "Very Good")
	cutsQty = []
	for c in cuts:
		cutsQty.append(df[df['cut'] == f'{c}'].count()[0])
	
	colors = ("D","E","F","G","H","I","J")
	colorQty = []
	for c in colors:
		colorQty.append(df[df['color'] == f"{c}"].count()[0])
	
	clarity = ("SI1", "SI2", "VS1", "VS2", "WS1", "WS2", "I1", "IF")
	clarityQty = []
	for c in clarity:
		clarityQty.append(df[df['clarity'] == f"{c}"].count()[0])

	fig,(ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,6))
	fig.suptitle('Categorical Data Distributions')
	ax1.pie(cutsQty, labels= cuts, autopct='%.2f%%', explode=[0.1,0.1,0,0,0.1])
	ax1.set_title('Cuts Distribution')
	ax2.pie(colorQty, labels= colors, autopct='%.2f%%', explode=[0,0,0,0,0.1,0.1,0.1])
	ax2.set_title('Color Distribution')
	ax3.pie(clarityQty, labels= clarity, autopct='%.2f%%', explode=[0,0,0,0,0.1,0.1,0.1,0.1])
	ax3.set_title('Clarity Distribution')
	st.pyplot(fig)

	st.write("Correlation matrix of non-categorical data.")
	fig, ax = plt.subplots()
	sns.heatmap(df[["z","y","x","price","table","depth","carat"]].corr(),annot=True)
	st.write(fig)

	st.write("One-hot encode the categorical data for later use and drop the old columns.")

	st.code('''df = pd.read_csv("Diamonds Prices2022.csv", index_col=0)
df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'])
df.head()''')

	df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'])
	st.table(df.head())

	st.subheader("Step 3: Spilt the data in to train and test  samples.")
	st.code('''X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)''')

	st.subheader("Step 4: Import and Train models.")
	st.write("Use SKlearn to import regression models for testing.")
	st.code('''from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression''')

	st.write("Define the models and a random state for reproducibility.")
	st.code('''forest = RandomForestRegressor(random_state=42)
gradient = GradientBoostingRegressor(random_state=42)
linear = LinearRegression()''')

	st.write("Train the models on the training data.")
	st.code('''forest.fit(X_train,y_train)
gradient.fit(X_train,y_train)
linear.fit(X_train, y_train)''')

	st.write("Use the models to make predictions.")
	st.code('''forest_predictions = forest.predict(X_test)
gradient_predictions = gradient.predict(X_test)
linear_predictions = linear.predict(X_test)''')

	st.subheader("Evaluate the models")
	st.code('''from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
for i, (pred, model) in enumerate(zip([forest_predictions, gradient_predictions, linear_predictions], ['Forest', 'Gradient', 'Linear']), start=1):
	print(f"Scores for {model}:")
	print(f"MAE: {mean_absolute_error(y_test, pred)}")
	print(f"MAPE: {mean_absolute_percentage_error(y_test, pred)}")''')

	col1,col2,col3 = st.columns(3)

	with col1:
		st.write("**Scores for Forest:**")
		st.write("MAE: 280.37")
		st.write("MAPE: 0.06")

	with col2:
		st.write("**Scores for Gradient:**")
		st.write("MAE: 411.89") 
		st.write("MAPE: 0.12")

	with col3:
		st.write("**Scores for Linear:**")
		st.write("MAE: 746.82")
		st.write("MAPE: 0.38")

	st.write("The score for the models show that the Random Forest Regressor model performs the best on the test data with a Mean Absolute Error of 280.37 meaning the average prediction was only off by 6%.")

	st.subheader("Step 5: Save the Forest model.")
	st.code('''import joblib
joblib.dump(forest, 'diamond_random_forest_regressor.pkl')''')

	st.subheader("Conclusion:")
	st.write("The best scoring model was the random forest regressor with a MAE of 280.37. This model was then saved to be used on the app. Try the demo by selecting demo at the top of the page.")
