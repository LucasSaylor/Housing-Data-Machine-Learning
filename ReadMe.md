## Introduction 

Houses are all around the world. Depending on where you live tiny apartments in one geographic region can cost as much as mansions in another region. Understanding housing prices and what factors influence these prices can be an extremely helpful asset when trying to evaluate the potential value of a home. Machine learning can be utilized by companies like real estate brokers or other large cooperation’s when buying and selling homes. This GitHub repository will evaluate how housing prices can be predicted through machine learning by using housing and property values. 

```python
```

## Data

The data being used to determine housing prices comes from Realitor.com through Rapid API. Rapid API provides Python Requests code where the user can input variables such as city, state, area radius, and other housing factors. When applied in Python the Python Request outputs a json file. For you to use this API you will need to create an account with Rapid API to obtain an API key.

```python
import requests

url = "https://realtor.p.rapidapi.com/properties/v2/list-sold"

querystring = {"radius":"20","sort":"sold_date","city":"Vancouver","offset":"0","state_code":"WA","limit":"500"}

headers = {
    'x-rapidapi-host': "realtor.p.rapidapi.com",
    'x-rapidapi-key': " Rapid API Key"
    }

response = requests.request("GET", url, headers=headers, params=querystring)

```

To unpack this Json file you need to use a json_normalize which converts the json file into a pandas dataframe.

```python
import json
from pandas.io.json import json_normalize
import pandas as pd

response_data = json.loads(response.text)
df = json_normalize(response_data['properties'])
```

Pull out the desired housing features.

```python
buildings = df[['prop_type','year_built','beds','baths_full','garage','baths_half','price','baths','address.city','address.state','address.postal_code','address.county','lot_size.size','building_size.size']]
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prop_type</th>
      <th>year_built</th>
      <th>beds</th>
      <th>baths_full</th>
      <th>garage</th>
      <th>baths_half</th>
      <th>price</th>
      <th>baths</th>
      <th>address.city</th>
      <th>address.state</th>
      <th>address.postal_code</th>
      <th>address.county</th>
      <th>lot_size.size</th>
      <th>building_size.size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>condo</td>
      <td>2020.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>254900</td>
      <td>1</td>
      <td>Portland</td>
      <td>Oregon</td>
      <td>97217</td>
      <td>Multnomah</td>
      <td>NaN</td>
      <td>651.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>single_family</td>
      <td>2008.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>330000</td>
      <td>3</td>
      <td>Vancouver</td>
      <td>Washington</td>
      <td>98661</td>
      <td>Clark</td>
      <td>2614.0</td>
      <td>1558.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>single_family</td>
      <td>1995.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>NaN</td>
      <td>350000</td>
      <td>2</td>
      <td>Vancouver</td>
      <td>Washington</td>
      <td>98682</td>
      <td>Clark</td>
      <td>6970.0</td>
      <td>1201.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>single_family</td>
      <td>1973.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>320000</td>
      <td>1</td>
      <td>Washougal</td>
      <td>Washington</td>
      <td>98671</td>
      <td>Clark</td>
      <td>5227.0</td>
      <td>1041.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>condo</td>
      <td>2013.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>535000</td>
      <td>3</td>
      <td>Portland</td>
      <td>Oregon</td>
      <td>97211</td>
      <td>Multnomah</td>
      <td>2614.0</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>195</td>
      <td>single_family</td>
      <td>1923.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>640000</td>
      <td>2</td>
      <td>Portland</td>
      <td>Oregon</td>
      <td>97211</td>
      <td>Multnomah</td>
      <td>4792.0</td>
      <td>1912.0</td>
    </tr>
    <tr>
      <td>196</td>
      <td>single_family</td>
      <td>2007.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>430000</td>
      <td>3</td>
      <td>Portland</td>
      <td>Oregon</td>
      <td>97203</td>
      <td>Multnomah</td>
      <td>2614.0</td>
      <td>1380.0</td>
    </tr>
    <tr>
      <td>197</td>
      <td>single_family</td>
      <td>1914.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>NaN</td>
      <td>470000</td>
      <td>2</td>
      <td>Portland</td>
      <td>Oregon</td>
      <td>97203</td>
      <td>Multnomah</td>
      <td>4792.0</td>
      <td>1738.0</td>
    </tr>
    <tr>
      <td>198</td>
      <td>condo</td>
      <td>2004.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>274000</td>
      <td>3</td>
      <td>Portland</td>
      <td>Oregon</td>
      <td>97233</td>
      <td>Multnomah</td>
      <td>2178.0</td>
      <td>1236.0</td>
    </tr>
    <tr>
      <td>199</td>
      <td>single_family</td>
      <td>1955.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>363000</td>
      <td>2</td>
      <td>Portland</td>
      <td>Oregon</td>
      <td>97230</td>
      <td>Multnomah</td>
      <td>7405.0</td>
      <td>1080.0</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 14 columns</p>
</div>

### Data Cleaning

To clean the data the means of each numeric column was taken and were inserted for all the missing values. The only column that did not have the mean applied was the garage feature because either a home has a garage or it does not.

```python
X['year_built'].fillna(means.loc['mean'][0],inplace=True)
X['beds'].fillna(means.loc['mean'][1],inplace=True)
X['baths_full'].fillna(means.loc['mean'][2],inplace=True)
X['garage'].fillna(0,inplace=True)
X['baths_half'].fillna(means.loc['mean'][1],inplace=True)
X['lot_size.size'].fillna(means.loc['mean'][6],inplace=True)
X['building_size.size'].fillna(means.loc['mean'][7],inplace=True)

```

The next step for the data cleaning was to sepearte the numeric and catigorical columns. 

```python
X_num = X[X.select_dtypes(exclude='object').columns]
X_cat = X[X.select_dtypes('object').columns]
```

The categorical variables are then replaced with dummy variables to make it easier to perform data analysis.

```python
X_cat = pd.get_dummies(X_cat)

```

Once the dummy variables have been created and the numeric missing values filled with the means the dataframes are combined for a clean dataframe that can be used for data analysis.

```python
X_clean
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year_built</th>
      <th>beds</th>
      <th>baths_full</th>
      <th>baths_half</th>
      <th>baths</th>
      <th>lot_size.size</th>
      <th>building_size.size</th>
      <th>prop_type_condo</th>
      <th>prop_type_farm</th>
      <th>prop_type_land</th>
      <th>...</th>
      <th>address.postal_code_98665</th>
      <th>address.postal_code_98671</th>
      <th>address.postal_code_98682</th>
      <th>address.postal_code_98683</th>
      <th>address.postal_code_98684</th>
      <th>address.postal_code_98685</th>
      <th>address.postal_code_98686</th>
      <th>address.county_Clark</th>
      <th>address.county_Columbia</th>
      <th>address.county_Multnomah</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2020.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.252525</td>
      <td>1</td>
      <td>15335.491713</td>
      <td>651.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2008.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.000000</td>
      <td>3</td>
      <td>2614.000000</td>
      <td>1558.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1995.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.252525</td>
      <td>2</td>
      <td>6970.000000</td>
      <td>1201.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1973.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.252525</td>
      <td>1</td>
      <td>5227.000000</td>
      <td>1041.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2013.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.252525</td>
      <td>3</td>
      <td>2614.000000</td>
      <td>2018.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>195</td>
      <td>1923.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.252525</td>
      <td>2</td>
      <td>4792.000000</td>
      <td>1912.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>196</td>
      <td>2007.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.000000</td>
      <td>3</td>
      <td>2614.000000</td>
      <td>1380.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>197</td>
      <td>1914.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.252525</td>
      <td>2</td>
      <td>4792.000000</td>
      <td>1738.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>198</td>
      <td>2004.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.000000</td>
      <td>3</td>
      <td>2178.000000</td>
      <td>1236.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>199</td>
      <td>1955.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.252525</td>
      <td>2</td>
      <td>7405.000000</td>
      <td>1080.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 68 columns</p>
</div>

## Methods and Results

### Machine Learning Models:

For this project the four machine learning models used were Linear Regression, Lasso Regression, K Nearest Neighbors, and Decision Tree Regression. These four models were chosen because Linear Regression allow for basic unweighted regression. Lasso regression was used because it has weights on each variable. K Nearest Neighbors was used beacuse of its ability to use values close to eachother to find the predicted value. Decision Tree Regression was used because the predictor variables can be split to make an accurate regression model. 

To create the models the data was split into training and testing data. The machine leanring models were fit on the training data and then the training and testing data were used as preictors to see how well the model predicted home prices. 20% of the data was training data and the other 80% was testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(X_clean,y, test_size=.2, random_state=42)
```

The two steps that were used to improve the model was introducing additional polynomial features and using a standard scaler to transform the data.

The two Explanation of Model evaulation metics used were r^2 and mse. The higher r^2 the better the model predicted the data. Also, the lower the mse the better the model represented the data.

#### Linear Regression:

```python
          Origional Data                 Added Polynomial Features             Standared Scaler
          
MSE Training: 5600321159.9145355    MSE Training: 81.14440552575785     MSE Training: 5728882144.65625  
MSE Testing: 4791021560.331575      MSE Testing: 1785485700055740.2     MSE Testing:  1.7362469560887756e+38
r^2 Training: 0.8910820717250114    r^2 Training: 0.9999999984218618    r^2 Training: 0.8885817515263617
r^2 Testing: 0.7344440246954169     r^2 Testing: -98964.61526595881     r^2 Testing:   -9.623641805566386e+27
```

#### Lasso Regression:

```python
          Origional Data                 Added Polynomial Features             Standared Scaler
          
MSE Training: 5678272064.038322    MSE Training: 398994343.39644694     MSE Training: 5605558479.719356
MSE Testing:  4036891793.971036    MSE Testing:  25413352816.033394     MSE Testing:  4426826515.345859
r^2 Training: 0.8895660424220679   r^2 Training: 0.9922401526563802     r^2 Training: 0.8909802136339188
r^2 Testing: 0.7762438085390594    r^2 Testing: -0.4086072475019631     r^2 Testing:  0.754630569288129
```

#### K Nearest Neighbors Regression:

```python
          Origional Data                 Added Polynomial Features             Standared Scaler
          
MSE Training: 31990717279.330566    MSE Training: 33291732617.879875      MSE Training: 31371522027.783813
MSE Testing:  9173943875.64425      MSE Testing:  25413352816.033394      MSE Testing:  10571846691.244501
r^2 Training: 0.37782806546245684   r^2 Training: 0.35252524955556763     r^2 Training: 0.38987049340014535
r^2 Testing: 0.49150810894751795    r^2 Testing:  0.36471929727067265     r^2 Testing:  0.4140253756925969
```

#### Decision Tree Regression:

```python
          Origional Data                 Added Polynomial Features             Standared Scaler
          
MSE Training: 11452806944.283335   MSE Training: 10110095308.092709     MSE Training: 11452806944.283335
MSE Testing:  13816323024.66889    MSE Testing:  10498149273.08941      MSE Testing:  13771299413.55778
r^2 Training: 0.7772599160502824   r^2 Training: 0.803373663013829      r^2 Training: 0.7772599160502824
r^2 Testing: 0.23419106139752144   r^2 Testing:  0.41811026437638954    r^2 Testing:  0.23668662289933984
```
