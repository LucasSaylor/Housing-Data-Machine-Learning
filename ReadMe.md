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
