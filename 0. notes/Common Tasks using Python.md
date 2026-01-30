#python [[Python]]

#### Convert csv to parquet [[Data Engineering]]
```
import pyarrow.parquet as paq
import pyarrow as pa


for file in file_names:
	df = pd.read(f + '.csv')
	table = pa.Table.from_pandas(df)
	pq.write_table(table, f+'.parquet')
```


#### One hot encoding a binary variable [[ML Concepts]] [[Encoding Categorical Data]]

```
df["class"] = (df["class"] == "g").astype(int)
```

#### Filter a dataframe basis one column and get corresponding values for another column


```
df[df["class"] == 1][label]
```


### Synthetic Data Generation [[Training Pipeline]]

```
def generate_data(n_samples, separation #optional):
	centers = [(0,0), (separation, separation)]
	X,y = [], []

	for i, center in enumerate(centers):
		data = np.random.randn(n_samples, 2) + center
		X.append(data)
		y += [i]*n_samples
	return np.vstack(X), np.array(y)
```

### Feature Selection using .iloc [[Preprocessing]]

```
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
```


### Drop pandas column [[Preprocessing]]

```
df= df.drop('column_name', axis=1)
```


#### Set precision for print [[numpy]]
```
np.set_printoptions(precision=2)
```


#### Generating a grid [[numpy]]
```
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = np.reshape(len(X_grid), 1)
```

#### Create new virtual env
```
python -m venv .pyspark-env
```


#### Setting environment variables:
```
setx JAVA_HOME "C:\Program Files\Java\jdk-11"
echo JAVA_HOME
```

#### Add items to path
```
setx PATH "%PATH%; %JAVA_HOME%\bin"
```



#### Open PDF
```
import pdfplumber
from ipywidgets import widgets
from io import BytesIO

upload_widget = widgets.FileUpload(
    accept='.pdf', 
    multiple=False,
    description='Upload PDF',
    layout=widgets.Layout(width='300px',height = '100px', border='2px dashed #cccccc', padding='10px')
)

if upload_widget.value:
        # Extract the first uploaded file
        uploaded_file = list(upload_widget.value)[0]
        pdf_file = uploaded_file['content']

        # Extract text from the PDF
        try:
            with pdfplumber.open(BytesIO(pdf_file)) as pdf:
                extracted_text = "\n".join(page.extract_text() for page in pdf.pages)
        except Exception as e:
	        print('No File Found')
```