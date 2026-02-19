# Build an ML Pipeline for Short-Term Rental Prices in NYC
This projectis forked from Udacity Academic project and added missing component to complete the ML Pipeline and monitored on W&B in personal account

Objective: Working for a property management company renting rooms and properties for short periods of time on various rental platforms. Task is to estimate the typical price for a given property based 
on the price of similar properties and new data is recieved in bulk every week. The model retrains with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project which built such a reusable pipeline.
W&B: https://wandb.ai/kunal-chandmare/nyc_airbnb
Github: https://github.com/kunalchandmare/build-ml-pipeline-for-short-term-rental-prices

## Table of contents

- [Introduction](#build-an-ml-pipeline-for-short-term-rental-prices-in-nyc)
- [Preliminary steps](#preliminary-steps)
- [Implementation Steps](#implementation-steps)
  * [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  * [Data cleaning](#data-cleaning)
  * [Data testing](#data-testing)
  * [Data splitting](#data-splitting)
  * [Train Random Forest](#random-forest-training)
  * [Optimize hyperparameters](#hyperparameter-optimization)
  * [Select the best model](#best-model-selection)
  * [Visualize the pipeline](#visualize-the-pipeline)
- Licence 
## Preliminary steps

### Supported Operating Systems

This project is compatible with the following operating systems:

- Windows
Note: Due to limitation on available Conda version and Network access , workaround was implemented for runtime environment creation, which shouold work without such workaround without such restrictions out of the boc

Please ensure you are using one of the supported OS versions to avoid compatibility issues.

### Python Requirement

This project requires **Python 3.13**. Please ensure that you have Python 3.13 installed and set as the default version in your environment to avoid any runtime issues.

### The Starter kit
Go to [(https://github.com/kunalchandmare/build-ml-pipeline-for-short-term-rental-prices)](https://github.com/kunalchandmare/build-ml-pipeline-for-short-term-rental-prices)

```
git clone https://github.com/[your github username]/build-ml-pipeline-for-short-term-rental-prices.git
```

and go into the repository:

```
cd build-ml-pipeline-for-short-term-rental-prices
```
Commit and push to the repository often while you make progress towards the solution. Remember 
to add meaningful commit messages.

### Create environment
Make sure to have conda installed and ready, then create a new environment using the ``merged_conda.yml``
file provided in the root of the repository and activate it:

```bash
> conda env create -f merged_conda.yml
> conda activate nyc_airbnb_dev
```

### Get API key for Weights and Biases
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```

### Cookie cutter
In order to make job a little easier, a cookie cutter template that you one use to create stubs for new pipeline components. It is not required to use this, but it might save a bit of 
boilerplate code. It creates a new component including the `conda.yml` file, the `MLproject` file as well as the script. One can then modify these as needed, instead of starting from scratch.
This template code is used as it is from Udacity repository. Usefull for future projects too

### The configuration
As usual, the parameters controlling the pipeline are defined in the ``config.yaml`` file defined in
the root of the starter kit. We will use Hydra to manage this configuration file. 
Open this file and get familiar with its content. Remember: this file is only read by the ``main.py`` script 
(i.e., the pipeline) and its content is
available with the ``go`` function in ``main.py`` as the ``config`` dictionary. For example,
the name of the project is contained in the ``project_name`` key under the ``main`` section in
the configuration file. It can be accessed from the ``go`` function as 
``config["main"]["project_name"]``.

NOTE: do NOT hardcode any parameter when writing the pipeline. All the parameters should be 
accessed from the configuration file.

### Running the entire pipeline or just a selection of steps
In order to run the pipeline when you are developing, you need to be in the root of the starter kit, 
then you can execute as usual:

```bash
>  mlflow run .
```
This will run the entire pipeline.

When developing it is useful to be able to run one step at the time. Say you want to run only
the ``download`` step. The `main.py` is written so that the steps are defined at the top of the file, in the 
``_steps`` list, and can be selected by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=download
```
If you want to run the ``download`` and the ``basic_cleaning`` steps, you can similarly do:
```bash
> mlflow run . -P steps=download,basic_cleaning
```
You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a ``hydra_options`` parameter. For example, say that we want to set the parameter
modeling -> random_forest -> n_estimators to 10 and etl->min_price to 50:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```

### Pre-existing components
In order to simulate a real-world situation, we are providing you with some pre-implemented
re-usable components. While you have a copy in your fork, you will be using them from the original
repository by accessing them through their GitHub link, like:

```python
_ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
```
where `config['main']['components_repository']` is set to 
[https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices#components](https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices/tree/main/components).
You can see the parameters that they require by looking into their `MLproject` file:

- `get_data`: downloads the data. [MLproject](https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices/blob/main/components/get_data/MLproject)
- `train_val_test_split`: segrgate the data (splits the data) [MLproject](https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices/blob/main/components/train_val_test_split/MLproject)

## Implementation Steps
### Exploratory Data Analysis (EDA)

I started by running only the data download step:

```bash
mlflow run . -P steps=download

```markdown

Then started the EDA notebook:

```bash
mlflow run src/eda
```

In the notebook I fetched `sample.csv:latest` from W&B, loaded it with pandas and created a pandas-profiling report.

**Key observations:**
- Missing values in multiple columns
- `last_review` stored as string (should be datetime)
- `price` contained many outliers (zeros + extremely high values)

After stakeholder discussion, I kept only prices between **$10** and **$350**.

Applied fixes in the notebook:

```python
idx = df['price'].between(10, 350)
df = df[idx].copy()
df['last_review'] = pd.to_datetime(df['last_review'])
```

Re-checked with profiling / `df.info()` — main issues were resolved.  
Finished the W&B run and closed Jupyter.

### Data Cleaning 

Created the step using cookiecutter:

```bash
cookiecutter cookie-mlflow-step -o src
```

Parameters defined:
- str: `input_artifact`, `output_artifact`, `output_type`, `output_description`
- float: `min_price`, `max_price`

In `run.py` I implemented:
1. Download input artifact from W&B
2. Filter price using `args.min_price` and `args.max_price`
3. Convert `last_review` to datetime
4. Filter invalid geolocation:

```python
idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
df = df[idx].copy()
```

5. Save to `clean_sample.csv` (index=False)
6. Upload as new artifact

Added step to `main.py` using config values for min/max price and `sample.csv:latest` input.

### Data Testing

Tagged latest `clean_sample.csv` artifact in W&B → alias `reference`.

Added tests in `src/data_check/test_data.py`:

```python
def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000

def test_price_range(data, min_price, max_price):
    assert data['price'].between(min_price, max_price).all()
```

Included `data_check` in `main.py`:
- `csv`: `clean_sample.csv:latest`
- `ref`: `clean_sample.csv:reference`
- used config values (e.g. `kl_threshold`)

Verified with:

```bash
mlflow run . -P steps=data_check
```

→ tests passed successfully.

### Data Splitting

Added pre-built `train_val_test_split` component (from components repo) to `main.py`.

Used config parameters:
- `test_size`
- `random_seed`
- `stratify_by`

Run created two artifacts: `trainval_data.csv` and `test_data.csv`.

### Random Forest Training

Completed `src/train_random_forest/run.py`:
- Loaded `trainval_data.csv:latest`
- Preprocessed: TF-IDF on `name` + numerical features
- Trained `RandomForestRegressor` with `rf_config` parameters
- Logged MAE, R², feature importances plot and model artifact named `random_forest_export`

Integrated step into the main pipeline.

### Hyperparameter Optimization

Executed small grid search via Hydra:

```bash
mlflow run . -P steps=train_random_forest \
  -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```

All combinations tracked automatically in W&B.

### Best Model Selection

In W&B → Runs → Table view showed:  
`ID • max_depth • n_estimators • mae • r2`

Sorted by MAE ascending → selected lowest-MAE run.

Went to Artifacts → located `random_forest_export` → added tag `prod`.

### Visualize The Pipeline

In W&B → Artifacts → selected `prod`-tagged model → **Graph** view.

Diagram displays complete flow:  
download → cleaning → checks → split → training → model export


## License

[License](LICENSE.txt)
