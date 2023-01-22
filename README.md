#  📋Checklist-4-ML-E2E-Projects📋
Checklist for ML projects. An almost incomplete collections of MLOPs bullet points. This list serves three purpouses:
- Provides a checklist for things that are obvious but are really done or mostly forgotten
- Provides a step-by-step guide to ML project
- Provides links and references
***

## Scoping
<details>
<summary>Expand ⬇️</summary>
<br>

- ❓ What is the project main objective?
- ❓ Which part of the main objective a ML model is addressing?
- 📈📉 Establish a [baseline](https://blog.ml.cmu.edu/2020/08/31/3-baselines/) against which your ML will be considered an successful (an improvement against the baseline)
- ❓ Are there any solutions not based on a ML model? You are likely to be asked to compared your method against some no-ML model!
- Choose: KPIs	(key performance indicators)
- 📈 Monitor your project's objective over time
- 🗣️ Talk to the domain expertes, they are those with the domain knowledge 

</details>

## Data
<details>
<summary>Expand ⬇️</summary>
<br>

- Data sourcing/collection/ingestion
    - Collect your data from the web via scraping
    - Build your own dataset
    - Create/augment your data with some synthetic data generation techniques
    - Dowload some open source. Best resource is probably [Kaggle](https://www.kaggle.com/)
- Data versioning
- Data cleaning
- Data labeling
- Establish a data schema which helps validate the data. Especially for [concept drift](https://machinelearningmastery.com/gentle-introduction-concept-drift-machine-learning/)
- Data storage
    - Structured: SQL
    - Unstructured: NoSQL
- Data transformation
- EDA (Exploratory Design Analysis)
- Build an ETL (Extra, Transform & Load) pipeline
    
</details>

***

## Programming
<details>
<summary>Expand ⬇️</summary>
<br>

- Code release:
    - Major
    - Minor
    - Patch
- Code versionning:
    - [GitHub](https://github.com/)
    - [GitLab](https://about.gitlab.com/)
- Code optimisation
    - Refactoring
    - Profilers
    - Caching
- Code testing
    - Unittesting
- Code obfuscation
    - Serialisation vith Cython
- Code shipping
    - Containersition Docker image 
    
</details>

***

## Modelling
<details>
<summary>Expand ⬇️</summary>
<br>
 
- Feature(s) vs. target(s)
- Model versioning
-  🐣 Create a baseline model
- Keep track of your model dependencies
- Feature selection
- Feature engineering
- Model metrics (Not model objective function and not necessarily KPIs!)    
- Model CV (Cross Valisation)
- Model hyperparameters:
    - Grid search
    - Successive halving
    - BOHB
- Model inference:
    - on CPUs
    - on GPUs
    - on TPUs
- Model serialisation (aka model persistence) / deserialisation
    - joblib
    - pickle
    - skpops
    - ONNX
    
</details>

***
