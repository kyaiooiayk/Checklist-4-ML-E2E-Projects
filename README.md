#  📋Checklist-4-ML-E2E-Projects📋
Checklist for ML projects. An almost incomplete collections of MLOps bullet points. This list serves three purpouses:
- Provides a **checklist** for things that are obvious but are rarely done or effectively mostly forgotten
- Provides a **step-by-step** guide (template) to ML project
- Provides **links & references**
***

## Striving to:
- Provide a small definition of each concept/bullet point/check
- Provide a link to notes where a concept is further discussed
- Provide a checklist to be followed in a chronological order
- List the available methods in the literature
- List tools/packages that have incorporated the methods above
- Provide the original paper of the cited method
***

## Project template: folder & contents

```diff
master-project-root-folder    #Project folder
├── conf.cfg        # Pipeline configuration files/master file
├── data            # Data + data splitting
├──── original copy
├──── train
├──── test
├──── valid
├── docs            # Documentation
├── logs            # Logs of pipeline runs
├──── run_1_RNN_ID_11.log
├──── run_2_LTSM_ID_12.log
├──── .....
├── notebooks       # Exploratory Jupyter notebooks 
├──── Notebook_RNN_V1.ipynb
├──── Notebook_LSTM_V2.ipynb
├──── .....
├── README.md       # README.md explaining your project, similar to what you'll find on GitHub
├── src             # Source code for pipelines: python + testing
├──── Preproces.py
├──── Train.py
├──── Postproces.py
├──── Unittesting
```
***

## Table of contents
1. [Scoping (Project Management)](#scoping-project-managment)
2. [Data](#data)
3. [Programming](#programming-focused-on-python)
4. [Modelling](#modelling)
5. [Deployment](#deployment)
6. [Responsible AI](#responsabile-ai)
7. [Continuous (MLOps)](#continuous-mlops)
***

## Scoping (Project Managment)
<details>
<summary>Expand ⬇️</summary>
<br>

- Frame the problem and look at the big picture    
- 🏦 BI (Business Intelligence) Tools involves the functions, strategies, and tools companies use to collect, process, and analyze data [Ref](https://www.coursera.org/articles/bi-tools). These tools can help framing the problem:
    - [Microsoft Power BI](https://powerbi.microsoft.com/en-us/what-is-power-bi/)
    - [Tableau](https://www.tableau.com/products/desktop)
    - [QlikSense](https://www.qlik.com/us/products/qlik-sense)
    - [Dundas BI](https://insightsoftware.com/dundas/)
    - [Sisense](https://www.sisense.com/)
- ❓ What is the project main objective(s)?
- ❓ Which part of the main objective the ML model(s) is addressing?
- 📈📉 Establish a [baseline](https://blog.ml.cmu.edu/2020/08/31/3-baselines/) against which your ML will be considered successful (an improvement against the baseline)
- ❓ Are there any solutions not based on a ML model? You are likely to be asked to compared your method against some no-ML model!
- ❓ Can How would you solve the problem manually?
    - ✅ Yes, then how would you do it?
    - ❌ No, then something more complex is needed
- Define the objectives in business terms. This involvs choosing the business KPIs (key performance indicators). These are what businesses use to measure the uplift brought in by the ML-based solution.
- 🚔 Now put yourself in the **user seat** and make sure there is an alignment btw business KPIs and those stricly related to the users.
- Think about how the ML soluion will be used
- 📈 Monitor your project's objective(s) over time. Yes, you heard it right; do not monitor only the results. Requirements and project's goal do tend to change over time unfortunately.
- 🗣️ Talk to the domain experts, they are those with the domain knowledge 
- ⚠️ Keep track of what did not work as you develop your ML solution. Knowledge is not only about what worked, but largely what didn't.
- 🔄 Keep in mind that ML solutions are not one-shot solutions. They need to be 1) followed and 2) developed over time
- Tool to manage/projects/people:    
    - [Jira](https://www.atlassian.com/software/jira)
    - [Confluence](https://www.atlassian.com/software/confluence) | [Jira vs. Confluence](https://elements-apps.com/blog/jira-and-confluence/)
    - [Trello](https://trello.com/home)
- Choose btw these different 3 scenarios (do not underestimate the importance of this, and this is the reason why it is under scoping and not under data or modelling section):
    - **Data driven**: means the creation of technologies, skills, and an environment by ingesting a large amount of data. This does not mean data centric.
    - **Data centric**: involves systematically altering/improving datasets in order to increase the accuracy of your ML applications.
    - **Model centric**: keep the data the same, and you only improve the code/model architecture. What happens when new data is added or changed? The risk of having a bias-to-that-batch-of-data model is very high. 
    - [Model centric vs. data centric](https://neptune.ai/blog/data-centric-vs-model-centric-machine-learning)
</details>

## Data
<details>
<summary>Expand ⬇️</summary>
<br>

- How much data do I need?
    - Rule of thumb #1: roughly 10 times as many examples (rows) as there are degrees of freedom (features) | [Ref](https://www.kdnuggets.com/2019/06/5-ways-lack-data-machine-learning.html)
    - If you are bound to a small dataset, this may be good for PoC (Proof of Concept), but for a production-ready model, you'd need many more | [Ref](https://www.kdnuggets.com/2019/06/5-ways-lack-data-machine-learning.html)
- Data sourcing/collection/ingestion:
    - Check legal obligations, and get the authorization if necessary
    - 🌐 Collect your data from the web via scraping | [Notes](https://github.com/kyaiooiayk/Website-Scrapers)
    - Collect data via third party API 
    - Build your own dataset
    - Create/augment your data with some synthetic data generation techniques
    - Dowload some open source. Best resource is probably [Kaggle](https://www.kaggle.com/)
    - Ensure sensitive information is deleted or protected (e.g., anonymised)
- Is the data enough? How do you deal with the lack of data?
    - Try to establish a real data culture within your organization. From now on, you start tracking users.
    - Build a free application and give it away while tacking how others use it. Facebook and Google are not far from this modus operandi.
    - Naive Bayes algorithm is among the simplest classifiers and learns remarkably well from relatively small data sets.
    - Consider using less complex algorithm; for instance limiting the depth of your decision tree.
    - Consider using ensemble method.
    - Consider using linear models such as liner/logistic regression where only linear interaction are modelled.
    - Use transfer learning and this is the de-facto standard for LLM.
    - Consider data augmentation. So for vision taks, you could rotate, scale etc ..
    - ⚠️ Keep in mind that using synthetic data could potentially introduce bias on a real world phenomenon.
- Is data labelling necessary?:
    - ✅ Yes, then is human expertise available? Labelling is expensive as it involves many man hours. Consider automating it as much as you can.
    - ❌ No, then unsupervised learning must be used
- Data versioning. Available tools:
    - [Hydra](https://hydra.cc/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/VCS/Hydra)
    - [DVC](https://dvc.org/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/VCS/DVC)
- ❓ Is there a data bias?
    - ✅ Yes, take action
    - ❌ No, proceed
- Keep a copy of the original unclean data where possible.
- Data ingestion/wrangling:
    - 🐼 [Pandas](https://pandas.pydata.org/) for dataset < 32Gb. For dataset that do not fit in memory you can load different chucks at the time | [Notes](https://github.com/kyaiooiayk/Pandas-Notes)
    - 🐻‍❄️ [Polars](https://github.com/pola-rs/polars) an optimised version of Pandas.
    - [Dask](https://www.dask.org/) for dataset 1Gb-100Gb | [Notes](https://github.com/kyaiooiayk/Dask) 
    - ✨[PySpark](https://spark.apache.org/docs/latest/api/python/) for dataset >100 Gb | [Notes](https://github.com/kyaiooiayk/pySpark-Notes)
    - 🧱 [Databricks](https://www.databricks.com/) develops a web-based platform for working with Spark, that provides automated cluster management and IPython-style notebooks. | [Databricks vs. Azure databricks](https://www.websitebuilderinsider.com/is-azure-databricks-same-as-databricks/)
- Data cleaning
- Data labeling
- Data validation. Establish a data schema which helps validate the data. Especially for [concept drift](https://machinelearningmastery.com/gentle-introduction-concept-drift-machine-learning/). Some commercial tools are:
    - [Pandera](https://pandera.readthedocs.io/en/stable/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Data_validation/Pandera)
    - [Great Expectations](https://greatexpectations.io/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Data_validation/Great_expectations)
- 💽 Data storage:
    - Structured data: SQL. RDB (relational database) is a database that stores data into tables with rows and columns. To be able to process SQL queries on huge volumes of data that is stored in Hadoop cluster, specialised tools are needed. Here are some options:
        - 🐝 [Hive](https://hive.apache.org/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/blob/master/tutorials/Hive.md) (twice as popular as Pig and developed by Facebook). Hive provides SQL type querying language for the ETL purpose on top of Hadoop file system. 
        - 🐷 [Pig](https://pig.apache.org/) (less popular than Hive)
        -  🦌 [Impala](https://impala.apache.org/docs/build/html/topics/impala_langref.html) | [Hive vs. Impala](https://www.tutorialspoint.com/impala/impala_overview.htm)
    - Unstructured data: NoSQL
- Data transformation
- EDA (Exploratory Design Analysis): explore the data to gain insights:
    - Type (categorical, int/float, bounded/unbounded, text, structured, etc.)
    - % of missing values
    - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)
    - Type of distribution (Gaussian, uniform, logarithmic, etc.)
    - Study the correlations between features and targets
        - If no method shows some sort of correlation btw features and targets, then you may want to study the problem harder!
    - Document in a report what you have learnt
- Data cleaning:
    - Are the any outliers? If yes, ask yourself why.
    - Fill in missing values via some imputation strategies. Treat your data transformation choices as hyperparameters, especially when you are not sure about them (e.g., replace with zero, mean, meadina or just drop the rows?):
        - Zero, mean or median
        - Drop row values or the entire columns if too many row values are missing
- Features scaline:
    - If a deep learning application this almost certaintly done
    - If not a DL application it depends. 
- Feature engineering:
    - Discretize continuous features
    - Add transformations like: log(x), sqrt(x), x^2, etc...
    - Aggregate features into common bin
- Dashboard:
    - Bokeh
    - Plotly
- Data splitting:
    - Large dataset:
        - Train
        - Test: (no data snooping!)
    - Small dataset:
        - Train
        - Test: (no data snooping!)
        - Validation
- Build an ETL (Extra, Transform & Load) pipeline
    
</details>

***

## Programming (focused on python)
<details>
<summary>Expand ⬇️</summary>
<br>

- Code release:
    - Major
    - Minor
    - Patch
- Code versionning:
    - [GitHub](https://github.com/) | [Notes](https://github.com/kyaiooiayk/Git-Cheatsheet)
    - [GitLab](https://about.gitlab.com/)
- Linters | [Notes](https://github.com/kyaiooiayk/Python-Programming/blob/main/tutorials/Code_style.md):
    - [Black](https://black.readthedocs.io/en/stable/) is essentially an autoformatter.
    - [pycodestyle](https://pypi.org/project/pycodestyle/) is similar to black but the big difference between black and pycodestyle is that black does reformat your code, whereas pycodestyle just complains.
    - [Flake8](https://flake8.pycqa.org/en/latest/) does much more than what black does. Flake8 is very close to be perfectly compatible with black.
- Code optimisation
    - Refactoring
    - Profilers
    - Caching
- Code testing
    - [Unittesting](https://docs.python.org/3/library/unittest.html) | [Notes](https://github.com/kyaiooiayk/Python-Programming/tree/main/tutorials/Unittesting)
- Code obfuscation | [Notes](https://github.com/kyaiooiayk/Python-Source-Code-Obfuscation/edit/main/README.md):
    - [pyarmor](https://pypi.org/project/pyarmor/) - It provides full obfuscation with hex-encoding; apparently doesn’t allow partial obfuscation of variable/function names only.
    - [python-minifier](https://pypi.org/project/python-minifier/) — It minifies the code and obfuscates function/variable names. 
    - [pyminifier](https://pypi.org/project/pyminifier/) - It does a good job in obfuscating names of functions, variables, literals; can also perform hex-encoding (compression) similar as pyarmor. Problem: after obfuscation the code may contain syntax errors and not run.
    - [cython](https://cython.org/) - Cython is an optimising static compiler that takes your .py modules and translates them to high-performant C files. Resulting C files can be compiled into native binary libraries with no effort. When the compilation is done there’s no way to reverse compiled libraries back to readable Python source code. What distinguishes this option from the other is that this can be used is to make your code run faster rather than obfuscating it.
- Code shipping:
    - Maven : it is used to create deployment package.
    - Containersition with [Docker](https://www.docker.com/) | [Notes](https://github.com/kyaiooiayk/Docker-Notes) is the golden and widespread standard
- Code packaging is the action of creating a package out of your python project wiht the intent to distribute it. This consists in adding the necessary files, structure and how to build the package. Further one can also upload it to the Python Package Index (PyPI). [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Packaging)
    
</details>

***

## Modelling
<details>
<summary>Expand ⬇️</summary>
<br>

- 📖 Read about the topic/field you are building a ML solution for
- ❓ How should you frame this problem supervised/unsupervised)?
- ❓ How is the data coming in: online/offline?
- Get a feeling of what the SOTA (State Of The Art)
- List the assumptions you or others have made so far.
- Keep track of your model versions
- Select what the feature(s) vs. target(s) are
-  🐣 Create a baseline model
- Keep track of your model dependencies
- Feature selection:
    - ❓ Can a domain expert help me determine which features are relevant?
    - Let the model decide which feature is important; after you can remove it to make the model more efficient
- Feature engineering
- How should performance be measured? This means chooseing the model metrics (Not model objective function and not necessarily KPIs!)
- Is the performance measure aligned with the business objective?
    - ✅ Yes, non techical people / higher level managment will be able to follow the development
    - ❌ No, then ask why? It is fine, but it necessay to find a proxy to link technical and business metrics
- Choose a model(s)-
    - First scenario: there are plenty of SOTA options and these are cheap to run. One option would be to explore many different models and short-list the best ones.
    - Second scenario: there are much less SOTA options and these are expesnive to run. This is especially true for DL model. One option would be to concentrate on one of them.
- Choose a framework:
    - Non Deep Learning:
        - Scikit-Learn
        - XGBoost
        - LightGBM
    - Deep Learning:
        - Tensor flow or KERAS
        - PyTorch
        - JAX
- Model training:
    - On premesis
    - On the cloud which means using cluster machines on the cloud. **Bare-metal** cloud is a public cloud service where the customer rents dedicated hardware resources from a remote service provider, without (hence bare) any installed operating systems or virtualization infrastructure. You have three options:
        - [AWS (Amazon Web Services)](https://aws.amazon.com/?nc2=h_lg) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/AWS)
        - [Microsoft Azure](https://azure.microsoft.com/en-gb/)
        - [GCP (Google Cloud Platform)](https://cloud.google.com/)
- Model CV (Cross Valisation)
- Model hyperparameters:
    - Methods:
        - Grid search: doable when the parameters are small 
        - Random search: preferred over random search over grid search
        - Successive halving
        - BOHB
        - Bayesian optimisation: preferred if training is very long | [Ref](https://goo.gl/PEFfGr)
    - Tools:
        - [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) is a Python library for fast hyperparameter tuning at scale. | [Paper](https://arxiv.org/abs/1807.05118)
        - [Optuna](https://optuna.org/) is an open source hyperparameter optimization framework to automate hyperparameter search. It is framework agnostic you can use it with any machine learning or deep learning framework. | [Paper](https://dl.acm.org/doi/10.1145/3292500.3330701)
    - Don'ts:
        - Once you are confident about your final model, measure its performance on the test set to estimate the generalization error. Don't tweak your model after measuring the generalization error: you would just start overfitting the test set. This is very hard in practice to enforce. Resist the temptation!
- Model inference:
    - on CPUs
    - on GPUs
    - on TPUs
- Latency vs. throughput:
    - If our application requires **low latency**, then we should deploy the model as a real-time API to provide super-fast predictions on single prediction requests over HTTPS.
     - For **less-latency-sensitive** applications that require high throughput, we should deploy our model as a batch job to perform batch predictions on large amounts of data.
- Model serialisation (aka model persistence) / deserialisation. Serialisation is the process of translating a data structure or object state into a format that can be stored or transmitted and reconstructed later. [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Model_Serialisation) Some of the formats used are :
    - hdf5
    - json
    - dill
    - joblib
    - pickle
    - skops
    - ONNX
- Model optimisation:
    - Quantisation
    - Pruning
    - Teacher-student models
- Reporting results:
    - Tell a story with data | [Ref](https://pbs.twimg.com/media/E-C33uFWUAA2UiD?format=jpg&name=large)
    - List your assumptions and your system's limitations.
    - Explain why your solution achieves the business objective.
    - Describe lessons learnt: what did not work is as much valuable as what did not.
- Keep in mind that your production model will likely be changed in the future!

</details>

***

## Deployment
<details>
<summary>Expand ⬇️</summary>
<br>

- RESTful API:
    - Django
    - [[Flask](https://flask.palletsprojects.com/en/2.1.x/) | [Notes](https://github.com/kyaiooiayk/Flask-Notes)]
    - [[Node.js]() | Notes]
    - [[Express.js]() | Notes]
    - [[React](https://reactjs.org/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/React)]
    - Redis
    - [[FastAPI](https://fastapi.tiangolo.com/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/FastAPI)]
    - [[Streamlit](https://streamlit.io/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Streamlit)]
    - [[Electron](https://www.electronjs.org/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Electron.md)]
    - [[Dash](https://plotly.com/building-machine-learning-web-apps-in-python/)]
    - [[Gradio](https://github.com/gradio-app/gradio)]
- Service end point:
    - [FastAPI](https://fastapi.tiangolo.com/): fast and a good solution for testing, has limitation when it comes to clients' request workload
    - [Flask](https://flask.palletsprojects.com/en/2.2.x/): it is less complex but not as complete as Dijango
    - [Django](https://www.djangoproject.com/): for most advanced stuff
- Public server deployment
    - [Heroku](https://www.heroku.com/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Heroku) - allows access directly to your GitHub account
    - [PythonAnywhere](https://www.pythonanywhere.com/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/pythonanywhere) - does not allow access directly to your GitHub account
    - [Netlify](https://www.netlify.com/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Netlify.md) - allows access directly to your GitHub account
- Servers:
    - [uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/) stands for Web Server Gateway Interface and is an application server that aims to provide a full stack for developing and deploying web applications and services. It is named after the Web Server Gateway Interface, which was the first plugin supported by the project.
    - [Nginx](https://www.nginx.com/) is a web server that can also be used as a reverse proxy (which provides a more robust connection handling), load balancer, mail proxy and HTTP cache.
- Deployment patters:
    - Canary
    - Green/blue
- Monitoring:
    - Latency
    - IO
    - Memory
    - Uptime: system reliability
    - Load testing: Apache Jmeter
- [Kubernets](https://kubernetes.io/) | [Notes](https://github.com/kyaiooiayk/Kubernetes-Notes) cluster:
    - Cloud vendors:
        - EKS by Amazon
        - AKS by Microsoft
        - GKS by Google
    - Local machine:
        - minikube
        - kind
        - k3s
    - Orchestration
        - [Argo Workflows](https://github.com/argoproj/argo-workflows) an open-source container-native workflow engine for orchestrating parallel jobs on Kubernetes.

</details>

***

## Responsabile AI
<details>
<summary>Expand ⬇️</summary>
<br>

- 👩 Explainability | [Tutorials](https://github.com/kyaiooiayk/Explainable-AI-xAI-Notes) | [Notes](https://drive.google.com/drive/u/1/folders/1YTvctHR28vG2zBrSPpq5I1JcbV--FS6v)
    - SHAP
- 🔐 Security
- ⚖️ Fairness
- 👮‍♀️ Auditability
- What-if-tool
- 🔐 Ensure sensitive information is deleted or protected (e.g., anonymised)

</details>

***


## Continuous (MLOps)
<details>
<summary>Expand ⬇️</summary>
<br>

- Testing
- Continuous integration is about how the project should be built and tested in various runtimes, automatically and continuously.
- Continuous deployment is needed so that every new bit of code that passes automated testing can be released into production with no extra effort. 
- Training
- Delivery
- Monitoring: see concept drift for instance
- Tools:
    - [GitHub Actions](https://github.com/features/actions) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/blob/master/tutorials/GitHub_Actions.md)
    - [Jenkins](https://www.jenkins.io/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Jenkins)
- 🤹‍♂ Orchestration tools:
    - [Kredo](https://kedro.readthedocs.io/en/stable/introduction/introduction.html) is an open-source Python framework for creating reproducible, maintainable and modular data science code. It borrows concepts from software engineering and applies them to machine-learning code; applied concepts include modularity, separation of concerns and versioning. Kedro is hosted by the LF AI & Data Foundation.
    - [ZenML](https://docs.zenml.io/getting-started/introduction) is an extensible, open-source MLOps framework to create production-ready machine learning pipelines. Built for data scientists, it has a simple, flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions that are catered towards ML workflows.
    - [Metaflow](https://docs.metaflow.org/) is a human-friendly Python library that helps scientists and engineers build and manage real-life data science projects. Metaflow was originally developed at Netflix to boost the productivity of data scientists who work on a wide variety of projects from classical statistics to state-of-the-art deep learning.
    - [Kredo vs. ZenML vs. Metaflow](https://neptune.ai/blog/kedro-vs-zenml-vs-metaflow)

</details>

***

## Other checklists
[Ref#1](https://github.com/ageron/handson-ml3/blob/main/ml-project-checklist.md) | [Ref#2](https://github.com/RJZauner/machine-learning-project-checklist) |
