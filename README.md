#  📋Checklist-4-ML-E2E-Projects📋
Checklist for ML projects. An almost incomplete collections of MLOps bullet points. This list serves three purpouses:
- Provides a **checklist** for things that are obvious but are rarely done/mostly forgotten
- Provides a **step-by-step** guide (template) to ML project
- Provides **links & references**
***

# Table of contents
1. [Scoping (Project Management)](#scoping-project-managment)
2. [Data](#data)
3. [Programming](#programming)
4. [Modelling](#modelling)
5. [Deployment](#deployment)
6. [Responsible AI](#responsabile-ai)
7. [Continuous (MLOps)](#continuous-mlops)
***

## Scoping (Project Managment)
<details>
<summary>Expand ⬇️</summary>
<br>

- ❓ What is the project main objective(s)?
- ❓ Which part of the main objective the ML model(s) is addressing?
- 📈📉 Establish a [baseline](https://blog.ml.cmu.edu/2020/08/31/3-baselines/) against which your ML will be considered successful (an improvement against the baseline)
- ❓ Are there any solutions not based on a ML model? You are likely to be asked to compared your method against some no-ML model!
- Choose the business KPIs (key performance indicators). These are what businesses use to measure the uplift brought in by the ML-based solution.
- 📈 Monitor your project's objective(s) over time
- 🗣️ Talk to the domain experts, they are those with the domain knowledge 
- ⚠️ Keep track of what did not work as you develop your ML solution. Knowledge is not only about what worked, but largely what didn't.
- 🔄 Keep in mind that ML solutions are not one-shot solutions. They need to be 1) followed and 2) developed over time

</details>

## Data
<details>
<summary>Expand ⬇️</summary>
<br>

- Data sourcing/collection/ingestion:
    - Collect your data from the web via scraping
    - Build your own dataset
    - Create/augment your data with some synthetic data generation techniques
    - Dowload some open source. Best resource is probably [Kaggle](https://www.kaggle.com/)
- Data versioning. Available tools:
    - [Hydra](https://hydra.cc/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/VCS/Hydra)
    - [DVC](https://dvc.org/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/VCS/DVC)
- Data ingestion/wrangling:
    - 🐼 [Pandas](https://pandas.pydata.org/) for dataset < 32Gb. For dataset that do not fit in memory you can load different chucks at the time | [Notes](https://github.com/kyaiooiayk/Pandas-Notes)
    - 🐻‍❄️ [Polars](https://github.com/pola-rs/polars) an optimised version of Pandas.
    - [Dask](https://www.dask.org/) for dataset 1Gb-100Gb | [Notes](https://github.com/kyaiooiayk/Dask) 
    - ✨[PySpark](https://spark.apache.org/docs/latest/api/python/) for dataset >100 Gb | [Notes](https://github.com/kyaiooiayk/pySpark-Notes)
- Data cleaning
- Data labeling
- Data validation. Establish a data schema which helps validate the data. Especially for [concept drift](https://machinelearningmastery.com/gentle-introduction-concept-drift-machine-learning/). Some commercial tools are:
    - [Pandera](https://pandera.readthedocs.io/en/stable/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Data_validation/Pandera)
    - [Great Expectations](https://greatexpectations.io/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Data_validation/Great_expectations)
- 💽 Data storage:
    - Structured data: SQL. RDB (relational database) is a database that stores data into tables with rows and columns. To be able to process SQL queries on huge volumes of data that is stored in Hadoop cluster, specialised tools are needed. Here are some options:
        - 🐝 [Hive](https://hive.apache.org/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/blob/master/tutorials/Hive.md) (twice as popular as Pig and developed by Facebook). Hive provides SQL type querying language for the ETL purpose on top of Hadoop file system. 
        - 🐷 [Pig](https://pig.apache.org/) (less popular than Hive)
        -  🦌 [Impala](https://impala.apache.org/docs/build/html/topics/impala_langref.html)    
    - Unstructured data: NoSQL
- Data transformation
- EDA (Exploratory Design Analysis)
- Dashboard:
    - Bokeh
    - Plotly
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
    - [GitHub](https://github.com/) | [Notes](https://github.com/kyaiooiayk/Git-Cheatsheet)
    - [GitLab](https://about.gitlab.com/)
- Code optimisation
    - Refactoring
    - Profilers
    - Caching
- Code testing
    - [Unittesting](https://docs.python.org/3/library/unittest.html) | [Notes](https://github.com/kyaiooiayk/Python-Programming/tree/main/tutorials/Unittesting)
- Code obfuscation
    - Serialisation vith Cython
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
- Get a feeling of what the SOTA (State Of The Art)
- Keep track of your model versions
- Select what the feature(s) vs. target(s) are
-  🐣 Create a baseline model
- Keep track of your model dependencies
- Feature selection
- Feature engineering
- Model metrics (Not model objective function and not necessarily KPIs!)    
- Model training:
    - On premesis
    - On the cloud which means using cluster machines on the cloud. **Bare-metal** cloud is a public cloud service where the customer rents dedicated hardware resources from a remote service provider, without (hence bare) any installed operating systems or virtualization infrastructure. You have three options:
        - [AWS (Amazon Web Services)](https://aws.amazon.com/?nc2=h_lg) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/AWS)
        - [Microsoft Azure](https://azure.microsoft.com/en-gb/)
        - [GCP (Google Cloud Platform)](https://cloud.google.com/)
- Model CV (Cross Valisation)
- Model hyperparameters:
    - Grid search
    - Successive halving
    - BOHB
- Model inference:
    - on CPUs
    - on GPUs
    - on TPUs
- Latency vs. throughput:
    - If our application requires **low latency**, then we should deploy the model as a real-time API to provide super-fast predictions on single prediction requests over HTTPS.
     - For **less-latency-sensitive** applications that require high throughput, we should deploy our model as a batch job to perform batch predictions on large amounts of data.
- Model serialisation (aka model persistence) / deserialisation. Serialisation is the process of translating a data structure or object state into a format that can be stored or transmitted and reconstructed later. [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Model_Serialisation) Some of the format used are :
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

</details>

***

## Responsabile AI
<details>
<summary>Expand ⬇️</summary>
<br>

-  👩 Explainability
-  🔐 Security
-  ⚖️ Fairness
-  👮‍♀️ Auditability

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
</details>

***
