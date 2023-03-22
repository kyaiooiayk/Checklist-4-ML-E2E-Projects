#  📋Checklist-4-ML-E2E-Projects📋
Checklist for ML (but I guess will touch topics related to SE, DE & DevOps) projects. See [here](https://github.com/kyaiooiayk/The-Data-Scientist-Mind-Map) to see the same but in a mind map. An almost incomplete collections of MLOps bullet points. I admit classifying all the available option is not an easy task and what you see here is not written in stone and should be considered my personal preference. This list serves the following purpouses:
- Provides a **checklist** for things that are obvious but are rarely done or effectively mostly forgotten
- Provides a **step-by-step** guide (template) to ML project
- Provides **links & references**
- Provides the complete list of **available options** along with a brief explanation
- Provides a check list to follow going from a simple PoC (Proof of Concept), to MVP (Minimum Viable Product) and finally to a fully productionaised ML-based solution
***

## Striving to:
- Provide a small definition of each concept/bullet point/check
- Provide a link to notes where a concept is further discussed
- Provide a checklist to be followed in a chronological order
- List the available methods in the literature
- List tools/packages that have incorporated the methods above
- Provide the original paper of the cited method
***
  
## Project template: folders & contents

```diff
master-project-root-folder    #Project folder
├── conf.cfg # Pipeline configuration files/master file
├── data
├──── original copy
├──── train
├──── test
├──── valid
├── docs # Documentation
├── logs # Logs of pipeline runs
├──── run_1_RNN_ID_11.log
├──── run_2_LTSM_ID_12.log
├──── .....
├── notebooks # Exploratory Jupyter notebooks 
├──── Notebook_RNN_V1.ipynb
├──── Notebook_LSTM_V2.ipynb
├──── .....
├── README.md # Project's goal and explanations, similar to what you'll find on GitHub
├── src # python + test python files
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
8. [What a Data Scientist about MLOps](#what-a-data-scientist-about-mlops)
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
- <ins>Choose btw these different 3 scenarios</ins> (do not underestimate the importance of this, and this is the reason why it is under scoping and not under data or modelling section):
    - **Data driven**: means the creation of technologies, skills, and an environment by ingesting a large amount of data. This does not mean data centric.
    - **Data centric**: involves systematically altering/improving datasets in order to increase the accuracy of your ML applications.
    - **Model centric**: keep the data the same, and you only improve the code/model architecture. What happens when new data is added or changed? The risk of having a bias-to-that-batch-of-data model is very high. 
    - [Model centric vs. data centric](https://neptune.ai/blog/data-centric-vs-model-centric-machine-learning)
- [Here is a list of lessons learnt](https://github.com/kyaiooiayk/Awesome-ML-Projects/blob/main/README.md#lessons-learnt)
- <ins>Pipeline types:</ins>
  - Data pipeline
  - Model pipeline
  - Serving pipeline
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
    - [DVC](https://dvc.org/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/VCS/DVC)
    - [DAGsHub](https://dagshub.com/)
    - [Activeloop](https://www.activeloop.ai/)
    - [Modelstore](https://modelstore.readthedocs.io/en/latest/)
    - [ModelDB](https://github.com/VertaAI/modeldb/)
- ❓ Is there a data bias?
    - ✅ Yes, take action
    - ❌ No, proceed
- Keep a copy of the original unclean data where possible.
- Data ingestion/wrangling:
    - 🐼 [Pandas](https://pandas.pydata.org/) for dataset < 32Gb. For dataset that do not fit in memory you can load different chucks at the time | [Notes](https://github.com/kyaiooiayk/Pandas-Notes)
    - 🐻‍❄️ [Polars](https://github.com/pola-rs/polars) an optimised version of Pandas.
    - [Dask](https://www.dask.org/) for dataset 1Gb-100Gb | [Notes](https://github.com/kyaiooiayk/Dask) 
    - ✨[PySpark](https://spark.apache.org/docs/latest/api/python/) for dataset >100 Gb | [Notes](https://github.com/kyaiooiayk/pySpark-Notes)
    - 🏹 [Apache PyArrow](https://arrow.apache.org/docs/python/index.html) is a cross-language development platform for in-memory data. It is a good option when data is stored in many components, for example, reading a parquet file with Python (pandas) and transforming to a Spark dataframe, Falcon Data Visualization or Cassandra without worrying about conversion. [Ref](https://towardsdatascience.com/a-gentle-introduction-to-apache-arrow-with-apache-spark-and-pandas-bb19ffe0ddae)
    - 🧱 [Databricks](https://www.databricks.com/) develops a web-based platform for working with Spark, that provides automated cluster management and IPython-style notebooks. | [Databricks vs. Azure databricks](https://www.websitebuilderinsider.com/is-azure-databricks-same-as-databricks/)
- Data cleaning
  - A comprehensive guide to bad quality data scenarios can be found [here](https://github.com/Quartz/bad-data-guide)
  - [cleanlab](https://github.com/cleanlab/cleanlab) automatically detects problems in a ML dataset. This data-centric AI package facilitates machine learning with messy, real-world data by providing clean labels for robust training and flagging errors in your data.
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
- Data file format | [Notes](https://github.com/kyaiooiayk/Data-Format-Notes)
  - CSV (Comma Separated Values) is a row-based file format storage.
  - JSON (JavaScript Object Notation) is language agnostic and supports a number of data types which includes list, dictionary, string, integer, float, boolean, Null.
  - YAML (Yet Another Markup Language) is a human-readable data-serialisation language. It is commonly used for configuration files and in applications where data is being stored or transmitted.  Both JSON and YAML are developed to provide a human-readable data interchange format
  - Parquet is a column-based file format storage and is good for storing big data of any kind (structured data tables, images, videos, documents).
  - XML (Extensible Markup Language)is exclusively designed to send and receive data back and forth between clients and servers.
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
- Features scaling:
    - If a deep learning application this is almost certaintly done. If you have two options here:
      - Min/max scaling
      - Mean/std scaling
    - If not a DL application it depends. For instance model based on decision trees are insensitive to features scaling.
- <ins>Feature engineering</ins> | [Notes](https://github.com/kyaiooiayk/Feature-Correlation-Selection-Importance-Engineering-Notes):
    - Discretize continuous features
    - Add transformations like: log(x), sqrt(x), x^2, etc...
    - Aggregate features into common bin
- <ins>Dashboard</ins>:
    - Bokeh
    - Plotly
- <ins>Data splitting</ins> | [Notes](https://drive.google.com/drive/u/1/folders/1flGUtgLDQsC3FyK9Nm-aafoSEDMNj5Ir):
    - Large dataset (CV may not be necessary):
        - Train
        - Test: (no data snooping!)
    - Small dataset (use CV while testing):
        - Train
        - Test: (no data snooping!)
        - Validation
    - No data or only a handful of examples. Enough/handful means some in order to get a sense of the problem specification but too few to train an algorithm). Consider these options:
      - A literature review
      - Analyse what others have done may give you a sense of what’s feasible.
- Build an ETL/ELT (Extra, Transform & Load) pipeline | [Notes](https://github.com/kyaiooiayk/ETL-and-ML-Pipelines-Notes/blob/main/README.md):
    - **ETL** is best used for on-premise data that needs to be structured before uploading it to a relational data warehouse. This method is typically implemented when datasets are small and there are clear metrics that a business values because large datasets will require more time to process and parameters are ideally defined before the transformation phase.
    - **ELT** is best suited for large volumes of data and implemented in cloud environments where the large storage and computing power available enables the data lake to quickly store and transform data as needed. ELT is also more flexible when it comes to the format of data but will require more time to process data for queries since that step only happens as needed versus ETL where the data is instantly queryable after loading.
</details>

***

## Programming (focused on python)
<details>
<summary>Expand ⬇️</summary>
<br>

- <ins>OOP (Object-Oriented Programming)</ins>:
  - Inheritance vs. composition: 'is-a' vs. 'has-a' relationship | [Tutorial](https://github.com/kyaiooiayk/Python-Programming/blob/main/tutorials/Composition%20vs.%20inheritance.ipynb)
- <ins>Code release</ins>:
    - Major
    - Minor
    - Patch
- <ins>Code versionning</ins>:
    - [GitHub](https://github.com/) | [Notes](https://github.com/kyaiooiayk/Git-Cheatsheet)
    - [GitLab](https://about.gitlab.com/) GitHub and GitLab are remote server repositories based on GIT. GitHub is a collaboration platform that helps review and manage codes remotely. GitLab is the same but is majorly focused on DevOps and CI/CD. 
    - [Jenkins](https://www.jenkins.io/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Jenkins)
    - [CircleCI](https://circleci.com/)
    - [Travis CI](https://www.travis-ci.com/)  
- <ins>Linters | [Notes](https://github.com/kyaiooiayk/Python-Programming/blob/main/tutorials/Code_style.md)</ins>:
    - [Black](https://black.readthedocs.io/en/stable/) is essentially an autoformatter.
    - [pycodestyle](https://pypi.org/project/pycodestyle/) is similar to black but the big difference between black and pycodestyle is that black does reformat your code, whereas pycodestyle just complains.
    - [Flake8](https://flake8.pycqa.org/en/latest/) does much more than what black does. Flake8 is very close to be perfectly compatible with black.
- <ins>Code optimisation</ins>:
    - **Refactoring** aims to revisit the source code in order to improve operation without altering functionality. | [Tutorials](https://github.com/kyaiooiayk/Python-Programming/blob/main/tutorials/Code%20refactoring.ipynb)
    - **Profilers** are tools  that aim to assess the space or time complexity of a program, the usage of particular instructions, or the frequency and duration of function calls. | [Notes on how to profile parallel jobs](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/tree/master/tutorials/Profiling_SKLearn_Parallel_Jobs) | [Notes on how to profile on jupyter notebook](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/blob/master/tutorials/Code_profiling.ipynb)
    - **Caching** consists in keeping recently (or frequently) used data in a memory location that has cheap and fast access for repeated queries. | [Notes](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/blob/master/tutorials/Caching.ipynb)
- <ins>Code testing</ins>:
    - [Unittesting](https://docs.python.org/3/library/unittest.html) | [Notes](https://github.com/kyaiooiayk/Python-Programming/tree/main/tutorials/Unittesting) Unit tests point to a specific issue that requires fixing. 
    - [Doctest](https://docs.python.org/3/library/doctest.html#module-doctest) | is a module considered easier to use than the unittest, though the latter is more suitable for more complex tests. doctest is a test framework that comes prepackaged with Python. | [Notes](https://github.com/kyaiooiayk/Python-Programming/tree/main/tutorials/Doctest)
    - [Functional testing](https://brightsec.com/blog/unit-testing-vs-functional-testing/) checks the entire application thus, it mainly indicates a general issue without pointing out a specific problem.
    - TDD (Test Driven Development) is a software development method where you define tests before you start coding the actual source code.
 | [Notes](https://github.com/kyaiooiayk/Python-Programming/blob/main/tutorials/Test-driven%20Development%20(TDD)/README.md)
- <ins>Code obfuscation | [Notes](https://github.com/kyaiooiayk/Python-Source-Code-Obfuscation/edit/main/README.md)</ins>:
    - [pyarmor](https://pypi.org/project/pyarmor/) - It provides full obfuscation with hex-encoding; apparently doesn’t allow partial obfuscation of variable/function names only.
    - [python-minifier](https://pypi.org/project/python-minifier/) — It minifies the code and obfuscates function/variable names. 
    - [pyminifier](https://pypi.org/project/pyminifier/) - It does a good job in obfuscating names of functions, variables, literals; can also perform hex-encoding (compression) similar as pyarmor. Problem: after obfuscation the code may contain syntax errors and not run.
    - [cython](https://cython.org/) - Cython is an optimising static compiler that takes your .py modules and translates them to high-performant C files. Resulting C files can be compiled into native binary libraries with no effort. When the compilation is done there’s no way to reverse compiled libraries back to readable Python source code. What distinguishes this option from the other is that this can be used is to make your code run faster rather than obfuscating it.
- <ins>Code shipping</ins>:
    - Maven : it is used to create deployment package.
    - Containersition with [Docker](https://www.docker.com/) | [Notes](https://github.com/kyaiooiayk/Docker-Notes) is the golden and widespread standard
- <ins>Code packaging</ins> is the action of creating a package out of your python project wiht the intent to distribute it. This consists in adding the necessary files, structure and how to build the package. Further one can also upload it to the Python Package Index (PyPI). [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Packaging)
    
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
-  🐣 Is a base model available (at the beginning)?
  - Yes, consider it and benchmark any future models against it
  - No, create one and benchmark any future models against it
- Keep track of your model dependencies
- <ins>Feature selection</ins>:
    - ❓ Can a domain expert help me determine which features are relevant?
    - Let the model decide which feature is important; after you can remove it to make the model more efficient
- How should performance be measured? This means choosing the model metrics (Not model objective function and not necessarily KPIs!)
  - Objective function is a function you ae trying to minimise via some optimisation algorithm
  - Model metrics can be very different from what the objective function
- Is the performance measure aligned with the business objective?
    - ✅ Yes, non techical people / higher level managment will be able to follow the development
    - ❌ No, then ask why? It is fine, but it necessay to find a proxy to link technical and business metrics
- <ins>Choose a model(s)</ins>:
    - First scenario: there are plenty of SOTA options and these are cheap to run. One option would be to explore many different models and short-list the best ones.
    - Second scenario: there are much less SOTA options and these are expesnive to run. This is especially true for DL model. One option would be to concentrate on one of them.
- <ins>Choose a framework</ins>:
    - Non Deep Learning:
        - [Scikit-Learn](https://scikit-learn.org/stable/#)
        - XGBoost
        - LightGBM
        - [CatBoost](https://catboost.ai/) is an open-source software library developed by Yandex. It provides a gradient boosting framework which among other features attempts to solve for Categorical features using a permutation driven alternative compared to the classical algorithm.
    - Deep Learning:
        - [TensorFlow](https://www.tensorflow.org/) is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks | [Tutorials&Notes](https://github.com/kyaiooiayk/TensorFlow-TF-Notes)
        - [KERAS](https://keras.io/) It is a wrapper over TF. Most of the model in TF1/2 are implemented in KERAS. Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. It also has extensive documentation and developer guides. | [Tutorials&Notes](https://github.com/kyaiooiayk/Keras-Notes)
        - [PyTorch](https://pytorch.org/)
        - [PyTorch Lightning](https://www.pytorchlightning.ai/) is built on top of ordinary (vanilla) PyTorch. The purpose of Lightning is to provide a research framework that allows for fast experimentation and scalability, which it achieves via an OOP approach that removes boilerplate and hardware-reference code.
        - [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) is a GPU/TPU-accelerated version of NumPy. It vectorises a Python function and handle all the derivative calculations on said functions. It has a JIT (Just-In-Time) component that takes your code and optimizes it for the XLA compiler, resulting in significant performance improvements over TensorFlow and PyTorch. | [Tutorials&Notes](https://github.com/kyaiooiayk/JAX-Notes)
- <ins>Model versioning</ins>. Available tools:
    - [Hydra](https://hydra.cc/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/VCS/Hydra) is a framework to configure complex applications. Effectively, it is used to read in YMAL configuration files.
- <ins>Model training</ins>:
    - On premesis
    - On the cloud which means using cluster machines on the cloud. **Bare-metal** cloud is a public cloud service where the customer rents dedicated hardware resources from a remote service provider, without (hence bare) any installed operating systems or virtualization infrastructure. You have three options:
        - [AWS (Amazon Web Services)](https://aws.amazon.com/?nc2=h_lg) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/AWS)
        - [Microsoft Azure](https://azure.microsoft.com/en-gb/)
        - [GCP (Google Cloud Platform)](https://cloud.google.com/)
- <ins>Model CV (Cross Valisation)</ins> | [Notes](https://drive.google.com/drive/u/1/folders/1flGUtgLDQsC3FyK9Nm-aafoSEDMNj5Ir) | [Paper](https://arxiv.org/pdf/2108.02497.pdf)
- <ins>Model hyperparameters</ins> | [Notes](https://drive.google.com/drive/u/1/folders/1flGUtgLDQsC3FyK9Nm-aafoSEDMNj5Ir) | [Paper](https://arxiv.org/pdf/2003.05689.pdf):
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
- <ins>Model evaluation</ins> | [Article](https://arxiv.org/pdf/2108.02497.pdf):
  - Model is not doing well on the training set:
    - Model has enough capacity: that’s a strong sign that the input features do not contain enough information to predict y. If you can’t improve the input features x, this problem will be hard to crack.
    - Model does not have enough capacity: increase the capacity, this could be adding more layers or nodes in a MLP or increasin the number of trees in a gradient-boosted model
  - Model is doing well on the training set but not the test set, there’s still hope. Plotting a learning curve (to extrapolate how performance might look with a larger dataset) and benchmarking human-level performance (HLP) can give a better sense of feasibility.
  - Model does does well on the test set, then the question still remains open whether it will generalize to real-world data. Do extra checks.
- <ins>Experiment tracking/monitoring</ins> allows us to manage all the experiments along with their components, such as parameters, metrics, and more. It makes easier to track the evolution of your model as learn more and more about the problem. Here are some available tools:
  - [MLFlow](https://mlflow.org/) is an open source project that offers experiment tracking and multiframe‐work support including Apache Spark, but limited workflow support. If you need a lightweight, simple way to track experiments and run simple workflows, this may be a good choice.
  - [Comet ML](https://www.comet.com/site/) 
  - [Neptune](https://neptune.ai/)
  - [Weights and Biases](https://wandb.ai/site) is a developer-first MLOps platform. Build better models faster with experiment tracking, dataset versioning, and model management.
  - [TensorBoard](https://www.tensorflow.org/tensorboard)
- <ins>Modell complexity</ins>: O(N^3) | O(LogN) | O(N) | [Notes](https://drive.google.com/drive/u/1/folders/1-G4Ct4iMPd7T2W-gW75eBKtuiJ37hyJj) | [Tutorials](https://github.com/kyaiooiayk/Python-Programming/tree/main/tutorials/Algorithms)
  - Space cmoplexity: storage and this generally referr to the RAM requied
  - Time complexity: this is generally related to metric such as latency
- <ins>Model selection:</ins> (essentially what if two models are indistringuishable from an accuracy PoC) | [Article](https://arxiv.org/pdf/1811.12808.pdf):
  - Check cost to train
  - Check which one is the simplest to understand
  - Check which one is the simplest to deploy
  - Check which one is the most robust
  - Give a tolerance also to metrics, essentially which are the  extrema within which two models are essentially the same from a pure metrics PoV?
- <ins>Model inference:</ins>
    - on CPUs
    - on GPUs
    - on TPUs
- <ins>Business requirements</ins>:
  - Load
  - Latency
  - Throughput
  - Storage
- <ins>Latency vs. throughput</ins>:
    - If our application requires **low latency**, then we should deploy the model as a real-time API to provide super-fast predictions on single prediction requests over HTTPS.
     - For **less-latency-sensitive** applications that require high throughput, we should deploy our model as a batch job to perform batch predictions on large amounts of data.
- <ins>Model serialisation (aka model persistence)/deserialisation</ins>. Serialisation is the process of translating a data structure or object state into a format that can be stored or transmitted and reconstructed later. | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Model_Serialisation) | Some of the formats used are: 
    - hdf5
    - dill
    - joblib
    - pickle
    - [ONNX](https://onnx.ai/) changes the paradigm in the sense that it aims to store the instructions to replicate the NN model. This allows to train your model in PT and run inference on TF. | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Model_Serialisation)
- <ins>Model optimisation</ins>:
    - Quantisation
    - Pruning
    - Teacher-student models
    - [ONNX](https://onnx.ai/) is an open file format to store (trained) machine learning models/pipelines containing sufficient detail (regarding data types etc.) to move from one platform to another. | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Model_Serialisation)
- <ins>Reporting results</ins>:
    - Tell a story with data | [Ref](https://pbs.twimg.com/media/E-C33uFWUAA2UiD?format=jpg&name=large)
    - List your assumptions and your system's limitations.
    - Explain why your solution achieves the business objective.
    - Describe lessons learnt: what did not work is as much valuable as what did.
- Keep in mind that your production model will likely be changed in the future, thus think re-trainig scheduling.

</details>

***

## Deployment
<details>
<summary>Expand ⬇️</summary>
<br>

- <ins>Container registry</ins>:  is a place to store container images. Hosting all the images in one stored location allows users to commit, identify and pull images when needed. There are many tools/services that can store the container images:
    - [Docker Hub](https://hub.docker.com/)
    - [Amazon Elastic Container Registry (ECR)](https://aws.amazon.com/ecr/)
    - [JFrog Container Registry](https://jfrog.com/container-registry/)
    - [Google Container Registry](https://cloud.google.com/container-registry)
    - [Azure container Registry](https://azure.microsoft.com/en-in/products/container-registry/#features)
- <ins>Deplyoing vs. serving [Ref](https://stackoverflow.com/questions/67018965/what-is-the-difference-between-deploying-and-serving-ml-model)</ins>
  - Deploying is the process of putting the model into the server. 
  - Serving is the process of making a model accessible from the server (for example with REST API or web sockets).
  - Both deployment and serving can have REST API (or endpoint). Deployment doesn't necessarily require a REST API (an API would be sufficient).
- <ins>Serveless deployment</ins>. Serverless” doesn’t mean there is no server, it just means that you don’t care about the underlying infrastructure for your code and you only pay for what you use. 
  - [AWS Lambda Functions](https://aws.amazon.com/lambda/) | [Notes&Tutorials](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/AWS/AWS_Lambda)
  - [Google Cloud Functions](https://cloud.google.com/functions/)
  - [Azure Functions](https://azure.microsoft.com/en-us/products/functions/)
  - [IBM Cloud Functions](https://www.ibm.com/cloud/functions)
- <ins>RESTful API</ins>:
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
- <ins>Service end point</ins>:
    - [FastAPI](https://fastapi.tiangolo.com/): fast and a good solution for testing, has limitation when it comes to clients' request workload
    - [Flask](https://flask.palletsprojects.com/en/2.2.x/): it is less complex but not as complete as Dijango
    - [Django](https://www.djangoproject.com/): for most advanced stuff
- <ins>Public server deployment</ins>:
    - [Heroku](https://www.heroku.com/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Heroku) - allows access directly to your GitHub account
    - [PythonAnywhere](https://www.pythonanywhere.com/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/pythonanywhere) - does not allow access directly to your GitHub account
    - [Netlify](https://www.netlify.com/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Netlify.md) - allows access directly to your GitHub account
- <ins>Servers</ins>:
    - [uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/) stands for Web Server Gateway Interface and is an application server that aims to provide a full stack for developing and deploying web applications and services. It is named after the Web Server Gateway Interface, which was the first plugin supported by the project.
    - [Nginx](https://www.nginx.com/) is a web server that can also be used as a reverse proxy (which provides a more robust connection handling), load balancer, mail proxy and HTTP cache.
- <ins>Serving patters</ins>:
    - Canary
    - Green/blue
- <ins>Monitoring</ins>:
    - Latency
    - Throughput
    - IO
    - Memory
    - Uptime: system reliability
    - Load testing: Apache Jmeter
- <ins>[Kubernets](https://kubernetes.io/) | [Notes](https://github.com/kyaiooiayk/Kubernetes-Notes) cluster</ins>:
    - Cloud vendors have their own application to interfeace with Kunernetes:
        - EKS by Amazon
        - AKS by Microsoft
        - GKS by Google
    - If you want to run Kubernets on your local machine (generally this is done to quickly test everythong is OK):
        - [minikube](https://minikube.sigs.k8s.io/docs/)
        - [kind](https://kind.sigs.k8s.io/)
        - [k3s](https://k3s.io/) 
    - Other rchestration tools:
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
- Encryption
- Data governace policy as a series of step-by-step procedures
- How to detect data anomalies (this is not data cleaning; It is more something done on purpouse to change the data):
  - [Benford's Law](https://en.wikipedia.org/wiki/Benford%27s_law) is a theory which states that small digits (1, 2, 3) appear at the beginning of numbers much more frequently than large digits (7, 8, 9). In theory Benford's Law can be used to detect anomalies in accounting practices or election results, though in practice it can easily be misapplied. If you suspect a dataset has been created or modified to deceive, Benford's Law is an excellent first test, but you should always verify your results with an expert before concluding your data have been manipulated.

</details>

***


## Continuous (MLOps)
This section is concerned with all those aspects that are repetitive in their nature and help deliver and maintain a ML-based solution. 
<details>
<summary>Expand ⬇️</summary>
<br>

- <ins>Continuous:</ins>
  - Testing
  - Continuous integration is about how the project should be built and tested in various runtimes, automatically and continuously.
  - Continuous deployment is needed so that every new bit of code that passes automated testing can be released into production with no extra effort. 
  - Training is about re-training the model when a trigger monitoring the model's performance is activated.
  - Delivery
- <ins>Monitoring</ins>: systems can help give us confidence that our systems are running smoothly and, in the event of a system failure, can quickly provide appropriate context when diagnosing the root cause. Here is a list of available tools:
  - [Prometheus](https://prometheus.io/)
  - [Grafana](https://grafana.com/)
  - [Fiddler](https://www.fiddler.ai/ml-model-monitoring)
  - [EvidentlyAI](https://www.evidentlyai.com/)
  - [Kibana](https://www.elastic.co/kibana/)
- <ins>Tools for CI/CD</ins> | [Tools comparison](https://neptune.ai/blog/continuous-integration-continuous-deployment-tools-for-machine-learning):
    - [GitHub Actions](https://github.com/features/actions) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/blob/master/tutorials/GitHub_Actions.md)
    - [GitLab](https://about.gitlab.com/) GitHub and GitLab are remote server repositories based on GIT. GitHub is a collaboration platform that helps review and manage codes remotely. GitLab is the same but is majorly focused on DevOps and CI/CD. 
- 🤹‍♂ <ins>Orchestration tools</ins>:
    - [Kredo](https://kedro.readthedocs.io/en/stable/introduction/introduction.html) is an open-source Python framework for creating reproducible, maintainable and modular data science code. It borrows concepts from software engineering and applies them to machine-learning code; applied concepts include modularity, separation of concerns and versioning. Kedro is hosted by the LF AI & Data Foundation.
    - [ZenML](https://docs.zenml.io/getting-started/introduction) is an extensible, open-source MLOps framework to create production-ready machine learning pipelines. Built for data scientists, it has a simple, flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions that are catered towards ML workflows.
    - [Metaflow](https://docs.metaflow.org/) is a human-friendly Python library that helps scientists and engineers build and manage real-life data science projects. Metaflow was originally developed at Netflix to boost the productivity of data scientists who work on a wide variety of projects from classical statistics to state-of-the-art deep learning.
    - [Kredo vs. ZenML vs. Metaflow](https://neptune.ai/blog/kedro-vs-zenml-vs-metaflow)

</details>

***

## What a Data Scientist about MLOps
<details>
<summary>Expand ⬇️</summary>
<br>
  
- Version Control
- CI/CD
- Dev, UAT, PROD
- Cloud Compute (AWS, GCP, Azure)
- Batch Orchestration (Airflow)
- Load Balancer
- REST API Frameworks (Flask, Django, FastAPI)
- Non-Relational & Relational Databases
- Real-Time Processing (Spark, Kafka)n
- Containerization: Kubernetes | [Notes](), Docker | [Notes]()
</details>

***

## Other checklists
[Ref#1](https://github.com/ageron/handson-ml3/blob/main/ml-project-checklist.md) | [Ref#2](https://github.com/RJZauner/machine-learning-project-checklist) | [Ref#3](https://github.com/datastacktv/data-engineer-roadmap) | [Ref#4](https://github.com/igorbarinov/awesome-data-engineering#databases) | [awesome-production-machine-learning](https://github.com/zhimin-z/awesome-production-machine-learning)
