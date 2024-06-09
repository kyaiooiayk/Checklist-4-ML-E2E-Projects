#  📋Checklist-4-ML-E2E-Projects📋
Checklist for ML (via SE, DE & DevOps) projects. See [here](https://github.com/kyaiooiayk/The-Data-Scientist-Mind-Map) to see the same but in a mind map. I admit classifying all the available options is not an easy task and what you see here is not written in stone, and should be considered my personal preference. This list serves the following purpouses:
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
- List the available methods in th∂e literature
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
- Another project organisation may look like this (8 packages - see modules vs. packages):
  - configs: in configs we define every single thing that can be configurable and can be changed in the future. Good examples are training hyperparameters, folder paths, the model architecture, metrics, flags
  - dataloader is quite self-explanatory. All the data loading and data preprocessing classes and functions live here.
  - evaluation is a collection of code that aims to evaluate the performance and accuracy of our model.
  - executor: in this folder, we usually have all the functions and scripts that train the model or use it to predict something in different environments. And by different environments I mean: executors for GPUs, executors for distributed systems. This package is our connection with the outer world and it’s what our “main.py” will use.
  - model contains the actual deep learning code (we talk about tensorflow, pytorch etc)
  - notebooks include all of our jupyter/colab notebooks in one place.
  - ops: this one is not always needed, as it includes operations not related with machine learning such as algebraic transformations, image manipulation techniques or maybe graph operations.
  - utils: utilities functions that are used in more than one places and everything that don’t fall in on the above come here.
  - `__init__.py`
  - `main.py`
***

## Infrastructure
A good example how a common interface orchastrated by terraform can be created is offerd my [mlinfra](https://mlinfra.io/latest/#how-does-it-work)
```yaml
name: aws-mlops-stack
provider:
  name: aws
  account-id: xxxxxxxxx
deployment:
  type: kubernetes
stack:
  data_versioning:
    - lakefs # can also be pachyderm or neptune and so on
  experiment_tracker:
    - mlflow # can be weights and biases or determined, or neptune or clearml and so on...
  orchestrator:
    - zenml # can also be argo, or luigi, or aws-batch or airflow, or dagster, or prefect  or kubeflow or flyte
  artifact_tracker:
    - mlflow # can also be neptune or clearml or lakefs or pachyderm or determined or wandb and so on...
  model_registry:
    - bentoml # can also be  mlflow or neptune or determined and so on...
  model_serving:
    - nvidia_triton # can also be bentoml or fastapi or cog or ray or seldoncore or tf serving
  monitoring:
    - nannyML # can be grafana or alibi or evidently or neptune or mlflow or prometheus or weaveworks and so on...
  alerting:
    - mlflow # can be mlflow or neptune or determined or weaveworks or prometheus or grafana and so on...
```
***

## Table of contents
1. [Scoping (Project Management)](#scoping-project-managment)
2. [Product](#product)
3. [Version control](#version-control)
4. [Environment, Package and Project Manager](#environment-package-and-project-manager)
5. [Data](#data)
6. [Data Engineering](#%EF%B8%8Fdata-engineering)
7. [Programming](#%EF%B8%8Fprogramming-focused-on-python)
8. [Modelling](#%EF%B8%8Fmodelling)
9. [Deployment](#deployment)
10. [Responsible AI](#responsabile-ai)
11. [Continuous (MLOps)](#continuous-mlops)
12. [What a Data Scientist should know about MLOps](#what-a-data-scientist-should-know-about-mlops)
13. [Architecture](#architecture)
***

## 🌍Scoping (Project Managment)
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
- ❓ What are the project main objectives?
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
- ❓ Batch vs. real-time inference?
- <ins>Pipeline types:</ins>
  - Data pipeline
  - Model pipeline
  - Serving pipeline
- [List of lessons learnt | Learn from others' mistakes](https://github.com/kyaiooiayk/Awesome-ML-Lessons-Learnt)
- Patterns for building ML-driven product [Ref](https://eugeneyan.com/writing/llm-patterns/?utm_source=substack&utm_medium=email)
  - *Evaluations* to measure performance. A set of measurements used to assess a model’s performance on a task. They include benchmark data and metrics. 
  - *Retrive external (to the model) data*: to add recent, external knowledge. Fetch relevant data from outside the foundation model and enhances the input with this data, providing richer context to improve output.
  - *Fine-tuning*: To get better at specific tasks rather having some average product.
  - *Caching* to reduce latency & cost
  - *Guardrails* to ensure output quality. This is not limited to LLms.
  - *Defensive UX* to anticipate & manage errors gracefully
  - *Collect user feedback* to build our data flywheel.
- <ins>Frameworks like AAARRRg (g=growth) to identify your funnel KPIs.</ins>
  - 𝗔𝘄𝗮𝗿𝗲𝗻𝗲𝘀𝘀 - How does the product get discovered?
  - 𝗔𝗰𝗾𝘂𝗶𝘀𝗶𝘁𝗶𝗼𝗻 - How does it acquire users?
  - 𝗔𝗰𝘁𝗶𝘃𝗮𝘁𝗶𝗼𝗻 - How many use it?
  - 𝗥𝗲𝘁𝗲𝗻𝘁𝗶𝗼𝗻 - How many users return?
  - 𝗥𝗲𝘃𝗲𝗻𝘂𝗲 - What's the revenue?
  - 𝗥𝗲𝗳𝗲𝗿𝗿𝗮𝗹 - How many refer to the product?

</details>

## 📦Product
<details>
<summary>Expand ⬇️</summary>
<br>

- **MVP**: Focuses on core functionality for early validation.
- **MLP**: Enhances the MVP to create a delightful user experience.
- **MMP**: Prepares the product for market entry with a balance of essential features and market readiness.

</details>
  
## 💾Version control
<details>
<summary>Expand ⬇️</summary>
<br>

- Decide between GitHub and GitLab
- Create .gitignore file | [Example #1](https://github.com/kyaiooiayk/Git-Cheatsheet/blob/main/.gitignore)
- Configure git Hooks | [Notes](https://github.com/kyaiooiayk/Git-Cheatsheet/tree/main#git-hooks)

</details>
  
## Environment, Package and Project Manager
<details>
<summary>Expand ⬇️</summary>
<br>
  
- Conda | [Notes](https://github.com/kyaiooiayk/Environment-Package-and-Project-Manager)
- pip | [Notes](https://github.com/kyaiooiayk/Environment-Package-and-Project-Manager)
- Poetry | [Notes](https://github.com/kyaiooiayk/Environment-Package-and-Project-Manager)

</details>

## 💽Data
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
    - [lakeFS](https://lakefs.io/)
    - [Pachyderm](https://www.pachyderm.com/)
    - [Neptune](https://docs.neptune.ai/)
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
- What is EDA?
  - Explore the data to gain insights: Do I have the right signals for the model?
  - Identify the feasibility of the project: Is it possible to deliver a solution using the data I have?
  - Craft a story: Can I reveal useful patterns in the data to the stakeholder?
- What to do in a EDA (Exploratory Design Analysis)?
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
- <ins>Feature engineering</ins> | [Notes](https://drive.google.com/drive/u/2/folders/1ABSeXMUvG-AbFcxvFxJ0J0xpFDYUuA21) | [Tutorials](https://github.com/kyaiooiayk/Feature-Correlation-Selection-Importance-Engineering-Notes):
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

</details>

***

## ⛓️Data Engineering
<details>
<summary>Expand ⬇️</summary>
<br>

- <ins>ETL: Extract, Transform, Load</ins>:
  - **Extraction** involves the process of extracting the data from multiple homogeneous or heterogeneous sources.
  - **Transformation** refers to data cleansing and manipulation in order to convert them into a proper format.
  - **Loading** is the injection of the transformed data into the memory of the processing units that will handle the training (whether this is CPUs, GPUs or even TPUs)
    
- <ins>ETL vs. ELT pipeline</ins> | [Notes](https://github.com/kyaiooiayk/ETL-and-ML-Pipelines-Notes/blob/main/README.md):
    - **ETL** is best used for on-premise data that needs to be structured before uploading it to a relational data warehouse. This method is typically implemented when datasets are small and there are clear metrics that a business values because large datasets will require more time to process and parameters are ideally defined before the transformation phase.
    - **ELT** is best suited for large volumes of data and implemented in cloud environments where the large storage and computing power available enables the data lake to quickly store and transform data as needed. ELT is also more flexible when it comes to the format of data but will require more time to process data for queries since that step only happens as needed versus ETL where the data is instantly queryable after loading.
- <ins>How to load your data (and how to optimise it)?</ins>
  - **All at once vs. lazy loading with iterator**. Running a for-loop in a dataset requires you to load the entire dataset into memory. What if you do not have that much memory to spare? An iterator is nothing more than an object that enables us to traverse throughout our collection with the advantage of that it loads each data point only when it's needed.
  - **Bathching**: Batch processing has a slightly different meaning for a SE and a MLE. For a SE batching is a method to run high volume, repetitive jobs into groups with no human interaction while the latter thinks of it as the partitioning of data into chunks as this makes the training much more efficient because of the way the stochastic gradient descent algorithm works.
  - **Prefetching** allows you to run preprocessing and model execution at the same time. While the model is executing training step m, the input pipeline is reading the data for next m+1 step. Prefetching is alike to a decoupled producer-consumer system coordinated by a buffer. The producer = data processing and the consumer = model, while the buffer is handling the transportation of the data from one to the other.
  - **Caching** is a way to temporarily store data in memory to avoid repeating some operations. The caveat here is that we have to be very careful on the limitations of our resources, to avoid overloading the cache with too much data.
  - **Streaming** allows to transmit or receiving data as a steady, continuous flow, allowing playback to start while the rest of the data is still being received. We can open a connection with an external data source and keep processing the data and training a ML model as long as they come. [Kafka](https://kafka.apache.org/) is a high performant, distributed messaging system that takes care of this streaming idea.
- <ins>Data vs. model parallelism</ins>
  - **Data parallelism [easy]**: concerns how how to distribute our data and train the model in multiple devices (CPUs, GPUs, TPUs) with different chunks.
  - **Model parallelism [harder]**: When a model is so big that it doesn't fit in the memory of a single device, we can divide it into different parts, distribute them across multiple machines and train each one of them independently using the same data. In an encoder-decoder architecture to train the decoder and the encoder into different machines. This can be combined with data parallelism: feeding the exact same (n-th) data batch into both machines.
- <ins>How to scale your DB</ins>:
  - For **SQL-based** solution: scalability techniques such as master-slave replication, sharding, denormalization, federation
  - For **for NoSQL** solution: scalability techniques are key-value, document, column-based, graph

</details>
  
***

## 🧑‍💻️Programming (focused on python)
<details>
<summary>Expand ⬇️</summary>
<br>

- [DRY](https://www.earthdatascience.org/courses/earth-analytics/automate-science-workflows/write-efficient-code-for-science-r/) - Don't repeat yourself. If you find yourself writing the same code more than twice. Modularise it and save it in a repo, for yourself in the next project and colleagues.
- [SOLID](https://en.wikipedia.org/wiki/SOLID): is a mnemonic acronym for five design principles intended to make object-oriented designs more understandable, flexible, and maintainable.
  - The Single-responsibility principle: "There should never be more than one reason for a class to change."[5] In other words, every class should have only one responsibility.
  - The Open–closed principle: "Software entities ... should be open for extension, but closed for modification."[7]
  - The Liskov substitution principle: "Functions that use pointers or references to base classes must be able to use objects of derived classes without knowing it."[8] See also design by contract.[8]
  - The Interface segregation principle: "Clients should not be forced to depend upon interfaces that they do not use."[9][4]
  - The Dependency inversion principle: "Depend upon abstractions, [not] concretions."[10][4]
- <ins>OOP (Object-Oriented Programming)</ins>:
  - Inheritance vs. composition: 'is-a' vs. 'has-a' relationship | [Tutorial](https://github.com/kyaiooiayk/Python-Programming/blob/main/tutorials/Composition%20vs.%20inheritance.ipynb)
- <ins>Docs | [Tutorials](https://github.com/kyaiooiayk/Awesome-Python-Programming-Notes/blob/main/tutorials/GitHub_MD_rendering/Docstrings.ipynb)</ins>:
  - [NumPy/SciPy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) Combination of reStructured and GoogleDocstrings and supported by Sphinx    
  - [EpyDoc](http://epydoc.sourceforge.net/) Render Epytext as series of HTML documents and a tool for generating API documentation for Python modules based on their Docstrings
  - [Google Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) Google's Style
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
- <ins>Production-grade code</ins> | [Notes](https://github.com/kyaiooiayk/Awesome-Python-Programming-Notes/tree/main/tutorials/Production-grade%20code):
    - **Factory Pattern** is used to decouple data IO, or in other words the data sources (SQL, pandas etc ..)
    - **Strategy Pattern** is used to decouple algorithms.
    - **Adapter Pattern** is used to decouple external services.
- <ins>Python style guide</ins>
  - [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#Threading)
  - [Python code style guidelines](https://github.com/kyaiooiayk/Awesome-Python-Programming-Notes/blob/main/tutorials/Code_style.md)
- <ins>Linters & Formatter</ins> Linter is there to catch potential bugs and issues, whereas a formatter is there to enforce a consistent code style and format. | [Notes #1](https://github.com/kyaiooiayk/Python-Programming/blob/main/tutorials/Code_style.md) | [Notes #2](https://inventwithpython.com/blog/2022/11/19/python-linter-comparison-2022-pylint-vs-pyflakes-vs-flake8-vs-autopep8-vs-bandit-vs-prospector-vs-pylama-vs-pyroma-vs-black-vs-mypy-vs-radon-vs-mccabe/):
    - [Pylint](https://pypi.org/project/pylint/) is a static code analyser for Python 2 or 3. it analyses your code without actually running it. It checks for errors, enforces a coding standard, looks for code smells, and can make suggestions about how the code could be refactored. | [Why no one uses Pylint](https://pythonspeed.com/articles/pylint/). Install it with `pip install pylint`. Usage: `pylint file.py --errors-only --disable=C,R` or `pylint file.py --errors-only --disable=C,R`
    - Pyflakes
    - autopep8
    - Bandit
    - Prospector
    - Pylama
    - Pyroma
    - [isort](https://pycqa.github.io/isort/) is a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type. 
    - [Mypy](https://mypy.readthedocs.io/en/stable/) is an optional static type checker for Python that aims to combine the benefits of dynamic (or "duck") typing and static typing. Mypy combines the expressive power and convenience of Python with a powerful type system and compile-time type checking. 
    - Radon
    - mccabe 
    - [Black](https://black.readthedocs.io/en/stable/) is essentially an autoformatter.
    - [pycodestyle](https://pypi.org/project/pycodestyle/) is similar to black but the big difference between black and pycodestyle is that black does reformat your code, whereas pycodestyle just complains.
    - [Flake8](https://flake8.pycqa.org/en/latest/) does much more than what black does. Flake8 is very close to be perfectly compatible with black.
    - ⭐️[Ruff](https://beta.ruff.rs/docs/) is an extremely fast Python linter, written in Rust. Ruff can be used to replace Flake8 (plus dozens of plugins), isort, pydocstyle, yesqa, eradicate, pyupgrade, and autoflake, all while executing tens or hundreds of times faster than any individual tool.
- <ins>Static type checker</ins> is the process of verifying and enforcing the constraints of types.:
  - [Pytype](https://github.com/google/pytype)
  - IDE such pycharm will do it automatically.
- <ins>Production Code (How maintainable is it?)</ins>:
    - **Refactoring** aims to revisit the source code in order to improve operation without altering functionality. | [Notes](https://github.com/kyaiooiayk/Awesome-Python-Programming-Notes/tree/main/tutorials/Production-grade%20code)     
- <ins>Code optimisation (How fast is it?)</ins>:
    - **Profilers** are tools  that aim to assess the space or time complexity of a program, the usage of particular instructions, or the frequency and duration of function calls. | [Notes on how to profile parallel jobs](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/tree/master/tutorials/Profiling_SKLearn_Parallel_Jobs) | [Notes on how to profile on jupyter notebook](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/blob/master/tutorials/Code_profiling.ipynb)
    - **Caching** consists in keeping recently (or frequently) used data in a memory location that has cheap and fast access for repeated queries. | [Notes](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/blob/master/tutorials/Caching.ipynb)
    - **Multi-threading** [Tutorials](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/tree/master/tutorials) | [Notes](https://drive.google.com/drive/u/1/folders/13mzxrofldkbdgF_eT5EPZ1cEiCgOT78d)
    - **Multi-processing** [Tutorials](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/tree/master/tutorials) | [Notes](https://drive.google.com/drive/u/1/folders/13mzxrofldkbdgF_eT5EPZ1cEiCgOT78d)
    - **Cython** [Note](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/tree/master/tutorials)
    - **Numba** [Note](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/tree/master/tutorials)
    - **Scoop** [Note](https://github.com/kyaiooiayk/High-Performance-Computing-in-Python/tree/master/tutorials)
- <ins>Code testing</ins> [Tutorials](https://github.com/kyaiooiayk/Awesome-Python-Programming-Notes/tree/main/tutorials/Testing):
    - [Unittesting](https://docs.python.org/3/library/unittest.html) | [Notes](https://github.com/kyaiooiayk/Python-Programming/tree/main/tutorials/Unittesting) Unit tests point to a specific issue that requires fixing. | [How to unittest DL model](https://theaisummer.com/unit-test-deep-learning/)
    - [Doctest](https://docs.python.org/3/library/doctest.html#module-doctest) | is a module considered easier to use than the unittest, though the latter is more suitable for more complex tests. doctest is a test framework that comes prepackaged with Python. | [Notes](https://github.com/kyaiooiayk/Python-Programming/tree/main/tutorials/Doctest)
    - [Functional testing](https://brightsec.com/blog/unit-testing-vs-functional-testing/) checks the entire application thus, it mainly indicates a general issue without pointing out a specific problem.
    - TDD (Test Driven Development) is a software development method where you define tests before you start coding the actual source code.
 | [Notes](https://github.com/kyaiooiayk/Python-Programming/blob/main/tutorials/Test-driven%20Development%20(TDD)/README.md)
- <ins>Code obfuscation | [Notes](https://github.com/kyaiooiayk/Python-Source-Code-Obfuscation/edit/main/README.md)</ins>:
    - [pyarmor](https://pypi.org/project/pyarmor/) - It provides full obfuscation with hex-encoding; apparently doesn’t allow partial obfuscation of variable/function names only.
    - [python-minifier](https://pypi.org/project/python-minifier/) — It minifies the code and obfuscates function/variable names. 
    - [pyminifier](https://pypi.org/project/pyminifier/) - It does a good job in obfuscating names of functions, variables, literals; can also perform hex-encoding (compression) similar as pyarmor. Problem: after obfuscation the code may contain syntax errors and not run.
    - [cython](https://cython.org/) - Cython is an optimising static compiler that takes your .py modules and translates them to high-performant C files. Resulting C files can be compiled into native binary libraries with no effort. When the compilation is done there’s no way to reverse compiled libraries back to readable Python source code. What distinguishes this option from the other is that this can be used is to make your code run faster rather than obfuscating it.
- <ins>Logging</ins>:
    - [Python logging package](https://docs.python.org/3/library/logging.html) | [Tutorial](https://github.com/kyaiooiayk/Awesome-Python-Programming-Notes/blob/main/tutorials/Logging%20module.ipynb) | [How to log your DL model](https://theaisummer.com/logging-debugging/)
- <ins>Metrics</ins>:
  - [Dora | DevOps Research and Assessment](https://laszlo.substack.com/p/dora-metrics-simplified?utm_source=substack&utm_medium=email)   
- <ins>Code shipping</ins>:
    - Maven : it is used to create deployment package.
    - Containersition with [Docker](https://www.docker.com/) | [Notes](https://github.com/kyaiooiayk/Docker-Notes) is the golden and widespread standard
- <ins>Code packaging</ins> is the action of creating a package out of your python project wiht the intent to distribute it. This consists in adding the necessary files, structure and how to build the package. Further one can also upload it to the Python Package Index (PyPI). | [Notes](https://github.com/kyaiooiayk/Python-project-template/blob/main/README.md)
    
</details>

***

## ⚙️Modelling
<details>
<summary>Expand ⬇️</summary>
<br>

- 📖 Read about the topic/field you are building a ML solution for
- ❓ How should you frame this problem supervised/unsupervised)?
- ❓ How is the data coming in: online/offline?
- Get a feeling of what the SOTA (State Of The Art)
- List the assumptions you or others have made so far
- <ins>Although this checklist is heavily focused on ML-based model, consider the following</ins>:
  - Build a heuristic model. This can be used as a back-up solution to fall to and an easy one to explain.
  - Build a statistical model. Although, this is said not to scale well for large data, there is still room for some experimentation.
  - Buil a ML model. Yes, I am aware of the fact that some ML solution are simply best in class (see CV) and building other model is simple not worth your time!
  - Build a hybrid model if possible. Reality is never black and white, it's a mix!
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
    - [hdf5](https://www.hdfgroup.org/solutions/hdf5)
    - [dill](https://pypi.org/project/dill/) is used when pickle or joblib won’t work, or when you have custom functions that need to be serialised as part of the model. In general, dill will provide the most flexibility in terms of getting the model serialised and should be considered the path of least resistance when it comes to serialising ML models for production.
    - [joblib](https://joblib.readthedocs.io/en/latest/index.html) is used for objects which contain lots of data in numpy arrays.
    - [pickle](https://docs.python.org/3/library/pickle.html#module-pickle) is used to serialise objects with an importable hierarchy.
    - [ONNX](https://onnx.ai/) changes the paradigm in the sense that it aims to store the instructions to replicate the NN model. This allows to train your model in PT and run inference on TF. | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Model_Serialisation)
- <ins>Model optimisation</ins> | [Notes](https://github.com/kyaiooiayk/Cheap-ML-models):
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

## 🚢Deployment
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
    - [FastAPI](https://fastapi.tiangolo.com/): fast and a good solution for testing, has limitation when it comes to clients' request workload [More here](https://www.reddit.com/r/Python/comments/2jja20/is_flask_good_enough_to_develop_large_applications/?rdt=56532)
    - [Flask](https://flask.palletsprojects.com/en/2.2.x/): it is less complex but not as complete as Dijango
    - [Django](https://www.djangoproject.com/): for most advanced stuff
- <ins>Public server deployment</ins>:
    - [Heroku](https://www.heroku.com/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Heroku) - allows access directly to your GitHub account
    - [PythonAnywhere](https://www.pythonanywhere.com/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/pythonanywhere) - does not allow access directly to your GitHub account
    - [Netlify](https://www.netlify.com/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Netlify.md) - allows access directly to your GitHub account
- <ins>Servers</ins>:
    - [uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/) stands for Web Server Gateway Interface and is an application server that aims to provide a full stack for developing and deploying web applications and services. It is named after the Web Server Gateway Interface, which was the first plugin supported by the project.
    - [Nginx](https://www.nginx.com/) is a web server that can also be used as a reverse proxy (which provides a more robust connection handling), load balancer, mail proxy and HTTP cache.
- <ins>Serving patterns</ins>:
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
    - If you want to run Kubernets on your local machine (generally this is done to quickly test everything is OK):
        - [minikube](https://minikube.sigs.k8s.io/docs/)
        - [kind](https://kind.sigs.k8s.io/)
        - [k3s](https://k3s.io/) 
    - Other orchestration tools:
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
  - mlflow
  - weights and biases
  - determined
  - neptune
  - clearml
- <ins>Tools for CI/CD</ins> | [Tools comparison](https://neptune.ai/blog/continuous-integration-continuous-deployment-tools-for-machine-learning):
    - [GitHub Actions](https://github.com/features/actions) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/blob/master/tutorials/GitHub_Actions.md)
    - [GitLab](https://about.gitlab.com/) GitHub and GitLab are remote server repositories based on GIT. GitHub is a collaboration platform that helps review and manage codes remotely. GitLab is the same but is majorly focused on DevOps and CI/CD. 
- 🤹‍♂ <ins>Orchestration tools</ins>:
    - [Kredo](https://kedro.readthedocs.io/en/stable/introduction/introduction.html) is an open-source Python framework for creating reproducible, maintainable and modular data science code. It borrows concepts from software engineering and applies them to machine-learning code; applied concepts include modularity, separation of concerns and versioning. Kedro is hosted by the LF AI & Data Foundation.
    - [ZenML](https://docs.zenml.io/getting-started/introduction) is an extensible, open-source MLOps framework to create production-ready machine learning pipelines. Built for data scientists, it has a simple, flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions that are catered towards ML workflows.
    - [Metaflow](https://docs.metaflow.org/) is a human-friendly Python library that helps scientists and engineers build and manage real-life data science projects. Metaflow was originally developed at Netflix to boost the productivity of data scientists who work on a wide variety of projects from classical statistics to state-of-the-art deep learning.
    - [Kredo vs. ZenML vs. Metaflow](https://neptune.ai/blog/kedro-vs-zenml-vs-metaflow)
    - ⭐[[Apache Airflow](https://airflow.apache.org/) | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/Airflow)] Apache is a very mature and popular option initially developed to orchestrate data engineering and extract-transform-load (ETL) pipelines for analytics workloads. Airflow has expanded into the machine-learning space as a viable pipeline orchestrator. 

</details>

***

## Architecture

<details>
<summary>Expand ⬇️</summary>
<br>

- <ins>Top 5 trade-off</ins>:
  - Cost vs. Performance
  - Reliability vs. Scalability
  - Performance vs. Consistency
  - Security vs. Flexibility
  - Speed vs. Quality
    
- <ins>Every system/architecture should be built based on some basic principles</ins>:
  - **Separation of concerns**: The system should be modularised into different components with each component being a separate maintainable, reusable and extensible entity.
  - **Scalability**: The system needs to be able to scale as the traffic increases
  - **Reliability**: The system should continue to be functional even if there is software of hardware failure
  - **Availability**: The system needs to continue operating at all times
  - **Simplicity**: The system has to be as simple and intuitive as possible
    
</details>

***

## What a Data Scientist should know about MLOps

<details>
<summary>Expand ⬇️</summary>
<br>

This is a super compressed list.
- Version Control
- CI/CD
- Testing can be separated into [4 different stages](https://www.linkedin.com/pulse/qa-testing-what-dev-sit-uat-prod-kavitha-mandli/?trk=public_profile_article_view) — DEV, SIT, UAT User Acceptance Testing) and PROD
- Major cloud computing provide (AWS, GCP, Azure)
- Batch Orchestration (Airflow)
- Load Balancer
- REST API Frameworks: Flask | [Notes](https://github.com/kyaiooiayk/Flask-Notes) , Django, FastAPI | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations/tree/master/tutorials/FastAPI)
- Non-Relational & Relational Databases | [Notes](https://github.com/kyaiooiayk/MLOps-Machine-Learning-Operations)
- Real-Time Processing (Spark, Kafka)
- Containerisation: Kubernetes | [Notes](https://github.com/kyaiooiayk/Kubernetes-Notes), Docker | [Notes](https://github.com/kyaiooiayk/Docker-Notes)
</details>

***

## Other checklists
[Ref#1](https://github.com/ageron/handson-ml3/blob/main/ml-project-checklist.md) | [Ref#2](https://github.com/RJZauner/machine-learning-project-checklist) | [Ref#3](https://github.com/datastacktv/data-engineer-roadmap) | [Ref#4](https://github.com/igorbarinov/awesome-data-engineering#databases) | [Ref#5](https://theaisummer.com/best-practices-deep-learning-code/) | [awesome-production-machine-learning](https://github.com/zhimin-z/awesome-production-machine-learning) | [Made with ML](https://madewithml.com/) | [Google best practices guide on ML](https://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf?utm_source=substack&utm_medium=email)
