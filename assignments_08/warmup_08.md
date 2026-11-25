# Week 8 Warmup

## Part 1: Cloud Concepts

### Question 1

Cloud computing uses a pay-as-you-go model, where you pay for the computing resources you use instead of buying physical servers. Owning servers requires upfront costs, maintenance, and upgrades, while cloud computing allows you to adjust resources based on your needs.

### Question 2

Vertical scaling means upgrading an existing machine with more CPU, RAM, or processing power.
Horizontal scaling means adding more machines to share the workload.

Scenario 1: Horizontal scaling is best because adding more servers helps the web application handle the sudden increase in users.

Scenario 2: Vertical scaling is best because upgrading to a machine with a faster GPU and more RAM improves model training performance.

Scenario 3:Horizontal scaling is best because multiple machines can process the 10,000 files simultaneously.
Your vertical and horizontal scaling definitions.

### Question 3

Gmail — SaaS: Provides a ready-to-use email application.
Azure Virtual Machines — IaaS: Provides virtual servers that users configure and manage.
AWS S3 — IaaS: Provides cloud storage infrastructure without requiring physical servers.
GitHub Codespaces — PaaS: Provides a managed cloud development environment.
Snowflake — PaaS: Provides a managed platform for storing and analyzing data.
Supabase — BaaS: Provides ready-made backend services such as databases and authentication.

IaaS (Infrastructure as a Service): Provides virtual infrastructure. For example, Azure Virtual Machines lets developers manage operating systems, applications, and data.

PaaS (Platform as a Service): Provides a managed platform for developing applications. For example, Snowflake manages the underlying infrastructure while developers manage their data and queries.

SaaS (Software as a Service): Provides ready-to-use software. For example, Gmail manages the application and infrastructure, while users manage their emails and settings.

### Question 4

A managed data platform like Snowflake or Databricks provides ready-to-use tools for storing, processing, and analyzing data. Unlike AWS or GCP, which offer more control over infrastructure, managed platforms handle much of the setup and maintenance. You gain simplicity and convenience but give up some flexibility and control.

### Question 5

Cloud computing may not be the best choice when a company has predictable workloads and already owns reliable servers. Using existing servers may be cheaper than paying ongoing cloud costs.

## Part 2: Cloud Landscape

## Question 1

Amazon Web Services (AWS): Offers a wide range of cloud services and is commonly used by startups and large businesses.

Microsoft Azure: Integrates well with Microsoft products and is popular among large organizations already using Microsoft tools.

Google Cloud Platform (GCP): Strong in data analytics, AI, and machine learning, making it useful for data-driven and technology companies.

### Question 2

Cost: Supabase offers a free tier that makes it easier for students to practice without paying for cloud services.

Simplicity: Supabase is easier to set up and use for beginner projects.

Learning focus: Supabase allows students to focus on databases and application development instead of managing complex cloud infrastructure.

Reflection: When choosing a cloud tool, I should consider cost, ease of use, and whether it meets my project's needs.

### Question 3

Storing 10 TB of images
Object storage: Amazon S3 stores large amounts of files and allows them to be retrieved when needed.

Running ML training on a GPU for four hours
GPU compute: Amazon EC2 GPU instances provide powerful computing resources that can be shut down after training.

Hosting an automatically scaling web API
Serverless compute: AWS Lambda automatically scales based on incoming requests.

Sending data to a large language model
LLM API: OpenAI API processes structured input and generates text responses.

### Question 4

I would create a weather analysis project using Supabase to store weather data and the OpenAI API to generate weather summaries.

Using one provider could simplify management and billing, but I might lose access to specialized tools and flexibility offered by other providers.
