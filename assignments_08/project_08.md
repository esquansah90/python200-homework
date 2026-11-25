# Week 8 Project

## Part A: Supabase Setup

I successfully created a Supabase project named python200 using the free plan. I created and verified the weather_raw and weather_enriched tables with the required columns. I also configured my environment variables in a .env file and confirmed that Git ignores the file to protect my credentials.

## Part B: Cloud Cost Analysis

In Scenario A, a t3.micro EC2 instance costs $1.66 per month or $19.92 per year. In Scenario B, the GPU-based p3.2xlarge EC2 instance, along with RDS and S3 storage, costs $2,393.68 per month or $28,724.16 per year. I was surprised by how much more expensive Scenario B was compared to Scenario A. This comparison showed me that GPU instances are worth the higher cost for demanding tasks like machine learning and AI training, but they are not necessary for simple applications or workloads that can run on smaller, less expensive instances.

## YouTube Video

https://youtu.be/s7hUpkJaVNI
