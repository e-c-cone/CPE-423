# CPE-423
BEEGUS: Bicameral Election Estimator for General Elections of the United States

BEEGUS is an AI-driven election estimator, designed with forecasting U.S. congressional elections in mind. It is developed as a capstone project within the Electrical and Computer Engineering Department of Stevens Institute of Technology.

The goal of BEEGUS is to predict the outcome of public elections with a robust understanding of as much available data as possible. The AI will additionally generate a candidate report card, highlighting which issues are key to a given region. Candidates will be able to gain a deeper understanding of crucial contention points and utilize this as a competitive edge on their opponents. 

To accomplish this, BEEGUS will weigh and aggregate several forms of available information.

1) Web Scraping: A natural language processing (NLP) sentiment prediction model will scrape information from influencial social media platforms such as Twitter. By conducting real-time analysis of social media posts sentiments and volume, BEEGUS will be able to generalize community consensus.

2) Regional Statistics: In modern politics, it is uncommon to see a wide-scale transition in a community overnight. By understanding previous regional outcomes and critical voter issues, BEEGUS will be able to understand the direction of future voters.

3) Candidate Data: With candidate policy information readily available, our model will cross-reference candidate platforms with the needs of the community they wish to represent.

4) Economic Data: This branch of the model will analyze the current state of the economy, as well as socioeconomic status of the voting community.

These various election prediction tactics will be fed as input data into three machine learning models. These will include a long short-term memory, a logistic regression, and a neural network. The three models will jointly weighted to generate the optimal candidate characteristics. A report card will subsequently be generated for a given candidate indicating deviation from the optimal characteristics desired by the community.

 
