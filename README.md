# Smart Underwriter

## Background 

The underwriting process is gradually becoming more and more
automatic. Fannie Mae and Freddie Mac, two major US housing GSE (government
sponsor enterprise) have their own AUS (automated underwriting system): [Desktop
Underwriter](https://www.fanniemae.com/singlefamily/desktop-underwriter "Desktop
Underwriter") and [Loan Prospector](http://www.loanprospector.com/ "Loan
Prospector"). The AUS system is great as it can provide an objective and fast
decision based on the mortgage data.

Fannie Mae is the largest housing mortgage backer in US housing market. It has
released "a subset of Fannie Mae’s 30-year, fully amortizing, full
documentation, single-family, conventional fixed-rate mortgages" on its website
to "promote better understanding of the credit performance of Fannie Mae
mortgage loans". This data is also a perfect source to build our own mortgage
risk assessment model.

In this project, those Fannie Mae data was downloaded, compiled, aggregated and
then fed into a machine learning model to build a credit risk prediction
model. You can find the demo site at [Here]().

## Workflow

To recreate data processing, modeling and web development. You can follow the
following steps:

1. Download the dataset at
   [Fannie Mae's website](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html
   "Download the data"). You can use the `download.sh` script in the `processing`
   folder. But keep in mind that you will need to supply a separate cookie file
   in order to download the data. If you are using Firefox, you can install
   [Export Cookie](https://addons.mozilla.org/en-US/firefox/addon/export-cookies/)
   extension. 
   
2. Aggregate the loan performance data. At this time, I am only focusing on the
   terminal status of the loan. `data_process.py` will get the last status of
   each loan and disgard any intermediate status. In the future, a time-series
   based model will be developed to predict the time dependent loan status.
   **CAUTION:** Data was processed on a DO droplet containing 16G
   memory. Current script uses Python pandas to process the data. My plan is to
   rewrite the whole thing by Spark.
   
3. Further aggregate quarterly data into yearly and then
   multi-years. `merged-quarter.py` and `merged-year.py`.
   
4. Use the `learning.py` script to do the machine learning. Currently, logistic
   model and stochastic gradient descent (SGD) based support vector machine algorithm
   are used. SGD gives better AUC-ROC value so it is picked. 
   
5. Run flask web server by using `python3 run.py`.

## License

(C) copyright by Zhenqing Li. GPL v3

