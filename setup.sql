USE telecom;
SET autocommit = 0;

-- Offer
UPDATE telecom_churn2 SET Offer = 'No Offer' WHERE Offer IS NULL;

-- Phone service related
UPDATE telecom_churn2 SET `Avg Monthly Long Distance Charges` = 0 WHERE `Avg Monthly Long Distance Charges` IS NULL;
UPDATE telecom_churn2 SET `Multiple Lines` = 'No Phone Service' WHERE `Multiple Lines` IS NULL;

-- Internet service related
UPDATE telecom_churn2 SET `Internet Type` = 'No Internet' WHERE `Internet Type` IS NULL;
UPDATE telecom_churn2 SET `Avg Monthly GB Download` = 0 WHERE `Avg Monthly GB Download` IS NULL;
UPDATE telecom_churn2 SET `Online Security` = 'No Internet Service' WHERE `Online Security` IS NULL;
UPDATE telecom_churn2 SET `Online Backup` = 'No Internet Service' WHERE `Online Backup` IS NULL;
UPDATE telecom_churn2 SET `Device Protection Plan` = 'No Internet Service' WHERE `Device Protection Plan` IS NULL;
UPDATE telecom_churn2 SET `Premium Tech Support` = 'No Internet Service' WHERE `Premium Tech Support` IS NULL;
UPDATE telecom_churn2 SET `Streaming TV` = 'No Internet Service' WHERE `Streaming TV` IS NULL;
UPDATE telecom_churn2 SET `Streaming Movies` = 'No Internet Service' WHERE `Streaming Movies` IS NULL;
UPDATE telecom_churn2 SET `Streaming Music` = 'No Internet Service' WHERE `Streaming Music` IS NULL;
UPDATE telecom_churn2 SET `Unlimited Data` = 'No Internet Service' WHERE `Unlimited Data` IS NULL;

-- Churn category
UPDATE telecom_churn2 SET `Churn Category` = 'No Churn' WHERE `Churn Category` IS NULL;

commit;