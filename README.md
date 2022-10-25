# Crypto-Trading

Just a few notes about the Python scripts:

The scripts outlined here assume that sensitive information is stored in a file called Credentials.py in a directory called Credentials where the files are being ran from.

CreateDatabsae.py - This outlines how to take a Pandas DataFrame and add it to a PostgreSQL (although this should work for other SQL databases) table.This script assumes that the database has already been created, and database connection credentials are stored in a directory

MessingAroundForex.py - This file outlines how to use some TA-Lib indicators as well as trying to figure out how to plot the different indicators. I also was testing/figure out how to find the ticket symbol for YFinance data.

PredictionModel.py - This my attempt to learn Karas and TensorFlow. I've piecemealed this script from various YouTube channels and Udemy courses that I will link below for interest.

Udemy:
https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/

Youtube:
https://www.youtube.com/c/NeuralNine
https://www.youtube.com/c/RohanPaul-AI
https://www.youtube.com/c/CodeTradingCafe
