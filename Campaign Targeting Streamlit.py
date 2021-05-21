import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import pickle
from PIL import Image
from datetime import datetime
import base64
from Data_Preprocessing import data_preprocessing

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="big-font">Menu & Filters</p>', unsafe_allow_html=True)
st.title("Marketing Campaign")
st.write('The following application analyses customer data in order to predict who might converge when targeted by a specific media campaign.'
         ' A response model can provide a significant boost to the efficiency of a marketing campaign by reducing expenses. The objective is'
         ' to predict who will respond to a campaign for a product or service.')
st.write('The required column names are the following: ')

col1, col2 = st.beta_columns(2)

col1.write('AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise')
col1.write('AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise')
col1.write('AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise')
col1.write('AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise')
col1.write('AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise')
col1.write('Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise')
col1.write('Complain - 1 if customer complained in the last 2 years')
col1.write('DtCustomer - date of customer’s enrolment with the company')
col1.write('Education - customer’s level of education')
col1.write('Marital - customer’s marital status')
col1.write('Kidhome - number of small children in customer’s household')
col1.write('Teenhome - number of teenagers in customer’s household')
col2.write('Income - customer’s yearly household income')
col2.write('MntFishProducts - amount spent on fish products in the last 2 years')
col2.write('MntMeatProducts - amount spent on meat products in the last 2 years')
col2.write('MntFruits - amount spent on fruits products in the last 2 years')
col2.write('MntSweetProducts - amount spent on sweet products in the last 2 years')
col2.write('MntWines - amount spent on wine products in the last 2 years')
col2.write('MntGoldProds - amount spent on gold products in the last 2 year')
col2.write('NumDealsPurchases - number of purchases made with discount')
col2.write('NumCatalogPurchases - number of purchases made using catalogue')
col2.write('NumStorePurchases - number of purchases made directly in stores')
col2.write('NumWebVisitsMonth - number of visits to company’s web site in the last month')
col2.write('Recency - number of days since the last purchase')

data = pd.DataFrame()
data = pd.read_csv("data.csv")

uploaded_file = st.sidebar.file_uploader("Upload your own data", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)

if len(data) > 0:
    data_copied = data.copy()

    data_copied["Dt_Customer"] = pd.to_datetime(data_copied["Dt_Customer"])

    #  Marital Status Filter
    marital_filter = st.sidebar.multiselect('Marital Status:', list(set(data_copied['Marital_Status'])),
                                         default=list(set(data_copied['Marital_Status'])))
    data_to_use = data_copied.loc[data_copied['Marital_Status'].isin(marital_filter)]

    #  Education Filter
    Education_filter = st.sidebar.multiselect('Education Level:', list(set(data_to_use['Education'])),
                                            default=list(set(data_to_use['Education'])))
    data_to_use = data_to_use.loc[data_to_use['Education'].isin(Education_filter)]

    #  Income Filter
    income_slider = st.sidebar.slider('Income', min(data_to_use['Income']),
                                      max(data_to_use['Income']), (min(data_to_use['Income']), max(data_to_use['Income'])))
    data_to_use = data_to_use.loc[(data_to_use['Income'] <= (income_slider[1])) & (data_to_use['Income'] >= (income_slider[0]))]

    #start_date = st.sidebar.date_input('Start Date for Customer Enrolment', min(data_to_use['Dt_Customer']))
    #end_date = st.sidebar.date_input('End Date for Customer Enrolment', max(data_to_use['Dt_Customer']))

    st.subheader('Raw data - Top 50')
    st.write(data_to_use.head(50))

    col1, col2 = st.beta_columns(2)

    #  Education Levels bar chart
    education = data_to_use['Education'].value_counts()
    education = pd.DataFrame({'Level': education.index, 'Count': education.values})
    education_fig = px.bar(education, x='Level', y='Count', title='Education Levels')
    education_fig.update_traces(marker_color='brown')
    col1.plotly_chart(education_fig)

    #  Marital Status Bar Chart
    marital_status = data_to_use['Marital_Status'].value_counts()
    marital_status = pd.DataFrame({'Marital_Status': marital_status.index, 'Count': marital_status.values})
    marital_status_fig = px.bar(marital_status, x='Marital_Status', y='Count', title='Marital Status')
    marital_status_fig.update_traces(marker_color='brown')
    col2.plotly_chart(marital_status_fig)

    col1, col2 = st.beta_columns(2)

    #  Income vs. Amount Spent on fruits
    income_fruits = px.line(data_to_use.sort_values(by=['Income']), x='Income', y='MntFruits', title='Income vs. Amount Spent on Fruits')
    col1.plotly_chart(income_fruits)

    #  Income vs. Amount Spend on Mean Products
    income_meat = px.line(data_to_use.sort_values(by=['Income']), x='Income', y='MntMeatProducts', title='Income vs. Amount Spent on Meat Products')
    col2.plotly_chart(income_meat)

    col1, col2 = st.beta_columns(2)

    #  Income vs. Number of Web Purchases
    NumWebPurchases = px.line(data_to_use.sort_values(by=['Income']), x='Income', y='NumWebPurchases',title='Income vs. Number of Web Purchases')
    col1.plotly_chart(NumWebPurchases)

    #  Income vs. Number of Web Visits per Month
    NumWebVisitsMonth = px.line(data_to_use.sort_values(by=['Income']), x='Income', y='NumWebVisitsMonth',
                              title='Income vs. Number of Web Visits per Month')
    col2.plotly_chart(NumWebVisitsMonth)

    ## Heat map correlation between dimensions
    data_to_use.drop(['Z_Revenue', 'Z_CostContact', 'ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                      'AcceptedCmp4', 'AcceptedCmp5'], axis=1, inplace=True)
    corr = data_to_use.corr()
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.index.values,
        y=corr.columns.values,
        colorscale='oxy'))

    heatmap_fig.update_layout(title='Dimension Correlation', width=1500, height=700)
    st.plotly_chart(heatmap_fig)

    #  Clean data for customer conversion model
    data_ML = data_preprocessing(data)
    customer_ids = data_ML["Customer_ID"].tolist()
    X_data_to_use = data_ML.drop(["Customer_ID"], axis=1)
    responses = loaded_model.predict(X_data_to_use).tolist()
    dict_data = {'Customer_ID': customer_ids, 'Response': responses}
    potential_customers = pd.DataFrame(dict_data, columns=['Customer_ID', 'Response'])
    length_success = len(potential_customers[potential_customers["Response"] == 1])
    length_failure = len(potential_customers[potential_customers["Response"] == 0])
    avg_cost_contact = data["Z_CostContact"].mean()
    avg_revenue_contact = data['Z_Revenue'].mean()
    cost = [length_failure*avg_cost_contact + length_success*avg_cost_contact, length_success*avg_cost_contact]
    revenue = [length_success * avg_revenue_contact, length_success * avg_revenue_contact]
    profit = [revenue[0]-cost[0], revenue[1]-cost[1]]
    categories = ["Target All", 'Target Predicted Successful']

    title = 'Which customers shall be targeted in the next campaign? Out of ' + str(length_failure+length_success) + ' leads,' + str(length_success) + ' would converge.'
    st.title(title)

    col1, col2 = st.beta_columns(2)

    #  Prediction table for Conversions
    table_predictions = go.Figure(data=[go.Table(
        header=dict(values=list(potential_customers.columns),
                    align='left'),
        cells=dict(values=[potential_customers['Customer_ID'], potential_customers['Response']],
                   align='left'))
    ])

    table_predictions.update_layout(title='Potential Customer Predictions')


    #  Cost per target audience
    cost_fig = px.bar(x=categories, y=cost, title='Cost per Target Audience')
    cost_fig.update_traces(marker_color='brown')
    col1.plotly_chart(table_predictions)
    col2.plotly_chart(cost_fig)

    col1, col2 = st.beta_columns(2)

    #  Revenue per Target Audience
    revenue_fig = px.bar(x=categories, y=revenue, title='Revenue per Target Audience')
    revenue_fig.update_traces(marker_color='brown')

    #  Profit per Target Audience
    profit_fig = px.bar(x=categories, y=profit, title='Profit per Target Audience')
    profit_fig.update_traces(marker_color='brown')

    col1.plotly_chart(revenue_fig)
    col2.plotly_chart(profit_fig)

    #  Download Results
    csv = potential_customers.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="campaign-predictions.csv" target="_blank">Download final result as csv file</a>'
    st.sidebar.markdown("") # for spacing
    st.sidebar.markdown("") # for spacing
    st.sidebar.markdown(href, unsafe_allow_html=True)





