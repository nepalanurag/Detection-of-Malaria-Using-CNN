# Contents of ~/my_app/pages/page_3.py
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from base64 import b64encode


df = pd.read_csv('data.csv')
st.markdown("# Malaria Data Visualized")
user_menu = st.sidebar.radio(
    'Select an option',
    ('Line Graph', 'Chloropeth Map','Report')
)

country = df.Country.unique()
y = df.Year.unique()
def interactive_plot():
    col1, col2 = st.columns(2)    

    x_axis_val = col1.selectbox('Select a country', options=country)
    y_axis_val = col2.selectbox('Select the Y-axis', options=['No. of cases','No. of deaths'])
    st.markdown("Line graph of " + y_axis_val + " in " +x_axis_val )
    df1=df[df['Country']==x_axis_val]
    plot = px.line(df1, x=df1.Year, y=y_axis_val)
    st.plotly_chart(plot, use_container_width=False)   
def map():
    
    selected_year = st.select_slider('Choose a year', options=range(2000, 2018))

    selected_data = df[df['Year'] == selected_year]
    selected_data = selected_data.sort_values('No. of cases', ascending=False)

    st.markdown(f"## Malaria cases in {selected_year}")
    st.markdown(f"*Source: World Health")

    fig = go.Figure(data=go.Choropleth(
        locations=selected_data['Country'],
        locationmode='country names',
        z=selected_data['No. of cases'],
        zauto=False,
        zmin=min(df['No. of cases'].values),
        zmax=max(df['No. of cases'].values),
        text=selected_data['No. of cases'],
        colorscale=[[0, '#FFFFFF'], [1, '#FF0000']],
        autocolorscale=False,
        reversescale=False,
        marker_line_color='black',
        marker_line_width=0.5,
        colorbar_tickprefix=' ',
        colorbar_title='Number of Malaria Cases',
    ))

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        font=dict(family='Comic Sans', size=14, color='#FF0000'),
    )
    st.plotly_chart(fig, use_container_width=True)

    top_five = selected_data.head(5)
    most_cases = f"### Countries with most malaria cases: \n\n"

    for index, row in top_five.iterrows():
        most_cases += f" {row['Country']} : {row['No. of cases']}\n\n"

    st.markdown(most_cases)  
def report():
    pdf_file = open("reports.pdf", "rb").read()
    st.markdown(f'<iframe src="data:application/pdf;base64,{b64encode(pdf_file).decode()}" width="800" height="800"></iframe>', unsafe_allow_html=True)

if user_menu=="Line Graph":
    interactive_plot()
elif user_menu=='Chloropeth Map':
    map()
else:
    report()


    

st.sidebar.markdown("# Visualization")


