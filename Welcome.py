import streamlit as st

# Set page title and background color
st.set_page_config(page_title="Malaria Detection using Machine Learning", page_icon=":microscope:", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(180deg, #f5f5f5, #e8e8e8) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display page header and overview of malaria
st.write("<h1 style='text-align: center;'>Malaria Detection using Machine Learning</h1>", unsafe_allow_html=True)
st.write("")
st.write("")
st.header("Overview on Malaria")
st.write("##### Malaria is an acute febrile illness caused by Plasmodium parasites, which are spread to people through the bites of infected female Anopheles mosquitoes. There are 5 parasite species that cause malaria in humans, and 2 of these species – P. falciparum and P. vivax – pose the greatest threat. P. falciparum is the deadliest malaria parasite and the most prevalent on the African continent. P. vivax is the dominant malaria parasite in most countries outside of sub-Saharan Africa.")
st.write("")
st.write("")
st.markdown("## Symptoms of Malaria")
st.write("")

# Display the list of symptoms in a pretty format
malaria_symptoms = [
    "Fever",
    "Chills",
    "Headache",
    "Sweats",
    "Nausea",
    "Vomiting",
    "Body aches",
    "Fatigue",
    "Cough"
]
symptoms_col1, symptoms_col2, symptoms_col3 = st.columns(3)
for i in range(3):
    with symptoms_col1:
        st.write(f"<h4>{malaria_symptoms[i]}</h4>", unsafe_allow_html=True)
    with symptoms_col2:
        st.write(f"<h4>{malaria_symptoms[i+3]}</h4>", unsafe_allow_html=True)
    with symptoms_col3:
        st.write(f"<h4>{malaria_symptoms[i+6]}</h4>", unsafe_allow_html=True)

# Add some padding to the bottom of the page
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
