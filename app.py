
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="SpendWise AI", layout="wide")
st.title("💰 SpendWise AI Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx","csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)

    page = st.sidebar.selectbox(
        "Navigation",
        ["Executive Summary","Descriptive Analytics","Customer Segmentation",
         "Association Rules","Predictive Analytics","Prescriptive Analytics",
         "Customer Scorer","Model Evaluation"]
    )

    if page == "Executive Summary":
        st.header("Executive Summary")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Customers", len(df))
        c2.metric("Avg Income", f"{df['Income'].mean():,.0f}")
        c3.metric("Avg Expenses", f"{df['Expenses'].mean():,.0f}")
        c4.metric("Avg Savings %", f"{df['Savings'].mean():.2f}")

        c5,c6,c7,c8 = st.columns(4)
        c5.metric("App Interest %", round((df["App_Interest"]=="Yes").mean()*100,1))
        c6.metric("High Stress %", round(df["Financial_Stress"].astype(str).str.contains("High",case=False).mean()*100,1))
        c7.metric("Top Occupation", df["Occupation"].mode()[0])
        c8.metric("Avg Savings", round(df["Savings"].mean(),2))

        st.plotly_chart(px.histogram(df,x="Income",title="Income Distribution"),use_container_width=True)
        st.plotly_chart(px.histogram(df,x="Expenses",title="Expense Distribution"),use_container_width=True)
        st.plotly_chart(px.histogram(df,x="Savings",title="Savings Distribution"),use_container_width=True)

    elif page == "Descriptive Analytics":
        st.header("Descriptive Analytics")
        st.plotly_chart(px.histogram(df,x="Age",title="Age Distribution"),use_container_width=True)
        st.plotly_chart(px.bar(df["Occupation"].value_counts().reset_index(),
                               x="Occupation",y="count",title="Occupation Distribution"),use_container_width=True)
        st.plotly_chart(px.pie(df,names="Gender",title="Gender Distribution"),use_container_width=True)
        st.plotly_chart(px.pie(df,names="Financial_Stress",title="Financial Stress"),use_container_width=True)
        st.plotly_chart(px.bar(df["City_Tier"].value_counts().reset_index(),
                               x="City_Tier",y="count",title="City Tier Distribution"),use_container_width=True)
        st.plotly_chart(px.scatter(df,x="Income",y="Expenses",color="Occupation",
                                   title="Income vs Expenses"),use_container_width=True)

    elif page == "Customer Segmentation":
        st.header("K-Means Customer Segmentation")
        X = df[["Income","Expenses","Savings"]]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(X)

        names = {0:"Budget Conscious",1:"High Income Professionals",2:"Average Earners"}
        df["Segment"] = df["Cluster"].map(names)

        st.plotly_chart(px.scatter(df,x="Income",y="Savings",color="Segment"),
                        use_container_width=True)

        st.subheader("Cluster Centroids")
        st.dataframe(df.groupby("Segment")[["Income","Expenses","Savings"]].mean())

    elif page == "Association Rules":
        st.header("Apriori Association Rules")

        basket = pd.DataFrame()
        categories = sorted(set(df["Category_1"]).union(df["Category_2"]).union(df["Category_3"]))

        for cat in categories:
            basket[cat] = (
                (df["Category_1"]==cat) |
                (df["Category_2"]==cat) |
                (df["Category_3"]==cat)
            ).astype(int)

        frequent = apriori(basket, min_support=0.005, use_colnames=True)
        rules = association_rules(frequent, metric="confidence", min_threshold=0.1)

        if len(rules):
            st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
        else:
            st.warning("No rules found.")

    elif page == "Predictive Analytics":
        st.header("App Interest Prediction")

        data = df.copy()
        for col in data.columns:
            if data[col].dtype == "object":
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))

        X = data.drop("App_Interest", axis=1)
        y = data["App_Interest"]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        model = RandomForestClassifier(n_estimators=100,random_state=42)
        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        st.metric("Accuracy", round(accuracy_score(y_test,pred),3))
        st.dataframe(pd.DataFrame(confusion_matrix(y_test,pred)))

    elif page == "Prescriptive Analytics":
        st.header("Recommendations")
        threshold = st.slider("Savings Threshold",0,100,10)

        if threshold < 10:
            st.error("High Risk: Reduce impulse purchases and entertainment expenses.")
        elif threshold < 20:
            st.warning("Moderate Risk: Track spending weekly and improve budgeting.")
        else:
            st.success("Low Risk: Maintain current savings discipline and invest surplus funds.")

    elif page == "Customer Scorer":
        st.header("Customer Scorer")

        age = st.number_input("Age",18,80,25)
        income = st.number_input("Income",0,500000,50000)
        expenses = st.number_input("Expenses",0,500000,25000)

        savings = max(0, ((income-expenses)/max(income,1))*100)

        risk = "Low" if income-expenses > 20000 else "Moderate" if income-expenses > 10000 else "High"
        segment = "High Income" if income > 70000 else "Average" if income > 30000 else "Budget"

        st.metric("Predicted Savings %", round(savings,2))
        st.metric("Risk Level", risk)
        st.metric("Customer Segment", segment)

    elif page == "Model Evaluation":
        st.header("Model Evaluation")

        data = df.copy()
        for col in data.columns:
            if data[col].dtype == "object":
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))

        X = data.drop("App_Interest", axis=1)
        y = data["App_Interest"]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        model = RandomForestClassifier(n_estimators=100,random_state=42)
        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        st.metric("Accuracy", round(accuracy_score(y_test,pred),3))
        st.dataframe(pd.DataFrame(confusion_matrix(y_test,pred)))
        st.text(classification_report(y_test,pred))
else:
    st.info("Upload your SpendWise dataset to begin.")
