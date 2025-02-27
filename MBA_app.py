import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from MBA_backend import generate_rules
import io

# App title
st.title("Market Basket Analysis Recommendation System")

# File upload
uploaded_file = st.file_uploader("Upload your transaction dataset (CSV format)", type=["csv"])

# Sidebar for parameters
st.sidebar.header("Apriori Algorithm Parameters")
min_support = st.sidebar.slider("Minimum Support", 0.001, 0.01, 0.0045, 0.0001)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.2, 0.05)
min_lift = st.sidebar.slider("Minimum Lift", 1.0, 5.0, 3.0, 0.1)

if uploaded_file is not None:
    # Generate association rules
    with st.spinner("Processing the data..."):
        recommendations_df = generate_rules(uploaded_file, min_support, min_confidence, min_lift)
    
    # Display recommendations
    st.subheader("Generated Recommendations")
    st.write(recommendations_df)

    # Allow downloading results
    st.download_button(
        label="Download Recommendations as CSV",
        data=recommendations_df.to_csv(index=False),
        file_name="mba_recommendations.csv",
        mime="text/csv"
    )

    # Visualization options
    st.subheader("Visualizations")

    # Function to save the plot as a JPEG image
    def save_plot_as_jpeg(fig, filename):
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg")
        buf.seek(0)
        return buf

    # Bar Chart for Top Recommendations
    if st.checkbox("Show Bar Chart for Top Recommendations"):
        item = st.selectbox("Select a Product (Antecedent Item)", recommendations_df['Antecedent'].unique())
        top_recs = recommendations_df[recommendations_df['Antecedent'] == item].sort_values(by='Lift', ascending=False)
        
        if not top_recs.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Lift", y="Consequent", data=top_recs, palette="viridis")
            plt.title(f"Top Recommendations for '{item}'")
            plt.xlabel("Lift")
            plt.ylabel("Associated Product (Consequent)")

            # Save the plot as a JPEG image
            plot_buf = save_plot_as_jpeg(plt, "top_recommendations")
            st.pyplot(plt)

            # Provide a download button for the plot
            st.download_button(
                label="Download Bar Chart as JPEG",
                data=plot_buf,
                file_name="top_recommendations.jpg",
                mime="image/jpeg"
            )
        else:
            st.write("No recommendations found for this item.")

    # Network Graph for Association Rules
    if st.checkbox("Show Network Graph for Association Rules"):
        G = nx.Graph()

        # Add edges to the graph
        for index, row in recommendations_df.iterrows():
            G.add_edge(row['Antecedent'], row['Consequent'], weight=row['Lift'])

        # Plot the network graph
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue", alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, edge_color="gray")
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

        plt.title("Network Graph of Association Rules (Items Relationships)")

        # Save the network graph as a JPEG image
        network_graph_buf = save_plot_as_jpeg(plt, "network_graph")
        st.pyplot(plt)

        # Provide a download button for the network graph
        st.download_button(
            label="Download Network Graph as JPEG",
            data=network_graph_buf,
            file_name="network_graph.jpg",
            mime="image/jpeg"
        )

