import time
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Make sure this uses your Excel data

# Langchain model setup
model = OllamaLLM(model="llama3.2")

# Prompt template
template = """
You are a skincare advisor who only recommends products available with delivery to Tunisia.

Skin concern: {skin_concern}

Here are available products:
Product Name | Target Concern(s) | Price (USD/TND) | Buy Link
{products_table}

Question from user: {question}

Your task:
- Recommend only relevant products available in Tunisia for the user's concern.
- Provide prices (convert to TND if needed).
- Propose a complete routine if useful (morning/evening).
- Highlight the best price-quality ratio products.
- Be clear, concise, and avoid recommending unavailable products.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit interface
st.set_page_config(page_title="Tunisian Skincare Advisor", page_icon="💆‍♀️")

st.title("💆‍♀️ Personalized Skincare Advisor")
st.markdown("Enter your skin concern and ask your skincare question. I'll recommend the best products available in **Tunisia** 🇹🇳.")

with st.form("skin_form"):
    user_concern = st.text_input("🧴 What skin issues are you facing? (e.g., acne, dryness, redness, dullness)")
    question = st.text_area("❓ Ask your skincare question")
    submitted = st.form_submit_button("💡 Get Recommendation")

if submitted and user_concern and question:
    with st.spinner("Analyzing your skin needs..."):
        reviews = retriever.invoke(user_concern)

        result = chain.invoke({
            "skin_concern": user_concern,
            "products_table": reviews,
            "question": question
        })

    st.success("Here’s what I recommend for you:")
    st.markdown(f"```markdown\n{result}\n```")
