from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq

from few_shots import few_shots

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai.embeddings import OpenAIEmbeddings

import chromadb



def get_few_shot_db_chain():
    db_user = "root"
    db_password = "2004"
    db_host = "localhost"
    db_name = "martians_tshirt"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)

    llm = ChatGroq(
      model = "llama-3.1-70b-versatile",
      temperature = 0,
      groq_api_key = os.getenv("GROQ_API_KEY"),
    )
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )


    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}",
    )

    FewShot_Prompt = FewShotPromptTemplate(
          example_selector=example_selector,
          example_prompt=example_prompt,
          prefix="""
          

      You are a MySQL expert. Given an input question, your job is to:
      1. Create a syntactically correct SQL query.
      2. Run the query and provide only the result of the SQL query as the output.
      3. Only use the following tables:(-- discounts table
      CREATE TABLE discounts (
          discount_id INTEGER NOT NULL AUTO_INCREMENT, 
          t_shirt_id INTEGER NOT NULL, 
          pct_discount DECIMAL(5, 2), 
          PRIMARY KEY (discount_id), 
          CONSTRAINT discounts_ibfk_1 FOREIGN KEY(t_shirt_id) REFERENCES t_shirts (t_shirt_id), 
          CONSTRAINT discounts_chk_1 CHECK (pct_discount BETWEEN 0 AND 100)
      );

      -- t_shirts table
      CREATE TABLE t_shirts (
          t_shirt_id INTEGER NOT NULL AUTO_INCREMENT, 
          brand ENUM('Van Huesen', 'Levi', 'Nike', 'Adidas') NOT NULL, 
          color ENUM('Red', 'Blue', 'Black', 'White') NOT NULL, 
          size ENUM('XS', 'S', 'M', 'L', 'XL') NOT NULL, 
          price INTEGER, 
          stock_quantity INTEGER NOT NULL, 
          PRIMARY KEY (t_shirt_id), 
          CONSTRAINT t_shirts_chk_1 CHECK (price BETWEEN 10 AND 50)
      );
      )

      The response format must be strictly:
      Question: Question text
      SQLQuery: SQL query
      SQLResult: Result of the SQL query (Strictly the numerical value of SQLResult not the word SQLResult)
      """,
          suffix="""
      Only include the SQLResult in the response. Do not provide an Answer key or additional explanations.
      Question: {input}
      SQLQuery:
      """,
          input_variables=["input", "table_info", "top_k"],
    )

    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=FewShot_Prompt)
    return chain
  
if __name__ == "__main__":
  chain = get_few_shot_db_chain()
  chain.invoke("How many tshirts are left in total number of stock ?")
