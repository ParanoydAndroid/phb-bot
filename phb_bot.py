import dotenv
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

dotenv.load_dotenv()
settings = dotenv.dotenv_values(dotenv_path=dotenv.find_dotenv())

# loading from disk
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=settings.get('DB_PATH', './phb_db'), embedding_function=embedding)

retriever = vectorstore.as_retriever()

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

qa_system_prompt = """You are an expert dungeon master for dungeons and dragons and know the rules very well.
Use the following pieces of retrieved rules and context to answer the question.
Answer the question with as much detail as possible.
If you don't know the answer, just say that you don't know.
If a rule is ambiguous, point out the ambiguity.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)[:4096]


def contextualized_question(input: dict):
    if input.get("chat_history"):
        ans = contextualize_q_chain.invoke(input)
        print(f'Generated Question: {ans}')
        return contextualize_q_chain
    else:
        return input["question"]


rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

chat_history = []


def main():
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        result1 = rag_chain.invoke(
            {"question": user_input, "chat_history": chat_history}
        )

        print(result1)

        chat_history.extend(
            [HumanMessage(content=user_input), AIMessage(content=result1)]
        )


if __name__ == "__main__":
    main()
