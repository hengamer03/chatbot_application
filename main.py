from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


local_llm = "llama3.1"
model = ChatOllama(model=local_llm, temperature=0)

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a chatbot. you follow the users instructions and awer questions in a friendly and helpful way.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

while True:
    try:
        query = input("Enter your query: ")
        input_messages = [HumanMessage(query)]
        output = app.invoke({"messages": input_messages}, config)
        output["messages"][-1].pretty_print()  # output contains all messages in state
    except Exception as e:
        print(e)
        break