#Importing All Required Libraries
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

llm = ChatGroq(model="qwen-2.5-32b")

# Routing Schema
class Route(BaseModel):
    step: Literal["Blog_Title", "Blog_Content"] = Field(None, description="The next step in the routing process")

router = llm.with_structured_output(Route)

# State class
class State(TypedDict):
    input: str
    decision: str
    output: str

# Nodes
def llm_call_1(state: State):
    """Generate Blog Title"""
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_2(state: State):
    """Generate Blog Content"""
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_router(state: State):
    """Route input to Blog_Title or Blog_Content"""
    decision = router.invoke([
        SystemMessage(content="Route the input to Blog title and Blog content based on the user's request."),
        HumanMessage(content=state["input"]),
    ])
    return {"decision": decision.step}

# Conditional Routing
def route_decision(state: State):
    return "llm_call_1" if state["decision"] == "Blog_Title" else "llm_call_2"

# Build Workflow
router_builder = StateGraph(State)
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_router", llm_call_router)
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges("llm_call_router", route_decision, {"llm_call_1": "llm_call_1", "llm_call_2": "llm_call_2"})
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_workflow = router_builder.compile()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Blog Generator</title>
    <script>
        async function generateBlog() {
            const inputText = document.getElementById("userInput").value;
            const response = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ input: inputText })
            });
            const result = await response.json();
            document.getElementById("output").innerText = result.output;
        }
    </script>
</head>
<body>
    <h1>Blog Generator</h1>
    <input type="text" id="userInput" placeholder="Enter your topic...">
    <button onclick="generateBlog()">Generate</button>
    <h2>Output:</h2>
    <p id="output"></p>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    state = router_workflow.invoke({"input": data["input"]})
    return jsonify({"output": state["output"]})

if __name__ == "__main__":
    app.run(debug=True)

