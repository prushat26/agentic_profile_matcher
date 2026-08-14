# export_graph.py
from matching_agent import agent_graph

# Generate the Mermaid markup string from your LangGraph instance
mermaid_code = agent_graph.get_graph().draw_mermaid()

# Save it to a file
with open("workflow_graph.mmd", "w", encoding="utf-8") as f:
    f.write(mermaid_code)

print("✅ Saved workflow_graph.mmd successfully!")