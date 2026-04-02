from langchain_anthropic import ChatAnthropic

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, List, Annotated
import operator

urls = ["www.anthropic.com/engineering", "openai.com/news/"]

tools = [
    {
        "type": "web_fetch_20260209", 
        "name": "web_fetch"
    }
]

model = ChatAnthropic(model="claude-opus-4-6").bind_tools(tools)

def merge_dicts(current: dict, update: dict) -> dict:
    return {**current, **update}


class ResearchState(TypedDict):
    llm_calls: Annotated[int, operator.add]
    audience_focus: str
    processed_content: Annotated[dict, merge_dicts]
    messages: Annotated[str, operator.add]
    urls: List[str]
    current_url: str
    finished: bool = False


def controller(state: ResearchState) -> ResearchState:
    urls = state['urls']
    if len(urls) == 0:
        return {
            "finished": True
        }
    
    curr_url = state['urls'].pop()

    return {
        "current_url": curr_url,
    }

def research_url(state: ResearchState) -> ResearchState:
    return

def research_outer(state: ResearchState) -> ResearchState:
    return

def synthesize(state: ResearchState) -> ResearchState:
    return


def decide_to_deep_dive(state: ResearchState):
      #llm decides here whether or not to process or elucidate on content

      if state.get("needs_deep_dive"):
          return "research_outer"
      else:
          return "controller"

def decide_to_finish(state: ResearchState):
    if state.get("finished"):
          return "synthesize"
    else:
        return "research_url"


# model = init_chat_model(os.environ["CLAUDE_MODEL"])

if __name__ == "__main__":
    graph = StateGraph(ResearchState)
    graph.add_node('controller', controller)
    graph.add_node('research_url', research_url)
    graph.add_node('research_outer', research_outer)
    graph.add_node('synthesize', synthesize)

    graph.add_edge(START, "controller")
    graph.add_conditional_edges('controller', decide_to_finish, ["research_url", "synthesize"])

    graph.add_conditional_edges('research_url', decide_to_deep_dive, ["research_outer", "controller"])


    graph.add_edge('controller', 'synthesize')
    graph.add_edge('synthesize', END)

    compiled = graph.compile()
    # test = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
    # print(test)