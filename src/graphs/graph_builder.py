from langgraph.graph import StateGraph, START, END
from src.states.blog_state import BlogState
from src.llms.groq_llm import GroqLLM
from src.nodes.blog_node import BlogNode


class GraphBuilder:
    """
    A class to build and manage the state graph for the blog generation
    process.
    """

    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(BlogState)

    def build_topic_graph(self):
        """
        Builds a graph to generate blogs based on topics.
        """
        self.blog_node_obj = BlogNode(self.llm)
        self.graph.add_node(
            "title_creation",
            self.blog_node_obj.title_creation
        )
        self.graph.add_node(
            "content_generation",
            self.blog_node_obj.content_generation
        )

        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", END)

        return self.graph

    def build_language_graph(self):
        """
        Builds a graph to generate blogs based on topics and languages
        and inputs.
        """
        self.blog_node_obj = BlogNode(self.llm)
        self.graph.add_node(
            "title_creation",
            self.blog_node_obj.title_creation
        )
        self.graph.add_node(
            "content_generation",
            self.blog_node_obj.content_generation
        )
        self.graph.add_node(
            "hindi_translation",
            lambda state: self.blog_node_obj.translation(
                {
                    **state,
                    "current_language": "hindi"
                }
            )
        )
        self.graph.add_node(
            "french_translation",
            lambda state: self.blog_node_obj.translation(
                {
                    **state,
                    "current_language": "french"
                }
            )
        )
        self.graph.add_node("route", self.blog_node_obj.route)

        # Add edges and conditional edges
        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", "route")

        self.graph.add_conditional_edges(
            "route",
            self.blog_node_obj.route_decision,
            {
                "hindi": "hindi_translation",
                "french": "french_translation",
            }
        )
        self.graph.add_edge("hindi_translation", END)
        self.graph.add_edge("french_translation", END)

        return self.graph

    def setup_graph(self, usecase):
        """
        Sets up the graph with the provided topic.
        """
        if usecase == 'topic':
            self.build_topic_graph()
        if usecase == 'language':
            self.build_language_graph()

        return self.graph.compile()


# Below code is for langsmith langgraph studio
llm = GroqLLM().get_llm()

# get graph
graph_builder = GraphBuilder(llm)
graph = graph_builder.build_language_graph().compile()
