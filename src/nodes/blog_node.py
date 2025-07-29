from src.states.blog_state import BlogState
from langchain_core.messages import HumanMessage
from src.states.blog_state import Blog


class BlogNode:
    """
    Represents a node in the blog generation process.
    """
    def __init__(self, llm):
        self.llm = llm

    def title_creation(self, state: BlogState):
        """
        Generates a title for the blog.
        """
        if "topic" in state and state["topic"]:
            prompt = """
                You are an expert blog content writer. Use markdown formatting.
                Generate blog title for the topic: {topic}. This title shoulde
                be creative and SEO friendly.
            """
            system_message = prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog": {"title": response.content}}

    def content_generation(self, state: BlogState):
        """
        Generates content for the blog based on the title.
        """
        if "topic" in state and state["topic"]:
            prompt = """
                You are an expert blog content writer. Use markdown formatting.
                Generate blog detailed content with detailed breakdown for the
                topic: {topic}. This content should be informative and
                engaging.
            """
            system_message = prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog": {
                "title": state["blog"]["title"],
                "content": response.content
            }}

    def translation(self, state: BlogState):
        """
        Translates the blog content into the specified language.
        """
        translation_prompt = """
            Translate the following blog content into {current_language}:
            - Maintain the original tone, style, and formatting.
            - Adapt cultural references and idioms to be appropriate for
              the {current_language}.

            ORIGINAL CONTENT:
            {blog_content}
        """
        blog_content = state["blog"]["content"]
        messages = [
            HumanMessage(
                translation_prompt.format(
                    current_language=state["current_language"],
                    blog_content=blog_content
                )
            )
        ]
        translation_content = self.llm.with_structured_output(Blog).invoke(
            messages
        )
        return {"blog": {"content": translation_content}}

    def route(self, state: BlogState):
        """
        Routes the state based on the current language.
        """
        return {
            "current_language": state['current_language']
        }

    def route_decision(self, state: BlogState):
        """
        Decides the next step based on the current language.
        """
        if state['current_language'] == 'hindi':
            return "hindi"
        elif state['current_language'] == 'french':
            return "french"
        else:
            return state['current_language']
