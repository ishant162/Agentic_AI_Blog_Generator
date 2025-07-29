from pydantic import BaseModel, Field
from typing import TypedDict


class Blog(BaseModel):
    """
    Represents a blog post.
    """
    title: str = Field(..., description="The title of the blog post")
    content: str = Field(..., description="The main content of the blog post")


class BlogState(TypedDict):
    """
    Represents the state of the blog generation process.
    """
    topic: str
    blog: Blog
    current_language: str
