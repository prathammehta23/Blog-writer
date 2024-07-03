# Warning control
import warnings
warnings.filterwarnings('ignore')
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
import os

llm = ChatOpenAI(
    model="crewai-mistral",
    openai_api_key="NA",
    base_url="http://localhost:11434/v1"
)

topic = input("What is your topic? ")
planner = Agent(
    role="Content Planner",
    goal=f"Plan engaging and factually accurate content on {topic}",
    backstory="You're writing a letter to your father "
              f"about the topic: {topic}."
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
	verbose=True,
    llm=llm
)

writer = Agent(
    role="Content Writer",
    goal="Write an emotional heart-touching "
         f"letter about the topic: {topic}",
    backstory="You're working on a writing "
              f"a sweet letter about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. ",

    allow_delegation=False,
    verbose=True,
    llm=llm
)

editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            f"and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources.\n"
        "5. It must be include between 2000 words"
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.",
    agent=planner,
)


write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            f"blog post on {topic}.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
        "6. There should be at least 2000 words in the whole letter"
    ),
    expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs having 2000 words each.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs with 2000 words.",
    agent=editor
)
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)

result = crew.kickoff()
from IPython.display import Markdown
Markdown(result)
