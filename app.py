import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Workout Planner 💪  ")
st.markdown("Create personalized workout plans tailored to your fitness goals and preferred duration, ensuring a comprehensive approach to achieving optimal results.")

input = st.text_input("Enter your primary fitness goal and indicate the duration you plan to commit to achieving this goal: ",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def workout_generation(input):
    generator_agent = Agent(
        role="FITNESS TRAINER expert",
        prompt_persona=f"Your task  is to DESIGN CUSTOMIZED WORKOUT SPLITS that cater to various user goals, including STRENGTH, ENDURANCE, FLEXIBILITY, BODY COMPOSITION, PERFORMANCE, FUNCTIONAL FITNESS, HEALTH AND WELLNESS, SPORT-SPECIFIC requirements, CONSISTENCY in training, and the DURATION of the workout program."
    )

    prompt = f"""
You are an Expert FITNESS TRAINER. Your task is to DESIGN CUSTOMIZED WORKOUT SPLITS that cater to various user goals, including STRENGTH, ENDURANCE, FLEXIBILITY, BODY COMPOSITION, PERFORMANCE, FUNCTIONAL FITNESS, HEALTH AND WELLNESS, SPORT-SPECIFIC requirements, CONSISTENCY in training, and the DURATION of the workout program.

Follow these steps to create a comprehensive workout plan:

1. IDENTIFY the PRIMARY GOAL of the user from the list provided: strength, endurance, flexibility, body composition, performance, functional fitness, health and wellness, or sport-specific training.

2. ASSESS any SECONDARY GOALS that may complement or enhance the primary goal.

3. DETERMINE the appropriate DURATION for each workout session and the overall program. If not specified by the user, DEFAULT to a 16-week program.

4. DEVELOP a weekly SPLIT that balances workout intensity and rest days according to the primary and secondary goals.

5. ENSURE each workout includes exercises tailored to meet the identified goals while also promoting overall fitness.

6. INCORPORATE progression and variety over time to keep workouts CHALLENGING and ENGAGING.

7. PROVIDE guidelines on how users can measure their PROGRESS against their goals throughout the duration of their program.

You MUST also consider factors such as current fitness level and any limitations or injuries.
      """

    generator_agent_task = Task(
        name="workout Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Generate!"):
    solution = workout_generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent Optimize your code. For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)