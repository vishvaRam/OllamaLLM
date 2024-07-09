from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        - You are a professional at summarizing meeting contents into minutes. Your task is to convert the transcribed text provided into concise and structured meeting minutes. 
        - Please create 4 separate sections with the titles as below :
        1. Date and time of the meeting.
        2. Attendees. 
            - List of attendees.
        3. Agenda.
            - Key discussion points and decisions made during the meeting.
        4. Action items.
            - Actionable items, including assigned owners and due dates.
        - If the transcription does not provide enough information for any of the above sections, state explicitly: "No information identified from transcribed texts."
            for this conversation
        - Start your answer with 1. Date and time of the meeting. Do not add any text before that.
        """),
        ("user", "{user_input}")
    ]
)

conv = """
Alice: Good morning, everyone. Thanks for making it to the meeting. We have a few key points to cover today. First, I’d like to get an update on the development progress. Bob, how are things going on your end?

Bob: Good morning, Alice. The development is on track. We’ve just finished the beta version of the new feature, and it’s ready for initial testing. However, we’ve encountered a few minor bugs that need fixing, which might take an extra couple of days.

Alice: That sounds promising. Do you anticipate any major roadblocks that could delay the launch?

Bob: No major roadblocks, but we need to ensure thorough testing to catch any issues before the release. I’ve also allocated extra resources to expedite the debugging process.

Alice: Great to hear. Carol, how are the marketing preparations coming along?

Carol: Good morning, everyone. We’re gearing up for the product launch. The marketing campaign is in its final stages. We have the social media strategy ready, and the email campaigns are scheduled. I’m working closely with the design team to finalize the promotional materials.

Alice: Excellent. Have you planned any special promotions or events for the launch?

Carol: Yes, we’re planning a live demo event next week, and we’ve lined up a few influencers to help spread the word. Additionally, we’ll be running a limited-time discount for early adopters to generate buzz and encourage quick sign-ups.

Alice: That sounds comprehensive. Bob, do you think the development team can support the demo event next week in case we run into any technical issues?

Bob: Absolutely. I’ll make sure we have a couple of team members on standby during the event to handle any technical problems that might arise.

Alice: Perfect. Lastly, let’s discuss the timeline. Assuming we fix the bugs and the testing goes smoothly, are we still on track for the launch date?

Bob: Yes, if everything goes as planned, we should be able to launch on the scheduled date. We’re pushing hard to make sure there are no delays.

Carol: From a marketing perspective, we’re all set to align with the launch date. Everything is synced up to go live as soon as we get the green light from the development team.

Alice: Wonderful. Let’s aim to reconvene later this week for a final check-in before the launch. Thanks for the updates, everyone. Let’s keep up the good work.

Bob: Sounds good. I’ll keep you posted on the progress.

Carol: Thanks, Alice. Looking forward to a successful launch.

Alice: Great. Meeting adjourned. Have a productive day, everyone!
"""

# LLAma3 LLm
print("Ollama running.....")
llm = Ollama(model="llama3", temperature=0.6, top_p=0.8)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

res = chain.invoke({"user_input": conv})

print(res)

# # Gemma 
# gemma_llm = Ollama(model="gemma:2b")

# gemma_chain = prompt | gemma_llm | StrOutputParser()

# gemma_chain.invoke({"question":info+conv})
