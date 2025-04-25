import os
import dotenv
import asyncio 
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult


async def create_team(dataset_path):
    model = OpenAIChatCompletionClient(
        model='o3-mini',
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL')
    )

    developer = AssistantAgent(
        name='Developer',
        model_client=model,
        system_message=(
            f"You are a code developer agent. You will be given a csv file named '{dataset_path}' in your working directory and "
            "a question about it. You can develope python code to answer the question. "
            "You should always begin with your plan to answer the question. Then you "
            "should write the code to answer the question. "
            "You should always write the code in a code block with language(python) specified. "
            "If you need several code blocks, make sure to write down one at a time. "
            "You will be working with a code executor agent. Once you have a code block, "
            "you must wait for the code executor agent to execute the code. If the code "
            "is executed successfully, you can continue. "
            "Use pandas to answer the question if possible. If a library is not installed, "
            "use pip in a shell code block (with shell specified) to install it. "
            "If the user asks you to provide a plot, you should use matplotlib to plot it and "
            'you should save it as a png file and you should exactly say "GENERATED:<filename>" (like "GENERATED:plot.png") in your message (not in your code). '
            'in a new line after you are sure that the code executor has executed the code and generated the plot successfully.'
            "Once you have the code execution results, you should provide the final answer and "
            'after that, you should exactly say "TERMINATE" to terminate the conversation. '
            ""
        ),
    )

    # get the absolute path of the dataset
    host_absolute_path = os.path.abspath("./data")
    print(host_absolute_path)
    docker = DockerCommandLineCodeExecutor(
        work_dir='temp',
        image='amancevice/pandas:2.2.2',
        extra_volumes = {f'{host_absolute_path}': {'bind': '/mnt/data', 'mode': 'rw'}}
    )

    executor = CodeExecutorAgent(
        name='CodeExecutor',
        code_executor=docker
    )

    team = RoundRobinGroupChat(
        participants=[developer, executor],
        termination_condition=TextMentionTermination('TERMINATE'),
        max_turns=20,
    )
    return team, docker


async def orchestrate(team, docker, task):
    await docker.start()
    async for msg in team.run_stream(task=task):
        if isinstance(msg, TextMessage):
            print(message:=f'{msg.source}: {msg.content}')
            yield message
        elif isinstance(msg, TaskResult):
            print(message:=f'Stop reason: {msg.stop_reason}')
            yield message

    await docker.stop()


async def main():
    dataset_path = '/mnt/data/netflix_data.csv'
    task = f'My dataset is "{dataset_path}". What are the columns of the dataset.'
    team, docker = await create_team(dataset_path)
    async for msg in orchestrate(team, docker, task):
        pass


if __name__ == '__main__':
    # Read the API key and base url from the .env file
    dotenv.load_dotenv()
    asyncio.run(main())
