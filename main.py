from collections import defaultdict
from managers import ChatEnvironment, LLM, Assistant, Agent, FakeLLM
from NICOL.nicol_env import NICOLKnockingEnv
import argparse
import wandb
import numpy as np
import yaml


def log(prompt):
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "-a",
    "--sound_use_adjective",
    action="store_true",
    help="Use adjective instead of specific material feedback.",
)
parser.add_argument(
    # "-g", "--engine", default="text-davinci-003", help="OpenAI LLM engines."
    "-g",
    "--engine",
    default="Vicuna-13b",
    help="OpenAI LLM engines."
    # 'text-davinci-003', 'text-curie-001', 'text-babbage-001', "text-ada-001"
)
parser.add_argument(
    "-o",
    "--render_mode",
    default="None",
    help="Render mode. Optional: human, None, ...",
)
parser.add_argument("-e", "--episodes", default=30, type=int, help="Episodes to run.")
parser.add_argument(
    "-r", "--rounds", default=15, type=int, help="Maximum rounds in every episode."
)
parser.add_argument(
    "-m",
    "--mode",
    default="test",
    help="train or test, the source that sound comes from.",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Whether to set everything fixed for debugging.",
)
parser.add_argument(
    "-l",
    "--headless",
    action="store_true",
    help="Whether to run CoppeliaSim in headless mode, i.e. without GUI.",
)
parser.add_argument(
    "-t",
    "--max_tokens",
    default=128,
    help="Max tokens allowed for OpenAI GPT generations.",
)
parser.add_argument(
    "-T",
    "--temperature",
    default=0.7,
    help="Max tokens allowed for OpenAI GPT generations.",
)
parser.add_argument(
    "-p",
    "--plot_on",
    action="store_true",
    help="Plot on to show detections.",
)
parser.add_argument(
    "-w",
    "--use_wandb",
    action="store_true",
    help="Whether use wandb.ai to record.",
)
parser.add_argument(
    "-f",
    "--fake_llm",
    action="store_true",
    help="Whether use a fake (local designed for debugging) LLM.",
)
parser.add_argument(
    "-s",
    "--seed",
    default=-1,
    help="Seed used for randomizations.",
)
parser.add_argument(
    "--prompt_path",
    default="./prompts.txt",
    help="Few-shot prompts for in-context learning.",
)

args = parser.parse_args()
with open("config.yml", "r") as f:
    default_configs = yaml.safe_load(f)

for k, v in default_configs.items():
    setattr(args, k, v)
args.openai_api_base = args.engines[args.engine]["openai_api_base"]
args.openai_api_key = args.engines[args.engine]["openai_api_key"]

print(args)

run_name = (
    f"{args.engine}_adj" if args.sound_use_adjective else f"{args.engine}_material"
)
args.use_wandb and wandb.init(project="chatenv", config=args, name=run_name)


environment = ChatEnvironment(
    env_cls=NICOLKnockingEnv,
    mode=args.mode,
    headless=args.headless,
    debug=args.debug,
    render_mode=args.render_mode,
)
if args.fake_llm:
    LLM = FakeLLM
llm = LLM(
    engine=args.engine,
    openai_api_base=args.openai_api_base,
    openai_api_key=args.openai_api_key,
    prompt_path=args.prompt_path,
    max_tokens=int(args.max_tokens),
    temperature=int(args.temperature),
)
assistant = Assistant(sound_use_adjective=args.sound_use_adjective)
agent = Agent(assistant)


def reset_everything():
    environment.reset()
    llm.reset()
    agent.reset()
    vision = environment.render()
    prompt = ""
    command = ""
    caption = agent.vision(vision, plot_on=args.plot_on)
    instruction = environment.instruct()
    description = environment.instruct_with_caption(caption)
    skills = defaultdict(int)
    rewards = 0
    rud = 0
    return (
        vision,
        prompt,
        command,
        description,
        instruction,
        skills,
        rewards,
        False,
        rud,
    )


table_columns = [
    "Episode",
    "Vision",
    "Instruction",
    "Round",
    "Diversity",
    "Invalid",
    "Reward",
    "Conversation",
    "Explanation",
]

if args.use_wandb:
    major_table = wandb.Table(columns=table_columns)
for episode in range(args.episodes):
    if int(args.seed) == -1:
        if args.sound_use_adjective:
            seed = episode + 1000
        else:
            seed = episode
    else:
        seed = args.seed
    np.random.seed(seed)
    (
        vision,
        prompt,
        command,
        description,
        instruction,
        skills,
        rewards,
        done,
        rud,
    ) = reset_everything()
    while True:
        prompt = prompt + command + description
        command = llm.feed(prompt)
        print(command)
        description, explaination, skill, reward, done = agent.execute(
            environment, command
        )
        print(description)
        skills[skill] += 1
        rewards += reward
        rud += 1
        if done or rud >= args.rounds:
            prompt = prompt + command + description
            log(prompt)
            explaination = llm.feed(prompt)
            log(explaination)
            prompt = prompt + explaination + "done()"
            if args.use_wandb:
                table_data = (
                    episode,
                    wandb.Image(vision),
                    instruction,
                    rud,
                    len(skills),
                    agent.invalid_count,
                    rewards,
                    prompt,
                    explaination,
                )
                table = wandb.Table(columns=table_columns)
                table.add_data(*table_data)
                major_table.add_data(*table_data)
                wandb.log({f"ChatEnv {episode}": table})
            break

if args.use_wandb:
    wandb.log({f"ChatEnv Summary": major_table})
    wandb.finish()
