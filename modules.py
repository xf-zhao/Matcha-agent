import random
import openai
import numpy as np
from tqdm import tqdm
# from vild import ViLDDetector
import matplotlib.pyplot as plt
import imageio


###################### MODULES #################################################
DEFAULT_POSE = np.array([0.83443451, 0.76801813, -0.61692601, -0.13173638, 0.11043578])

SKILLS = {'knock_on':1,
                  'pick_up':2,
                  }

# imageio.imwrite('image.jpg', image)

image_path = "image.jpg"


class Assistant:
    def __init__(self, switch_vision=True, swith_sound=True) -> None:
        # switch_vision and self.vild = None
        pass

    def describe(self, image_path=None):
        # vilder.detect(image_path=image_path, category_names=category_names,plot_on=True)
        found_objects= {'yellow block': {'box': np.array([117.53889 ,  13.807173, 132.59515 ,  31.58398 ], dtype=np.float32),
  'coordinates': (125.0670166015625, 22.695575714111328),
  'normalized_coordinates': (0.6100830078125, 0.1107101254346894)},
 'red block on the left': {'box': np.array([74.45434 , 27.625505, 89.988945, 43.785732], dtype=np.float32),
  'coordinates': (82.22164154052734, 35.70561981201172),
  'normalized_coordinates': (0.4010811782464748, 0.17417375518054498)},
 'red block on the right': {'box': np.array([ 93.52788,  71.56357, 107.83365,  85.7643 ], dtype=np.float32),
  'coordinates': (100.6807632446289, 78.66393280029297),
  'normalized_coordinates': (0.4911256743640434, 0.38372650146484377)}}
        pass
        # return vild results

        objects = ', '.join([k for k in found_objects])
        caption = f"[{objects}]"

        category_names = [
    "blue block",
    "red block",
    # "green block",
    "orange block",
    "yellow block",
    "purple block",
    "pink block",
    # "cyan block",
    # "brown block",
    "blue bowl",
    "red bowl",
    "green bowl",
    "orange bowl",
    "yellow bowl",
    "purple bowl",
    "pink bowl",
    # "cyan bowl",
    "brown bowl",
]
        self.caption = caption
        self.found_objects = found_objects
        return caption

    def target_to_normalized_coordinates(self, target):
        invalid_target = False
        if target not in self.found_objects:
            target = random.choice([k for k in self.found_objects])
            invalid_target = True
        return self.found_objects[target]['normalized_coordinates'], target, invalid_target

    def feedback(self):
        return 

    def sound_classify(self, sounds):
        # TODO:
        material = 'metal'
        if len(sounds)>0:
            sound_like = f'It is made of {material}.' # or f'sounds {sharp}'
        else:
            sound_like = f'There is no obvious impact sound feedback.'
        return sound_like

    def touch_classify(self, touch_data=None):
        # TODO:
        feeling = 'soft'
        return feeling


DEFAULT_POSE = np.array([0.83443451, 0.76801813, -0.61692601, -0.13173638, 0.11043578]) # z and quatenion

class Agent:
    def __init__(self, assistant: Assistant) -> None:
        self.actions = {
                   'knock_on':self.knock_on,
                   'touch':self.touch,
                   'weigh':self.weigh,
                   'ask_human':self.ask_human,
                   'pick_up':self.pick_up,
                   'terminate':self.terminate,
                   }
        self.assistant = assistant
        self.action_indicator = '> robot.'
        self.invalid_count = 0

    def reset(self):
        self.invalid_count = 0

    def execute(self, environment, command):
        # command in a format: "AI: > robot.knock(blue block)"
        if self.action_indicator in command:
            infos = command.split(self.action_indicator)[1].split('(')
            skill = infos[0]
            target, *explanation = infos[1].split(')')
            # if len(explanation)>5:
            #     description, done = self.explain(explanation)
            skill_func = self.actions[skill]
            action, chosen_target, invalid_target = self._target_to_action(target, skill)
            if invalid_target:
                description_pre = f'Human: Invalid target. Randomly {skill} {chosen_target} instead.'
                self.invalid_count += 1
            if self.invalid_count > 3:
                skill_func = self.actions['pick_up']
                description_pre = f'Human: Too many invalid actions. Randomly pick up {chosen_target} instead.'
            description_post, done = skill_func(environment, action)
            description_post = description_post.replace('Human:', '')
            description = description_pre + description_post
        else:
            description, done = self.explain(command)
        return description, done

    def _target_to_action(self, target, skill):
        normmalized_coordinates, target, invalid_target = self.assistant.target_to_normalized_coordinates(target)
        return np.r_[normmalized_coordinates, DEFAULT_POSE, SKILLS[skill]], target, invalid_target 

    def knock_on(self, environment, action):
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        sounds = info['sounds']
        sound_like = self.assistant.sound_classify(sounds)
        description = f'Human: {sound_like}\nAI:'
        return description, False or terminate

    def pick_up(self,environment, action):
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        description = 'done()\n'
        return description, True or terminate

    def explain(self, command):
        print(command)
        description = 'Human: gon on.\nAI:'
        return description, False

    def touch(self, environment, target):
        # audio_path = env.step()
        feeling = self.assistant.touch_classify(target)
        description = f'Human: It feels {feeling}.\nAI:'
        return description, False

    def ask_human(self, target):
        answer = self.assistant.feedback()
        return answer, False

    def weigh(self, target):
        return '300g', False

    def terminate(self):
        return '', True


class Environment:
    def __init__(self, env_cls) -> None:
        env = env_cls(observation_mode='vision', headless=False, render_mode=None)
        # env.set_mode('train')
        env.set_mode('test')
        env.set_random(False)
        self.env = env

    def reset(self):
        try:
            obs = self.env.reset()
        except RuntimeError:
            print('Bad initialization, reset.')
        except Exception as e:
            raise e
        image = obs['front_rgb']
        instruction = self.env.info['instruction']
        self.instruction = f'Human: "{instruction}" in the scene that contains "XXX".\n'
        return 

    def instruct(self):
        return self.instruction

LLM_CACHE = {}
class LLM:

    def __init__(self,  openai_api_key_path, prompt_path=None, engine="text-ada-001") -> None:
        with open(openai_api_key_path) as f:
            openai_api_key = f.read()
            openai.api_key = openai_api_key
        if prompt_path is not None:
            with open(prompt_path) as f:
                prompt = f.read() + '\n'
        else:
            prompt = ''
        self._prompt = prompt
        self.engine=engine
        
    def reset(self):
        pass

    def feed(self, prompt=''):
        prompt = self._prompt + prompt
        response = self.gpt3_call(prompt)
        command = response['choices'][0]['text']
        # command = 'robot.knock(blue block)\nother things\n'
        if '\n' in command:
            command = command.split('\n')[0]
        command = command + '\n'
        return command

    def gpt3_call(self, prompt, max_tokens=64, temperature=0, logprobs=1, echo=False):
        id = tuple((self.engine, prompt, max_tokens, temperature, logprobs, echo))
        if id in LLM_CACHE.keys():
            print('cache hit, returning')
            response = LLM_CACHE[id]
        else:
            response = openai.Completion.create(engine=self.engine, 
                                                prompt=prompt, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature,
                                                logprobs=logprobs,
                                                echo=echo)
            LLM_CACHE[id] = response
        return response