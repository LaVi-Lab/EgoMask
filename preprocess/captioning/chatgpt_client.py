import os, sys
sys.path.append(os.getcwd())
from openai import OpenAI
from abc import abstractclassmethod, ABC
import cv2

import sys
sys.path.append("")

from utils.const import PROJECT_PATH
from utils.cache import SqliteKeyValueStore, write_to_key_value_store
from utils.image_process_func import encode_image, draw_box,build_base64_frames_from_np


class closed_model_prompter(ABC):
    def __init__(self,
                cache_dir,
                cache_filename,
                model_name,
                table_name):

        self.cache_path = os.path.join(cache_dir,cache_filename)
        os.makedirs(cache_dir, exist_ok=True)

        self.model_name = model_name
        self.table_name = table_name

    @abstractclassmethod
    def compute(self, request):
        raise NotImplementedError

    def model_prompt(self, request, force=False):
        # cache exist; from cache
        with SqliteKeyValueStore(self.cache_path,self.table_name) as key_value_store:
            response = key_value_store.get(request)
            if response:
                cached = True
            else:
                cached = False

            if force or not cached:
                # Compute and commit the request/response to SQLite
                response = self.compute(request)
                if_call_api = True

                write_to_key_value_store(key_value_store, request, response)
            else:
                if_call_api = False
        return response, cached, if_call_api


class chatgpt_prompter(closed_model_prompter):

    def __init__(
        self,
        model_name:str,
        cache_filename: str,
        table_name: str = "prepare_data",
        api_key: str = "",
        cache_dir: str = os.path.join(PROJECT_PATH, "cache"),
    ):
        super().__init__(cache_dir, cache_filename, model_name, table_name)
        self.client = OpenAI(api_key=api_key)

    def prompt(self, 
            query_text, 
            base64_frames,
            temperature=0.0, 
            top_p=1.0,  
            **kwargs):

        request = self.build_request(
            query_text,
            base64_frames,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return self.model_prompt(request)

    # different prompter, different way to build request
    def build_request(self, query_text, base64_frames, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"},
                    }  # noqa: E501
                    for base64_frame in base64_frames
                ]
                + [
                    {
                        "type": "text",
                        "text": query_text,
                    }
                ],
            }
        ]
        request = {
            "model": self.model_name,
            "messages" : messages,
            **kwargs
        }
        return request

    # how to call api
    def compute(self, request):
        completion = self.client.chat.completions.create(
            **request
        )
        chat_response = completion
        response = chat_response.choices[0].message.content
        return response

def test_prompter(model_name):
    # cal gpt4o /o1-preview
    # with cache: if in cache
    prompter = chatgpt_prompter(model_name=model_name,cache_filename=f"{model_name}.sqlit")
    image_path = "sample/00023.jpg"

    prompt_template = (
        "Please help me generate object descriptions.\n"
        "These are {num_frames} frames from a video with a frame rate of {sampling_fps} FPS. Based on the object in the red bounding box and its object tag, please generate its caption, visual attributes and affordance description (if applicable).\n"  # noqa: E501
        + "Object tag: {object_tag}\n"
        + "Output should consist of three lines, separated by a newline:\n"
        + '1. A clear object caption, starting with "Object caption: ".\n'  # noqa: E501
        + '2. The corresponding visual attributes, starting with "Visual attributes: ".\n'  # noqa: E501
        + '3. A concrete affordance description, starting with "Affordance description: ".\n'  # noqa: E501
        + "**Restriction Policies**:\n"
        + "- Visual attributes characterize the objects in images. They can be OCR characters on the object, spatial relations to surrounding objects, action relations to surrounding objects, relative size compared to surrounding objects, color, geometry shape, material, texture pattern, motion or dynamics of objects, and so on.\n"
        + "- The attribute descriptions should clearly identify the object based on to avoid any ambiguity without referencing bounding boxes. Use the provided object tag selectively, as it may contain noise. \n"
        + "- The affordance description should should focus on the object's potential actions, interactions, or functions, describing how the object can be utilized or manipulated in a given context. Avoid generic statements and provide specific and practical insights into the object's affordances. \n"
    )

    generate_parameter = {
        "max_tokens": 1024,
    } if "gpt-4o" in model_name else {"max_completion_tokens": 1024, "temperature":1}

    response, cached, if_call_api = prompter.prompt(
        query_text=prompt_template.format(
            num_frames=1, sampling_fps=5, object_tag="car"
        ),
        base64_frames=[encode_image(image_path)],
        **generate_parameter,
    )
    print(response, cached, if_call_api)
    
def test_color(model_name):
    prompter = chatgpt_prompter(model_name=model_name,cache_filename=f"{model_name}.sqlit")
    image_path = "sample/00023.jpg"
    generate_parameter = {
        "max_tokens": 100,
    }
    
    np_frame = cv2.imread(image_path)
    np_frame = draw_box(
            np_frame,
            boxes=[[0,100,0,100]],
            color="blue",
            RGB_in=False,
            RGB_out=True,
        )

    base64_frames = build_base64_frames_from_np([np_frame])
    
    response, cached, if_call_api = prompter.prompt(
        query_text="What is the color of the car and the bounding box?",
        base64_frames=base64_frames,
        **generate_parameter,
    )
    print(response, cached, if_call_api)


if __name__ == "__main__":

    test_color(model_name="gpt-4o-2024-11-20")
