import os
from openai import OpenAI
import json
import base64
import cv2
from colorama import Fore
import sys
from typing import Literal
import instructor


def get_api(api_key, api_url):
  os.environ['all_proxy'] = ""
  for k, v in os.environ.items():
    if 'proxy' in k.lower():
      os.environ.pop(k)

  openai_client = OpenAI(
    api_key=api_key,
    base_url=api_url,
  )

  client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)

  def inference(task, messages, model, temperature=None, top_p=None, top_k=None, setting: Literal["precise", "creative", "deterministic"] = None, response_model=None):
    assert isinstance(setting, str) or (temperature is not None and top_p is not None and top_k is not None), "Please set the setting or temperature, top_p, top_k"
    if setting is not None:
      if setting == "precise":
        temperature = 0.2
        top_p = 0.1
        top_k = 50
      # elif setting == "chat":
      #   temperature = 0.6
      #   top_p = 0.7
      #   top_k = 100
      elif setting == "creative":
        temperature = 0.6
        top_p = 0.7
        top_k = 100
      elif setting == "deterministic":
        temperature = 0.0
        top_p = 1.0
        top_k = 1
      else:
        raise ValueError(f"Unknown setting: {setting}")

    # ignore top_k
    chat_response = client.chat.completions.create(
      model=model,
      temperature=temperature,
      top_p=top_p,
      max_tokens=8192,
      response_model=response_model,
      # extra_body={
      #     "repetition_penalty": 1.05,
      # },
      messages=messages,
    )
    return chat_response
    # response_content = chat_response.choices[0].message.content
    # return response_content
    
  return inference


def get_response_fn(
  inference_fn,
  is_vl_model,
  model,
  max_retry=15,
  reraise=True,
  verbose=False,
  system_prompt="You are a helpful AI assistant",
  **kwargs
):
  def get_response(setting: Literal["precise", "creative", "deterministic"] = None, temperature=None, top_p=None, top_k=None):
    def _(task, user_prompt, image_np=None, check_fn=lambda x: x, response_model=None):

      # 构造 message
      if is_vl_model:
        if image_np is None:
          user_content = [
            {"type": "text", "text": user_prompt},
          ]
        else:
          success, img_str = cv2.imencode('.jpg', image_np, [cv2.IMWRITE_JPEG_QUALITY, 100])
          if not success:
            raise ValueError("Failed to encode image")
          b64_code = base64.b64encode(img_str.tobytes()).decode('utf-8')
          user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_code}"}}
          ]
      else:
        user_content = user_prompt

      messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
      ]
      
      # 调用 inference_fn
      response_content = inference_fn(task, messages, model, temperature, top_p, top_k, setting, response_model, **kwargs)

      # 打印 verbose 日志
      if verbose:
        f = open(verbose, 'a') if isinstance(verbose, str) else (sys.stdout if isinstance(verbose, bool) else verbose)
        print('+' * 10 + task + '+' * 10, file=f)
        print(Fore.CYAN + 'User prompt: ' + Fore.RESET, file=f)
        print(user_prompt, file=f)
        print(Fore.CYAN + 'Response: ' + Fore.RESET, file=f)
        print(response_content, file=f)
        print(Fore.CYAN + 'Response model: ' + Fore.RESET, file=f)
        print(response_model.model_json_schema(), file=f)
        # print(Fore.YELLOW + f'Token统计: 输入 {token_count} tokens, 输出 {response_token_count} tokens, 总计 {total_token_count} tokens' + Fore.RESET, file=f)
        print('\n'.join(['#' * 130] * 20), file=f)
        if f is not sys.stdout:
          f.close()

      js = json.loads(response_content.model_dump_json())
      js['raw'] = response_content
      js['input'] = {
        'user_prompt': user_prompt,
        'system_prompt': system_prompt,
      }
      if is_vl_model:
        js['input']['image_np'] = image_np
      
      # 添加token统计信息
      js['token_count'] = {
        'input': 0,
        'output': 0,
        'total': 0
      }

      return js

    return _

  return get_response
