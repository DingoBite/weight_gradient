import os
import pandas as pd
import torch
import math
import gradio as gr
from PIL import Image
from modules.processing import Processed, StableDiffusionProcessing, create_infotext
import modules.scripts as scripts
import re
from modules.processing import process_images, Processed
from modules.scripts import PostprocessImageArgs
import modules.images as images
from modules.shared import opts, cmd_opts


class EasingBase:
    limit = (0, 1)

    def __init__(self, start=0, end=1, duration=1):
        self.start = start
        self.end = end
        self.duration = duration

    @classmethod
    def func(cls, t):
        raise NotImplementedError

    def ease(self, alpha):
        t = self.limit[0] * (1 - alpha) + self.limit[1] * alpha
        t /= self.duration
        a = self.func(t)
        return self.end * a + self.start * (1 - a)

    def __call__(self, alpha):
        return self.ease(alpha)


class LinearInOut(EasingBase):
    def func(self, t):
        return t
    

class ExponentialEaseIn(EasingBase):
    def func(self, t):
        if t == 0:
            return 0
        return math.pow(2, 10 * (t - 1))


class ExponentialEaseOut(EasingBase):
    def func(self, t):
        if t == 1:
            return 1
        return 1 - math.pow(2, -10 * t)


class ExponentialEaseInOut(EasingBase):
    def func(self, t):
        if t == 0 or t == 1:
            return t

        if t < 0.5:
            return 0.5 * math.pow(2, (20 * t) - 10)
        return -0.5 * math.pow(2, (-20 * t) + 10) + 1
    

class CircularEaseIn(EasingBase):
    def func(self, t):
        return 1 - math.sqrt(1 - (t * t))


class CircularEaseOut(EasingBase):
    def func(self, t):
        return math.sqrt((2 - t) * t)


class CircularEaseInOut(EasingBase):
    def func(self, t):
        if t < 0.5:
            return 0.5 * (1 - math.sqrt(1 - 4 * (t * t)))
        return 0.5 * (math.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1)

dynamics = {
    'a': "Linear",
    'e': "Exponential",
    'ei': "ExponentialIn",
    'eo': "ExponentialOut",
    'c': "Circle",
    'ci': "CircleIn",
    'co' : "CircleOut"
    }

def make_first_easing_func(easing_code, start, end, duration):
    if easing_code == 'e':
        return ExponentialEaseInOut(start, end, duration)
    if easing_code == 'ei':
        return ExponentialEaseIn(start, end, duration)
    if easing_code == 'eo':
        return ExponentialEaseOut(start, end, duration)
    if easing_code == 'c':
        return CircularEaseInOut(start, end, duration)
    if easing_code == 'ci':
        return CircularEaseIn(start, end, duration)
    if easing_code == 'co':
        return CircularEaseOut(start, end, duration)         
    else:
        return LinearInOut(start, end, duration)
                
def make_second_easing_func(easing_code, start, end, duration):
    if easing_code == 'e':
        return ExponentialEaseInOut(start, end, duration)
    if easing_code == 'eo':
        return ExponentialEaseIn(start, end, duration)
    if easing_code == 'ei':
        return ExponentialEaseOut(start, end, duration)
    if easing_code == 'c':
        return CircularEaseInOut(start, end, duration)
    if easing_code == 'co':
        return CircularEaseIn(start, end, duration)
    if easing_code == 'ci':
        return CircularEaseOut(start, end, duration)  
    else:
        return LinearInOut(start, end, duration)

def is_valid_tokens(value):
    return value == '' or value.isspace() or value is None

def clamp(value, min_v, max_v):
    if value < min_v:
        return min_v
    elif value > max_v:
        return max_v
    return value


def preprocess_prompt(text, steps_count, is_log):
    try:
        pattern = r'([^{}]*)\s*\{\s*(\w+\s*:|\s*)\s*([^{}]+)\s*:\s*((?:[0-9]*\.[0-9]+|[0-9]+)\s*-\s*(?:[0-9]*\.[0-9]+|[0-9]+)|\s*)\s*:\s*([0-9]*\.[0-9]+|[0-9]+)\s*-\s*([0-9]*\.[0-9]+|[0-9]+)\s*(\s*-\s*([0-9]*\.[0-9]+|[0-9]+)|\s*)\s*(\s*:\s*\w+|\s*)\s*\}\s*([^{}]*)'
        prompt = text
        matches = re.finditer(pattern, text)
        
        prompt = ""
        i = 0
        for m in matches:
            i += 1
            left_tokens, process_mode, tokens, steps, start_weight, end_weight, probable_weight_group, probable_weight, dynamic_mode, right_tokens = m.groups()
            
            if not is_valid_tokens(left_tokens):
                prompt += left_tokens + " "
                
            pr_mode = ''
            is_process_mode = not is_valid_tokens(process_mode)
            if is_process_mode:
                pr_mode = f"Process code: {process_mode}. "
            
            if not is_valid_tokens(steps):
                split_steps = steps.split('-')
                start_step = float(split_steps[0])
                end_step = float(split_steps[1])
                if start_step < 1 or end_step < 1:
                    start_step = steps_count * clamp(start_step, 0 , 1)
                    if start_step == 0:
                        start_step = 1
                    end_step = steps_count * clamp(end_step, 0 , 1)
                start_step = int(start_step)
                end_step = int(end_step)
            else:
                start_step = 1
                end_step = steps_count
            
            pr_w_str = ''
            is_probable_weight = not is_valid_tokens(probable_weight_group)
            if is_probable_weight:
                probable_weight = probable_weight.replace("-", '')
                probable_weight = float(probable_weight)
                pr_w_str = f" - {probable_weight}"
            
            start_weight = float(start_weight)
            end_weight = float(end_weight)
            
            dm_str = ''
            is_dynamic_mode = not is_valid_tokens(dynamic_mode)
            dynamic_mode = dynamic_mode.replace(":", '').replace(" ", '')
            if is_dynamic_mode:
                dm_str += f"Dynamic mode: {dynamic_mode}. "
            
            if is_log:
                print(f"\n{pr_mode}{dm_str}Tokens: \"{tokens}\". Segment: {start_step} - {end_step}.  Weights: {start_weight} - {end_weight}{pr_w_str}\n")
            
            
            
            step_range = end_step - start_step
            if step_range < 1:
                print(f"Parse error:{text}")
                return text
            if is_probable_weight:
                int_range = int(step_range * 0.5)
                offset = step_range % 2
                first = make_first_easing_func(dynamic_mode, start_weight, end_weight, int_range)
                second = make_second_easing_func(dynamic_mode, end_weight, probable_weight, int_range + offset)
                def dynamic_func(i):
                    if i <= int_range:
                        return first(i)
                    else:
                        return second(i - int_range)
                dynamic = dynamic_func
            else:
                dynamic = make_first_easing_func(dynamic_mode, start_weight, end_weight, step_range)
            pr_str = ''
            log_pr = 2
            weight_pr = 4
            for i in range(0, step_range + 1):
                weight = dynamic(i)
                weight = round(weight, weight_pr)
                if abs(weight - start_step) < float(f"1e-{weight_pr}"):
                    weight = start_step
                if i == step_range:
                    pr_str += f"{round(weight, log_pr)}\n"
                else:
                    pr_str += f"{round(weight, log_pr)} | "
                if weight == 0:
                    continue
                
                if i == 0 and start_step == 1:
                    prompt += f"([{tokens}:: {start_step}] : {weight})"
                elif i == step_range:
                    prompt += f"([{tokens}: {start_step + i}]:{weight})"
                else:
                    prompt += f"([[{tokens}: {start_step + i - 1}] :: {start_step + i}]:{weight})"

            if not is_valid_tokens(right_tokens):
                prompt += " " + right_tokens
            if is_log:
                print(pr_str)
            
        if i == 0:
            return text
        
        return prompt
    except Exception as e:
        # print(e)
        return text


def plot_dynamic(mode):
    easing = make_first_easing_func(mode, 0, 1, 20)
    X = list(range(20))
    Y = [easing(x) for x in X]
    df = pd.DataFrame({'time':X, 'weight':Y})
    return df

class Script(scripts.Script):

    def title(self):
        return "Weight Gradient"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Weight Gradient", open=False):                                                          
            with gr.Row(equal_height=True):
                enabled_checkbox = gr.Checkbox(label="Enable", value=False)
                log_in_console_checkbox = gr.Checkbox(label="Log in console", value=False)
                figure_braces_exif_checkbox = gr.Checkbox(label="FigureBracesEXIF", value=False)
            with gr.Accordion("Documentation", open=False):
                with gr.Tabs():
                    with gr.TabItem(label="Info", id=1):
                        gr.HighlightedText(label="Form",
                            value=[
                                ("{", None), 
                                ("redhead girl", "Required. Tokens"),
                                (":", None), 
                                ("start - end","Steps where weight changes"),
                                (":", None), 
                                ("start - end","Required. Weight start to end range"),
                                ("- return", "Weight move to return weight"),
                                (": mode", "Gradient mode (e, ei, eo, c, ci, co)"),
                                ("}", None),
                                ],
                            combine_adjacent=True,
                            show_legend=True).style(color_map={
                                "Required. Tokens": "red",
                                "Required. Weight start to end range": "red",
                                "Steps where weight changes": "yellow",
                                "Weight move to return weight": "yellow",
                                "Gradient mode (e, ei, eo, c, ci, co)": "yellow",
                                })
                        
                        gr.HighlightedText(label="Examples",
                            value=[
                                ("{dog : 0.1 - 0.3 : 1 - 0}", "Linear decreasing from 1 to 0 in 20% of steps (10% - 30%)"), 
                                ("{cat : 1 - 15 : 1 - 0 - 1 : e}", "Exponencial decreasing from 1 to 0 in 8 steps then increasing from 0 to 1 in 7 steps"), 
                                ("{друже :: 0 - 1 : c}", "Circle increasing from 0 to 1 at every step"), 
                                ],
                            combine_adjacent=True,
                            show_legend=True)
                    with gr.TabItem(label="Modes Hint", id=2):
                        for d in sorted(dynamics):
                            name = f"{dynamics[d]}: {d}"
                            df_e = plot_dynamic(d)
                            gr.LinePlot(label=dynamics[d], value=df_e, x="time", y="weight", title=d,
                                        width=200,
                                        height=100,
                                        interactive=False)
            
        return [enabled_checkbox, log_in_console_checkbox, figure_braces_exif_checkbox]
    
    def process(self, p : StableDiffusionProcessing, enabled_checkbox, log_in_console_checkbox, figure_braces_exif_checkbox):    
        if not enabled_checkbox:
            return
        
        if figure_braces_exif_checkbox:
            self.prompt = p.prompt
            self.all_prompts = list.copy(p.all_prompts)
            self.all_negative_prompts = list.copy(p.all_negative_prompts)
        prev_prompt = p.prompt
        p.prompt = preprocess_prompt(p.prompt, p.steps, log_in_console_checkbox)
        self.is_proc = prev_prompt != p.prompt
        p.do_not_save_samples = self.is_proc and figure_braces_exif_checkbox
        
        for i in range(len(p.all_prompts)):
            if is_valid_tokens(p.all_prompts[i]):
                continue
            p.all_prompts[i] = preprocess_prompt(p.all_prompts[i], p.steps, False)
        for i in range(len(p.all_negative_prompts)):
            if is_valid_tokens(p.all_negative_prompts[i]):
                continue
            p.all_negative_prompts[i] = preprocess_prompt(p.all_negative_prompts[i], p.steps, log_in_console_checkbox)
    
    def postprocess_image(self, p: StableDiffusionProcessing, pp: PostprocessImageArgs, enabled_checkbox, _, figure_braces_exif_checkbox):
        if not enabled_checkbox or not self.is_proc:
            return
        
        if figure_braces_exif_checkbox:
            img : Image.Image = pp.image
            p.all_negative_prompts = self.all_negative_prompts
            custom_exif = create_infotext(p, self.all_prompts, p.all_seeds, p.all_subseeds, '')
            images.save_image(img, p.outpath_samples, "", p.seed, self.prompt, opts.samples_format, info=custom_exif, p=p, suffix="_figbr")
    