import os
from os.path import join
import pandas as pd
from openai import OpenAI
import math
import argparse
import json
import re
from sklearn.model_selection import train_test_split

df_path_in = join('data','consolodated_text_cleaned.csv')
df_path_out = join('data','consolodated_text.csv')
df_path_out_clean = join('data','consolodated_text_cleaned.csv')
MAX_TEXT_LEN = 100

#os.environ['OPENAI_API_KEY'] = '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'

#df.to_csv(df_path, index=False)

def split_block(output, delim, logging=False):
    if delim in output: 
        output = output.split(delim)[-1]
        if logging:
            print(f'0.{output}')
        if not isinstance(output, str):
            output = output[0]
    return output

def clean_output(output, logging=False):
    try:
        if '{' in output: #likely json object
            if 'input:' in output or '\"input\":' in output:
                output = json.loads(output)['input']
            elif 'instruction:' in output or '\"instruction\":' in output:
                output = json.loads(output)['instruction']
            elif 'text:' in output or '\"text\":' in output:
                output = json.loads(output)['text']
    except:
        print(f'Errored on [{output}]')

    output = split_block(output, '{ "input": { "task":')
    print(f'1. {output}') if logging else 1 == 1
    output = split_block(output, '}, "output": ')
    print(f'2. {output}') if logging else 1 == 1
    output = output.replace('[assistant]', '')
    output = output.replace('max_tokens:', '')
    print(f'3. {output}') if logging else 1 == 1
    output = output.replace('"', '')
    output = output.replace('#~#', '')
    print(f'4. {output}') if logging else 1 == 1
    output = output.replace('[', '')
    print(f'5. {output}') if logging else 1 == 1
    output = output.replace(']', '')
    print(f'6. {output}') if logging else 1 == 1
    output = output.replace('{', '')
    print(f'7. {output}') if logging else 1 == 1
    output = output.replace('}', '')
    print(f'8. {output}') if logging else 1 == 1
    output = output.replace('(', '')
    print(f'9. {output}') if logging else 1 == 1
    output = output.replace(')', '')
    print(f'10. {output}') if logging else 1 == 1
    output = output.replace('\\', '')
    print(f'11. {output}') if logging else 1 == 1
    output = output.replace('\n', '')
    print(f'12. {output}') if logging else 1 == 1
    output = output.replace('*', '')
    print(f'13. {output}') if logging else 1 == 1
    #output = split_block(output, '\n') # in case API returns "You can instruct Llama-2 as follows:\n prompt"
    output = split_block(output, ':') # in case API returns "You can instruct Llama-2 as follows: prompt"
    print(f'14. {output}') if logging else 1 == 1
    output = split_block(output, 'Llama-2,') # in case of: "Llama-2, generate a satirical ..."
    print(f'15. {output}') if logging else 1 == 1
    output = split_block(output, 'Instruct Llama-2 to')
    print(f'16. {output}') if logging else 1 == 1
    output = split_block(output, 'You can instruct Llama-2 to')
    print(f'17. {output}') if logging else 1 == 1
    output = split_block(output, 'Llama-2 to')
    print(f'18. {output}') if logging else 1 == 1
    output = output.strip()
    return output


#input = '''{
#  "input": "Generate a satirical news headline in the style of exaggerated online dating advertisements",
#  "max_tokens": 15
#}'''
#print(clean_output(input, logging=True))


def get_api_prompt(df_path):
    df = pd.read_csv(df_path)
    df = df.rename(columns={'instruction': 'input', 'text': 'output'})
    #df = df.sample(frac=1).reset_index(drop=True)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", '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'))
    df_size = df["output"].size
    cnt = 0
    #found_first = False
    made_changes = False
    for ind in df.index:
        if cnt % 5 == 0 and made_changes:
            df.to_csv(df_path, index=False)
            print(f'Saving dataframe to {df_path}')
            made_changes = False
        instruction = df['input'][ind]
        text = df['output'][ind]

        #print(f'instruction: {}, text: {text}')
        if isinstance(instruction, float) and math.isnan(instruction):
            #if not found_first:
            #    print(f'First instruction: {instruction}')
            #    print(f'Started talking to API on {ind}/{df_size}')
            #    cnt = ind
            #    found_first = True

            if len(text) > MAX_TEXT_LEN:
                df = df.drop(index=ind)
                continue

            completion = ''
            try:
                completion = client.chat.completions.create(model="gpt-4", 
                    messages=[{"role": "system", 
                                "content": "You are generating a Llama-2 instruction to create satirical news headlines."}, 
                                {"role": "user", "content": f"Given this headline, [{text}], instruct llama-2 concisely to create such a headline without any newlines, introduction text, or greeting Llama-2?"}])
            except Exception as e: 
                # error received from openAI API, save progress
                print(f'Errored on {ind}/{df_size}')
                df.to_csv(df_path, index=False)
                break    
                
            predicted_prompt = completion.choices[0].message.content
            cleaned_prompt = clean_output(predicted_prompt)
            print(f'{ind}/{df_size}. {cleaned_prompt} -> {text}')
            made_changes = True
            if len(cleaned_prompt) <= 10:
                print(f'\n[{predicted_prompt}] ::: [{cleaned_prompt}]')
                break
                df = df.drop(index=ind)
                print(f'Dropped')
                continue
            print(f'\n\n')

            df['input'][ind] = cleaned_prompt
            cnt += 1
    df.to_csv(df_path, index=False)

def clean_df(df_path_in, df_path_out):
    df = pd.read_csv(df_path_in)
    cnt = 0
    for ind in df.index:
        instruction = df['input'][ind]
        output = df['output'][ind]
        if isinstance(instruction, float) and math.isnan(instruction) or isinstance(output, float) and math.isnan(output):
            continue

        cleaned = clean_output(instruction)
        cleaned_output = clean_output(output)
        if cleaned != instruction or cleaned_output != output:
            cnt += 1
            df['input'][ind] = cleaned
            df['output'][ind] = cleaned_output
    df.to_csv(df_path_out, index=False)
    print(f'Edited {cnt} instructions.')

def setup_datasets(df_path_in, df_train_path_out, df_eval_path_out, test_size=0.05):
    df = pd.read_csv(df_path_in)
    df = df[~df.isnull().any(axis=1)] # remove all rows without a prompt
    df = df.rename(columns={'input': 'prompt', 'output': 'completion'})
    df['prompt'] = 'question: ' + df['prompt'] #+ ' context: You are creating satirical news headlines.'

    train_df, eval_df = train_test_split(df, test_size=test_size)

    print(f'Train contains {train_df["prompt"].size} entries. Eval contains {eval_df["prompt"].size} entries.')
    train_df.to_json(df_train_path_out, orient='records', lines=True, force_ascii=False)
    #eval_df = eval_df.drop('completion', axis=1)
    eval_df.to_json(df_eval_path_out, orient='records', lines=True, force_ascii=False)

parser = argparse.ArgumentParser(description='Example script with a -c flag.')
parser.add_argument('-c', '--clean', action='store_true', help='Clean api instruction prompts.')
parser.add_argument('-a', '--api', action='store_true', help='Call api to continue generating prompts.')
parser.add_argument('-s', '--splits', action='store_true', help='Make train/eval splits.')

args = parser.parse_args()
if args.api:
    get_api_prompt(df_path_out)
if args.clean:
    clean_df(df_path_out, join(df_path_out_clean))
if args.splits:
    setup_datasets(df_path_out_clean, join('data', f'finetune_train.jsonl'), join('data', f'finetune_eval.jsonl'))