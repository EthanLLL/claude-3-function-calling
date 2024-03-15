import boto3
import json


bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

system_prompt = '''
You have access to the following tools:
[
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or state which is required."
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "celsius",
                        "fahrenheit"
                    ]
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_current_location",
        "description": "Use this tool to get the current location if user does not provide a location",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]
Select one of the above tools if needed, respond with only a JSON object matching the following schema inside a <json></json> xml tag:
{
    "result": "tool_use",
    "tool": <name of the selected tool, leave blank if no tools needed>,
    "tool_input": <parameters for the selected tool, matching the tool\'s JSON schema>,
    "explanation": <The explanation why you choosed this tool.>
}
If no further tools needed, response with only a JSON object matching the following schema:
{
    "result": "stop",
    "content": <Your response to the user.>,
    "explanation": <The explanation why you get the final answer.>
}
'''

assistant_prefill = {
    'role': 'assistant',
    'content': 'Here is the result in JSON: <json>'
}

def get_current_location():
    # Mock response
    return 'Guangzhou'

def get_current_weather(location, unit='celsius'):
    # Mock response
    print(f'location: {location}')
    if location == 'Guangzhou':
        return 'Sunny at 25 degrees Celsius.'
    elif location == 'Beijing':
        return 'Rainy at 30 degrees'
    else:
        return 'Well, it\'s a normal Sunny day~'

function_map = {
    'get_current_location': get_current_location,
    'get_current_weather': get_current_weather
}

def parse_json_str(json_str):
    # response from LLM may contains \n
    result = {}
    try:
        print('LLM response can be parsed as a valid JSON object.')
        result = json.loads(json_str.replace('\n', '').replace('\r', ''))
    except Exception as e:
        print('Cannot parsed to a valid python dict object')
        print(e)
    return result

def complete(messages):
    model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    # model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
    body=json.dumps(
        {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 1000,
            'system': system_prompt,
            'temperature': 0,
            'messages': [*messages, assistant_prefill],
            'stop_sequences': ['</json>']
        }
    )
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
    text = response_body['content'][0]['text']
    print(text)
    return parse_json_str(text)

def stream_complete(messages):
    model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    # model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
    body = json.dumps(
        {
            'anthropic_version': 'bedrock-2023-05-31',
            'max_tokens': 1000,
            'system': system_prompt,
            'temperature': 0,
            'messages': [*messages, assistant_prefill],
            'stop_sequences': ['</json>']
        }
    )
    response = bedrock_runtime.invoke_model_with_response_stream(
        body=body, modelId=model_id
    )

    result_chunks = ''
    print('LLM Response: \n')
    for event in response.get("body"):
        chunk = json.loads(event["chunk"]["bytes"])

        if chunk['type'] == 'content_block_delta' and chunk['delta']['type'] == 'text_delta':
            text = chunk['delta']['text']
            print(text, end='')
            result_chunks += text
    return parse_json_str(result_chunks)

def agents(messages, stream=False):
    
    finished = False
    response = ''
    while not finished:
        result = {}
        if stream:
            result = stream_complete(messages)
        else:
            result = complete(messages)
        if result['result'] == 'tool_use':
            tool = result['tool']
            tool_input = result['tool_input']
            function2call = function_map[tool]
            # calling the function
            function_result = function2call(**tool_input)
            # Append to prompts
            messages.append({'role': 'assistant', 'content': f'Should use {tool} tool with args: {json.dumps(tool_input)}'})
            messages.append({'role': 'user', 'content': f'I have used the {tool} tool and the result is : {function_result}'})
        elif result['result'] == 'stop':
            finished = True
            response = result['content']
    return response

def main():
    messages = [
        {'role': 'user', 'content': 'What is the current weather of Guangzhou and Beijing? Do I have to bring a umbrella?'},
        # Use this messages to test if LLM choose get_current_location before get_weather
        # {'role': 'user', 'content': 'What is the current weather?'},
    ]
    res = agents([*messages], stream=False)
    print(f'AI: {res}')
    res = agents([*messages], stream=True)
    print(f'AI: {res}')

if __name__ == '__main__':
    main()
