import boto3
import json


bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

system_prompt = '''
You have access to the following tools:
[
    {
        "name": "do_pairwise_arithmetic",
        "description": "Calculator function for doing basic arithmetic. \nSupports addition, subtraction, multiplication",
        "parameters": {
            "type": "object",
            "properties": {
                "first_operand": {"type": "string", "description": "First operand (before the operator)"},
                "second_operand": {"type": "string", "description": "Second operand (after the operator)"},
                "operator": {"type": "string", "description": "The operation to perform. Must be either +, -, *, or /"}
            },
            "required": ["first_operand", "second_operand", "operator"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get local weather information based on longitude and latitude.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "string", "description": "refers to the angular distance north or south from the earth's equator measured in degrees."},
                "longitude": {"type": "string", "description": "Longitude refers to the geographic coordinate that specifies the east-west position of a point on the Earth's surface."},
            },
            "required": ["latitude", "longitude"]
        }
    },
    {
        "name": "get_lat_long",
        "description": "Get the latitude and longitude of the location.",
        "parameters": {
            "type": "object",
            "properties": {
                "place": {"type": "string", "description": "The location that needs to obtain longitude and latitude."}
            },
            "required": ["place"]
        }
    } 
]

Please think step by step. If needed, please select one or more tools whose parameters have already been provided. Respond with only a JSON object matching the following schema inside a <json></json> xml tag:
{
    "result": "tool_use",
    "tool_calls": [
        {
            "tool": "<name of the selected tool, leave blank if no tools needed>",
            "tool_input": <parameters for the selected tool, matching the tool\'s JSON schema>
        }
    ]
    "explanation": "<The explanation why you choosed these tools.>"
}

If no further tools needed, response with only a JSON object matching the following schema:
{
    "result": "stop",
    "content": "<Your response to the user.>",
    "explanation": "<The explanation why you get the final answer.>"
}
'''

assistant_prefill = {
    'role': 'assistant',
    'content': 'Here is the result in JSON: <json>'
}
def do_pairwise_arithmetic(first_operand, second_operand, operator):
    print(f'{first_operand} {operator} {second_operand}')
    return first_operand + second_operand

def get_lat_long(place):
    # Mock response
    print(place)
    return f'Location: {place} latitude: 31.411578687640844, longitude: 121.49308204650879'

def get_weather(latitude, longitude):
    # Mock response
    print(latitude)
    print(longitude)
    return 'It\'s a normal sunny day~'

function_map = {
    'get_weather': get_weather,
    'get_lat_long': get_lat_long,
    'do_pairwise_arithmetic': do_pairwise_arithmetic
}

def parse_json_str(json_str):
    # response from LLM may contains \n
    result = {}
    try:
        result = json.loads(json_str.replace('\n', '').replace('\r', ''))
        print('LLM response can be parsed as a valid JSON object.')
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
    # print(response_body)
    text = response_body['content'][0]['text']
    print(text)
    return parse_json_str(text)

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
            assistant_msg = ''
            function_msg = ''
            for t in result['tool_calls']:
                tool = t['tool']
                tool_input = t['tool_input']
                assistant_msg += f'Should use {tool} tool with args: {json.dumps(tool_input)}\n'
                function2call = function_map[tool]
                # calling the function
                function_result = function2call(**tool_input)
                # Append to prompts
                function_msg += f'I have used the {tool} tool with args: {json.dumps(tool_input)} and the result is : {function_result}\n'
            messages.append({'role': 'assistant', 'content': assistant_msg})
            messages.append({'role': 'user', 'content': function_msg})

        elif result['result'] == 'stop':
            finished = True
            response = result['content']
    return response

def main():
    messages = [
        # {'role': 'user', 'content': 'What is the current weather of Guangzhou, Shanghai and Beijing? Do I have to bring an umbrella?'},
        {'role': 'user', 'content': 'What is the current weather of Guangzhou and Beijing? Do I have to bring an umbrella? And what is 1 + 1?'},
        # {'role': 'user', 'content': '今日の広州と北京の天気はどうですか?外出時に傘が必要ですか?'},
        # {'role': 'user', 'content': '请问今天广州和北京的天气如何？出门需要带伞吗'},
        # Use this messages to test if LLM choose get_current_location before get_weather
        # {'role': 'user', 'content': 'What is the current weather?'},
        # Use these messages to test if tool is unnecessary.
        # {'role': 'user', 'content': 'What is the current timestamp?'}
        # {'role': 'user', 'content': 'Hi How are you?'}
    ]
    print(messages)
    res = agents([*messages], stream=False)
    print(f'AI: {res}')
    # res = agents([*messages], stream=True)
    # print(f'AI: {res}')

if __name__ == '__main__':
    main()
