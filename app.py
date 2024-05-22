from flask import Flask, render_template, request, jsonify, session
import requests
from requests.auth import HTTPBasicAuth
from xml.etree import ElementTree as ET
from datetime import datetime
import json
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from flask_session import Session

app = Flask(__name__)
app.secret_key = ""  # Change this to a random secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize the Mistral client
api_key = ""
model = "mistral-large-latest"
client = MistralClient(api_key=api_key)

# Function to retrieve total amount
def retrieve_total_amount(sa_id: str) -> str:
    username = 'INTUSER'
    password = 'INTUSER00'
    url = 'https://193.123.64.145:4443/ouaf/webservices/CM-SABAL?WSDL'
    headers = {
        'Content-Type': 'text/xml',
        'SOAPAction': 'getPayoffBalance'
    }
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    body = f"""
    <soapenv:Envelope xmlns:cm="http://ouaf.oracle.com/webservices/cm/CM-SABAL" xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/">
    <soapenv:Header>
        <wsse:Security soapenv:mustUnderstand="1" xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">
            <wsu:Timestamp wsu:Id="TS-EC6443AE3A24E2D1AB17156877864664">
                <wsu:Created>{timestamp}</wsu:Created>
                <wsu:Expires>{timestamp}</wsu:Expires>
            </wsu:Timestamp>
        </wsse:Security>
    </soapenv:Header>
    <soapenv:Body>
        <cm:getPayoffBalance>
            <!--Optional:-->
            <cm:saId>6101569569</cm:saId>
        </cm:getPayoffBalance>
    </soapenv:Body>
    </soapenv:Envelope>
    """
    response = requests.post(url, headers=headers, data=body, auth=HTTPBasicAuth(username, password), verify=False)

    if response.status_code == 200:
        root = ET.fromstring(response.content)
        namespace = {'ouaf': 'http://ouaf.oracle.com/webservices/cm/CM-SABAL'}
        total_amount = root.find('.//ouaf:totalAmount', namespace).text
        current_amount = root.find('.//ouaf:currentAmount', namespace).text
        print("Total Amount:", total_amount)
        print("Current Amount:", current_amount)
        return json.dumps({"total_amount": total_amount})
    else:
        return json.dumps({"error": "Failed to retrieve data. Status code: " + str(response.status_code)})

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_total_amount",
            "description": "Get the total amount to be paid based on account id",
            "parameters": {
                "type": "object",
                "properties": {
                    "sa_id": {
                        "type": "string",
                        "description": "The service agreement ID.",
                    }
                },
                "required": ["sa_id"],
            },
        },
    }
]

names_to_functions = {
    'retrieve_total_amount': retrieve_total_amount
}

@app.route('/')
def home():
    session.clear()
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data.get('message')
    
    # Retrieve or initialize the conversation history from the session
    conversation_history = session.get('conversation_history', [])
    
    # Add the user's message to the conversation history
    conversation_history.append(ChatMessage(role="user", content=message))
    print("1:",conversation_history)
    # Send the message to the Mistral model
    response = client.chat(
        model=model,
        messages=conversation_history,
        tools=tools,
        tool_choice="auto"
    )
    print(response)
    
    # Process the Mistral response
    assistant_message = response.choices[0].message
    conversation_history.append(assistant_message)
    print("2:",conversation_history)
    # Initialize function_params to avoid UnboundLocalError
    function_params = {}
    
    # Check if the assistant needs additional information
    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        function_params = json.loads(tool_call.function.arguments)
        
        if 'sa_id' not in function_params or not function_params['sa_id']:
            answer = "To provide you with the total amount to be paid, I need your service agreement ID."
            return jsonify({"answer": answer})
        
        # Call the function to retrieve the total amount
        function_result = names_to_functions[function_name](**function_params)
        print(function_result)
        conversation_history.append(ChatMessage(role="tool", name=function_name, content=function_result))
        print("3:",conversation_history)
        # Get the final response from the Mistral model
        response2 = client.chat(
            model=model,
            messages=conversation_history
        )
        answer = response2.choices[0].message.content
    else:
        answer = assistant_message.content
    
    # Save the updated conversation history to the session
    session['conversation_history'] = conversation_history
    
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
