import json
import openai
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

openai.api_key = st.secrets["OPENAI_API_KEY"]


def get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period='1y').Close.iloc[-1])


def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])


def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])


def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1 + rs.iloc[-1])))


def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    ema_12 = data.ewm(span=12, adjust=False).mean()
    ema_26 = data.ewm(span=26, adjust=False).mean()

    MACD = ema_12 - ema_26
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal

    return f"{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}"


def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(8, 5))
    plt.plot(data.index, data.Close)
    plt.title(f'{ticker} Stock Price Over Past Year')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.savefig('stock.png')
    plt.close()


functions = [
    {
        'name': 'get_stock_price',
        'description': 'Get the current stock price of a company given its ticker symbol',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker symbol of the company (for example, AAPL for Apple)'
                }
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_SMA',
        'description': 'Calculate the Simple Moving Average of a company given its ticker symbol and window size',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker symbol of the company (for example, AAPL for Apple)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size of the SMA'
                }
            },
            'required': ['ticker', 'window']
        }
    },
    {
        'name': 'calculate_EMA',
        'description': 'Calculate the Exponential Moving Average of a company given its ticker symbol and window size',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker symbol of the company (for example, AAPL for Apple)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size of the EMA'
                }
            },
            'required': ['ticker', 'window']
        }
    },
    {
        'name': 'calculate_RSI',
        'description': 'Calculate the Relative Strength Index of a company given its ticker symbol',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker symbol of the company (for example, AAPL for Apple)'
                }
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_MACD',
        'description': 'Calculate the Moving Average Convergence Divergence of a company given its ticker symbol',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker symbol of the company (for example, AAPL for Apple)'
                }
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'plot_stock_price',
        'description': 'Plot the stock price of a company given its ticker symbol',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The ticker symbol of the company (for example, AAPL for Apple)'
                }
            },
            'required': ['ticker']
        }
    }
]

available_function = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price
}

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title(':grey[Stock Analysis Chatbot Assistant]')

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"], avatar='🧑‍💻'):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message(message["role"], avatar='🤖'):
            if message["content"] == 'stock.png':
                st.image('stock.png')
            else:
                st.markdown(message["content"])

if user_input := st.chat_input("Send a message"):
    try:
        st.session_state.messages.append({'role': 'user', 'content': f'{user_input}'})

        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(user_input)
        response = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo-0613',
            messages = st.session_state.messages,
            functions=functions,
            function_call='auto'
        )

        response_message = response.choices[0]['message']

        one_parameter_functions = ['get_stock_price', 'calculate_RSI', 'calculate_MACD', 'plot_stock_price']

        if response_message.get('function_call'):
            function_name = response_message['function_call']['name']
            function_args = json.loads(response_message['function_call']['arguments'])

            if function_name in one_parameter_functions:
                args_dict = {'ticker': function_args.get('ticker')}
            else:
                args_dict = {'ticker': function_args.get('ticker'), 'window': function_args.get('window')}
            
            function_to_call = available_function[function_name]
            function_response = function_to_call(**args_dict)

            if function_name == 'plot_stock_price':
                with st.chat_message("assistant", avatar='🤖'):
                    st.image('stock.png')
                st.session_state.messages.append(
                {
                    'role': 'assistant',
                    'content': 'stock.png'
                }
            )
            else:
                st.session_state.messages.append(
                    {
                        'role': 'function',
                        'name': function_name,
                        'content': f'{function_response}'
                    }
                )
                second_response = openai.ChatCompletion.create(
                    model = 'gpt-3.5-turbo-0613',
                    messages = st.session_state.messages
                )
                with st.chat_message("assistant", avatar='🤖'):
                    st.markdown(second_response.choices[0]['message']['content'])
                st.session_state.messages.append(
                    {
                        'role': 'assistant',
                        'content': second_response.choices[0]['message']['content']
                    }
                )
        else:
            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(response_message['content'])
            st.session_state.messages.append(
                {
                    'role': 'assistant',
                    'content': f'{response_message["content"]}'
                }
            )

    except Exception as e:
        raise e


