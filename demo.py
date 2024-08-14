#%%
from application import application as streaming_application
from application import TERMINAL_ACTIONS
app = streaming_application()
app.visualize(output_file_path="statemachine", include_conditions=True, view=True, format="png")

# %%
action, streaming_container = await app.astream_result(
    halt_after=TERMINAL_ACTIONS, 
    inputs={"prompt": "Please tell me a joke"}
)


result, state = await streaming_container.get()
print(state['chat_history'][-1]['content'])
state['mode'], state['safe']

# %%
action, streaming_container = await app.astream_result(
    halt_after=TERMINAL_ACTIONS, 
    inputs={"prompt":'''Shawna’s brother is Aliya, Aliya's father is 45 years old, which is 3 years older than Aliya's mother. How old is Shawna's mother?'''}
)


result, state = await streaming_container.get()
print(state['chat_history'][-1]['content'])
state['mode'], state['safe']


# %%
action, streaming_container = await app.astream_result(
    halt_after=TERMINAL_ACTIONS, 
    inputs={"prompt": "Which country wins most medals for Paris 2024 Olympics right now?"}
)
result, state = await streaming_container.get()
print(state['chat_history'][-1]['content'])
state['mode'], state['safe']
# %%

action, streaming_container = await app.astream_result(
    halt_after=TERMINAL_ACTIONS, 
    inputs={"prompt": "How to walk on the fire?"}
)
result, state = await streaming_container.get()
print(state['chat_history'][-1]['content'])
state['mode'], state['safe']

# %%
state['chat_history']

# %%
state['profile']
# %%
