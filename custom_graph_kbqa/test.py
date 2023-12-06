from deeppavlov import build_model
model = build_model('kbqa_custom_graph.json', download=False)
question = "peripheral class game controllers supports which video game version?"
print(model([question]))