from deeppavlov import build_model
model = build_model('test.json', download=True)
question = 'which building has less than 9 floors?'
print(model([question]))